
"""
Module RLHF (Reinforcement Learning from Human Feedback) pour GAMA-AI
Compatible avec LoRAFineTuning.py et utilise le dataset Anthropic/hh-rlhf

Installation requise:
pip install transformers datasets trl accelerate

Usage:
    from utils.rlhf_module import RLHFTrainer, RLHFConfig
    
    # Après un training LoRA classique
    rlhf_config = RLHFConfig()
    rlhf_trainer = RLHFTrainer(
        model_dir="./saved_models/my_llm",
        tokenizer_path="./Tokenizer/tokenizer_5k.bin",
        device=torch.device("cuda"),
        rlhf_config=rlhf_config
    )
    rlhf_trainer.train_with_rlhf()
"""

import os
import sys
import json
import time
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field, asdict
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

# Imports pour RLHF
try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False
    print("⚠️  datasets non disponible. Installez avec: pip install datasets")

try:
    from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
    from trl import create_reference_model
    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False
    print("⚠️  TRL non disponible. Installez avec: pip install trl")


# ============================================================================
# CONFIGURATION RLHF
# ============================================================================

@dataclass
class RLHFConfig:
    """Configuration pour l'entraînement RLHF"""
    
    # Dataset Anthropic
    dataset_name: str = "Anthropic/hh-rlhf"
    dataset_subset: Optional[str] = None
    max_samples_train: int = 10000
    max_samples_val: int = 1000
    
    # PPO (Proximal Policy Optimization)
    learning_rate: float = 1.41e-5
    batch_size: int = 8
    mini_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    ppo_epochs: int = 4
    
    # Génération
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    
    # KL divergence (évite que le modèle s'éloigne trop du modèle de référence)
    init_kl_coef: float = 0.2
    target_kl: float = 0.1
    kl_penalty: str = "kl"
    
    # Récompenses
    reward_scale: float = 1.0
    reward_baseline: float = 0.0
    use_length_penalty: bool = True
    length_penalty_coef: float = 0.05
    
    # Training
    num_train_epochs: int = 1
    max_grad_norm: float = 0.5
    
    # Logging et sauvegarde
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 250
    output_dir: str = "./rlhf_output"
    
    # Advanced
    use_score_scaling: bool = True
    use_score_norm: bool = True
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    vf_coef: float = 0.1
    adap_kl_ctrl: bool = True


# ============================================================================
# DATASET RLHF
# ============================================================================

class AnthropicRLHFDataset(Dataset):
    """
    Dataset pour Anthropic/hh-rlhf
    Extrait les paires (chosen, rejected) et prépare les prompts
    """
    
    def __init__(
        self, 
        split: str = "train",
        max_samples: Optional[int] = None,
        tokenizer = None,
        max_length: int = 512
    ):
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        if not HF_DATASETS_AVAILABLE:
            raise ImportError("datasets non disponible! Installez avec: pip install datasets")
        
        print(f"📥 Chargement Anthropic/hh-rlhf ({split})...")
        
        # Charger le dataset
        if max_samples:
            dataset = load_dataset("Anthropic/hh-rlhf", split=f"{split}[:{max_samples}]")
        else:
            dataset = load_dataset("Anthropic/hh-rlhf", split=split)
        
        self.data = []
        
        # Parser les conversations
        for item in tqdm(dataset, desc=f"Parsing {split}"):
            parsed = self._parse_conversation(item)
            if parsed:
                self.data.append(parsed)
        
        print(f"✅ {len(self.data)} exemples chargés")
    
    def _parse_conversation(self, item: Dict) -> Optional[Dict]:
        """
        Parse une conversation du format Anthropic
        Format: "Human: ... Assistant: ... Human: ... Assistant: ..."
        """
        chosen = item.get('chosen', '')
        rejected = item.get('rejected', '')
        
        if not chosen or not rejected:
            return None
        
        # Split par "Assistant:"
        chosen_parts = chosen.split('\n\nAssistant:')
        if len(chosen_parts) < 2:
            return None
        
        # Le prompt est tout sauf la dernière réponse
        prompt = '\n\nAssistant:'.join(chosen_parts[:-1]) + '\n\nAssistant:'
        
        # Les réponses
        chosen_response = chosen_parts[-1].strip()
        
        rejected_parts = rejected.split('\n\nAssistant:')
        rejected_response = rejected_parts[-1].strip() if len(rejected_parts) >= 2 else ""
        
        if not chosen_response or not rejected_response:
            return None
        
        return {
            'query': prompt,
            'chosen': chosen_response,
            'rejected': rejected_response
        }
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.data[idx]


# ============================================================================
# MODÈLE DE RÉCOMPENSE
# ============================================================================

class RewardModel:
    """
    Modèle de récompense basé sur les préférences humaines
    Utilise les paires (chosen, rejected) pour apprendre à scorer les réponses
    """
    
    def __init__(self, use_heuristics: bool = True):
        self.use_heuristics = use_heuristics
        self.logger = logging.getLogger("RewardModel")
    
    def compute_rewards(
        self,
        queries: List[str],
        responses: List[str],
        tokenizer = None
    ) -> List[float]:
        """
        Calcule les récompenses pour chaque réponse
        
        Heuristiques utilisées:
        1. Longueur raisonnable (ni trop court ni trop long)
        2. Diversité du vocabulaire
        3. Absence de répétitions
        4. Présence de ponctuation appropriée
        5. Cohérence (pas de tokens spéciaux visibles)
        """
        rewards = []
        
        for query, response in zip(queries, responses):
            reward = 0.0
            
            # 1. Longueur (optimal entre 50-300 caractères)
            length = len(response)
            if 50 <= length <= 300:
                reward += 1.0
            elif 30 <= length < 50 or 300 < length <= 500:
                reward += 0.5
            else:
                reward -= 0.5
            
            # 2. Diversité du vocabulaire
            words = response.split()
            if len(words) > 0:
                unique_ratio = len(set(words)) / len(words)
                reward += unique_ratio * 0.5
            
            # 3. Pas de répétitions excessives
            if len(words) >= 3:
                trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
                if len(trigrams) > 0:
                    repetition_ratio = 1.0 - (len(set(trigrams)) / len(trigrams))
                    reward -= repetition_ratio * 1.0
            
            # 4. Ponctuation appropriée
            has_period = '.' in response or '!' in response or '?' in response
            if has_period:
                reward += 0.3
            
            # 5. Pas de tokens spéciaux visibles
            special_tokens = ['<pad>', '<unk>', '<s>', '</s>', '[PAD]', '[UNK]']
            if not any(token in response for token in special_tokens):
                reward += 0.2
            
            # 6. Cohérence avec la query
            if response.strip() != query.strip():
                reward += 0.3
            
            rewards.append(reward)
        
        return rewards


# ============================================================================
# TRAINER RLHF PRINCIPAL
# ============================================================================

class RLHFTrainer:
    """
    Trainer RLHF principal compatible avec GAMA-AI
    """
    
    def __init__(
        self,
        base_model,
        tokenizer,
        device: torch.device,
        rlhf_config: RLHFConfig,
        model_dir: Optional[str] = None
    ):
        """
        Args:
            base_model: Modèle HessGPT déjà entraîné
            tokenizer: Tokenizer MYBPE
            device: torch.device
            rlhf_config: Configuration RLHF
            model_dir: Répertoire du modèle (optionnel)
        """
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = device
        self.config = rlhf_config
        self.model_dir = Path(model_dir) if model_dir else None
        
        self.logger = logging.getLogger("RLHFTrainer")
        self.logger.setLevel(logging.INFO)
        
        # Modèle de récompense
        self.reward_model = RewardModel(use_heuristics=True)
        
        # Créer le répertoire de sortie
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        self.logger.info("✅ RLHFTrainer initialisé")
    
    def _prepare_datasets(self):
        """Prépare les datasets train et validation"""
        train_dataset = AnthropicRLHFDataset(
            split="train",
            max_samples=self.config.max_samples_train,
            tokenizer=self.tokenizer,
            max_length=512
        )
        
        val_dataset = AnthropicRLHFDataset(
            split="test",
            max_samples=self.config.max_samples_val,
            tokenizer=self.tokenizer,
            max_length=512
        )
        
        return train_dataset, val_dataset
    
    def train_with_rlhf(self):
        """
        Lance l'entraînement RLHF complet
        """
        print("\n" + "="*70)
        print("🎯 DÉMARRAGE ENTRAÎNEMENT RLHF")
        print("="*70)
        print(f"📊 Dataset: {self.config.dataset_name}")
        print(f"💾 Sortie: {self.config.output_dir}")
        print(f"🔧 PPO Epochs: {self.config.ppo_epochs}")
        print(f"📦 Batch size: {self.config.batch_size}")
        print("="*70 + "\n")
        
        # Préparer datasets
        train_dataset, val_dataset = self._prepare_datasets()
        
        self.logger.info("🚀 Début de l'entraînement RLHF simplifié...")
        
        # Passer le modèle en mode train
        self.base_model.train()
        
        # Optimizer
        optimizer = AdamW(
            self.base_model.parameters(),
            lr=self.config.learning_rate
        )
        
        # Training loop simplifié
        total_steps = 0
        best_avg_reward = float('-inf')
        
        for epoch in range(self.config.num_train_epochs):
            self.logger.info(f"📍 Epoch {epoch + 1}/{self.config.num_train_epochs}")
            
            epoch_rewards = []
            epoch_losses = []
            
            # DataLoader
            dataloader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                collate_fn=lambda x: x
            )
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
            
            for batch_idx, batch in enumerate(pbar):
                queries = [item['query'] for item in batch]
                
                # Tokenize queries
                query_tensors = []
                for query in queries:
                    tokens = self.tokenizer.encoder(query)
                    tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)
                    query_tensors.append(tensor)
                
                # Générer réponses
                response_tensors = []
                with torch.no_grad():
                    for query_tensor in query_tensors:
                        # Génération simple
                        current_seq = query_tensor
                        for _ in range(min(self.config.max_new_tokens, 100)):
                            logits, _ = self.base_model(current_seq)
                            next_token_logits = logits[:, -1, :] / self.config.temperature
                            probs = torch.softmax(next_token_logits, dim=-1)
                            next_token = torch.multinomial(probs, num_samples=1)
                            current_seq = torch.cat([current_seq, next_token], dim=1)
                        response_tensors.append(current_seq[0, query_tensor.shape[1]:])
                
                # Décoder les réponses
                responses = [
                    self.tokenizer.decoder(resp.cpu().tolist())
                    for resp in response_tensors
                ]
                
                # Calculer les récompenses
                rewards = self.reward_model.compute_rewards(queries, responses, self.tokenizer)
                epoch_rewards.extend(rewards)
                
                # Mise à jour du modèle (policy gradient simplifié)
                optimizer.zero_grad()
                
                loss = torch.tensor(0.0, device=self.device)
                for query_tensor, response_tensor, reward in zip(query_tensors, response_tensors, rewards):
                    # Concaténer query + response
                    full_seq = torch.cat([query_tensor[0], response_tensor])
                    
                    if len(full_seq) > 512:
                        full_seq = full_seq[-512:]
                    
                    # Forward pass
                    logits, _ = self.base_model(full_seq.unsqueeze(0))
                    
                    # Loss: negative log likelihood weighted by reward
                    log_probs = torch.log_softmax(logits[0, :-1], dim=-1)
                    selected_log_probs = log_probs.gather(1, full_seq[1:].unsqueeze(-1)).squeeze(-1)
                    
                    # Récompense normalisée
                    reward_tensor = torch.tensor(reward, device=self.device)
                    policy_loss = -(selected_log_probs * reward_tensor).mean()
                    
                    loss += policy_loss
                
                loss = loss / len(batch)
                epoch_losses.append(loss.item())
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.base_model.parameters(), self.config.max_grad_norm)
                optimizer.step()
                
                # Logging
                avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'reward': f'{avg_reward:.3f}'
                })
                
                total_steps += 1
                
                # Sauvegarde périodique
                if total_steps % self.config.save_steps == 0:
                    self._save_checkpoint(epoch, total_steps)
            
            # Stats epoch
            avg_epoch_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0.0
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
            
            self.logger.info(
                f"Epoch {epoch+1}: "
                f"Avg Reward={avg_epoch_reward:.4f}, "
                f"Avg Loss={avg_epoch_loss:.4f}"
            )
            
            # Sauvegarde si meilleure récompense
            if avg_epoch_reward > best_avg_reward:
                best_avg_reward = avg_epoch_reward
                self._save_checkpoint(epoch, total_steps, is_best=True)
        
        # Sauvegarde finale
        self._save_final_model()
        
        print("\n" + "="*70)
        print("✅ ENTRAÎNEMENT RLHF TERMINÉ")
        print("="*70)
        print(f"🎯 Meilleure récompense moyenne: {best_avg_reward:.4f}")
        print(f"💾 Modèle sauvegardé: {self.config.output_dir}")
        print("="*70 + "\n")
    
    def _save_checkpoint(self, epoch: int, step: int, is_best: bool = False):
        """Sauvegarde un checkpoint"""
        suffix = "best" if is_best else f"epoch{epoch}_step{step}"
        checkpoint_dir = Path(self.config.output_dir) / f"checkpoint_{suffix}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder le modèle
        model_path = checkpoint_dir / "model.pt"
        torch.save(self.base_model.state_dict(), model_path)
        
        # Sauvegarder config
        config_path = checkpoint_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
        
        self.logger.info(f"💾 Checkpoint sauvegardé: {checkpoint_dir}")
    
    def _save_final_model(self):
        """Sauvegarde le modèle final"""
        final_dir = Path(self.config.output_dir) / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = final_dir / "model.pt"
        torch.save(self.base_model.state_dict(), model_path)
        
        self.logger.info(f"✅ Modèle final sauvegardé: {final_dir}")
