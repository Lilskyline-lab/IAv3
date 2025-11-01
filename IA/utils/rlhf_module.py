"""
Module DPO (Direct Preference Optimization) - Alternative moderne au RLHF/PPO
Plus simple, plus stable, et tout aussi efficace que PPO pour l'alignment

DPO élimine le besoin d'un reward model séparé et utilise directement
les préférences humaines pour optimiser la policy.

Paper: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
https://arxiv.org/abs/2305.18290

Installation:
    pip install datasets transformers torch tqdm
"""

import os
import sys
import json
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field, asdict
from tqdm import tqdm
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False
    print("⚠️  datasets non disponible. Installez: pip install datasets")


# ============================================================================
# CONFIGURATION DPO
# ============================================================================

@dataclass
class DPOConfig:
    """Configuration pour DPO (Direct Preference Optimization)"""
    
    # Dataset
    dataset_name: str = "Anthropic/hh-rlhf"
    max_samples_train: int = 10000
    max_samples_val: int = 1000
    max_prompt_length: int = 256
    max_response_length: int = 256
    
    # DPO Hyperparameters
    beta: float = 0.1  # Coefficient de régularisation KL (très important!)
    learning_rate: float = 5e-7  # LR plus petit pour stability
    num_epochs: int = 1
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    
    # Optimization
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    warmup_steps: int = 100
    
    # Logging & Saving
    logging_steps: int = 10
    eval_steps: int = 250
    save_steps: int = 500
    output_dir: str = "./dpo_output"
    
    # Advanced
    label_smoothing: float = 0.0  # Label smoothing pour stability
    use_weighting: bool = False  # Pondérer les exemples par leur margin


# ============================================================================
# DATASET POUR DPO
# ============================================================================

class DPODataset(Dataset):
    """
    Dataset pour DPO avec paires (chosen, rejected)
    
    Format attendu pour chaque exemple:
    {
        'prompt': "...",
        'chosen': "...",
        'rejected': "..."
    }
    """
    
    def __init__(
        self,
        split: str = "train",
        max_samples: Optional[int] = None,
        tokenizer = None,
        max_prompt_length: int = 256,
        max_response_length: int = 256
    ):
        self.split = split
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        
        if not HF_DATASETS_AVAILABLE:
            raise ImportError("datasets non disponible!")
        
        print(f"📥 Chargement Anthropic/hh-rlhf ({split})...")
        
        # Charger dataset
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
        
        print(f"✅ {len(self.data)} paires chargées")
    
    def _parse_conversation(self, item: Dict) -> Optional[Dict]:
        """
        Parse format Anthropic/hh-rlhf
        
        Format original:
        chosen: "Human: ... Assistant: ... Human: ... Assistant: GOOD_RESPONSE"
        rejected: "Human: ... Assistant: ... Human: ... Assistant: BAD_RESPONSE"
        """
        chosen = item.get('chosen', '')
        rejected = item.get('rejected', '')
        
        if not chosen or not rejected:
            return None
        
        # Split sur "\n\nAssistant:"
        chosen_parts = chosen.split('\n\nAssistant:')
        rejected_parts = rejected.split('\n\nAssistant:')
        
        if len(chosen_parts) < 2 or len(rejected_parts) < 2:
            return None
        
        # Prompt = tout sauf la dernière réponse
        prompt_chosen = '\n\nAssistant:'.join(chosen_parts[:-1])
        prompt_rejected = '\n\nAssistant:'.join(rejected_parts[:-1])
        
        # Vérifier que les prompts sont identiques
        if prompt_chosen != prompt_rejected:
            # Prendre le prompt le plus court (plus safe)
            prompt = min(prompt_chosen, prompt_rejected, key=len)
        else:
            prompt = prompt_chosen
        
        # Extraire les réponses
        chosen_response = chosen_parts[-1].strip()
        rejected_response = rejected_parts[-1].strip()
        
        if not chosen_response or not rejected_response:
            return None
        
        # Ajouter "Assistant:" au début pour cohérence
        prompt = prompt + '\n\nAssistant:'
        
        return {
            'prompt': prompt,
            'chosen': chosen_response,
            'rejected': rejected_response
        }
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        
        # Tokenize
        prompt_ids = self.tokenizer.encoder(item['prompt'])
        chosen_ids = self.tokenizer.encoder(item['chosen'])
        rejected_ids = self.tokenizer.encoder(item['rejected'])
        
        # Tronquer si nécessaire
        prompt_ids = prompt_ids[-self.max_prompt_length:]
        chosen_ids = chosen_ids[:self.max_response_length]
        rejected_ids = rejected_ids[:self.max_response_length]
        
        return {
            'prompt_ids': prompt_ids,
            'chosen_ids': chosen_ids,
            'rejected_ids': rejected_ids,
            'prompt_text': item['prompt'],
            'chosen_text': item['chosen'],
            'rejected_text': item['rejected']
        }


def dpo_collate_fn(batch: List[Dict], pad_id: int = 0) -> Dict:
    """Collate function pour DPO"""
    
    # Séparer les composants
    prompt_ids = [item['prompt_ids'] for item in batch]
    chosen_ids = [item['chosen_ids'] for item in batch]
    rejected_ids = [item['rejected_ids'] for item in batch]
    
    # Padding
    max_prompt_len = max(len(ids) for ids in prompt_ids)
    max_chosen_len = max(len(ids) for ids in chosen_ids)
    max_rejected_len = max(len(ids) for ids in rejected_ids)
    
    # Tensors pour chosen
    prompt_chosen_ids = []
    chosen_labels = []
    
    for p_ids, c_ids in zip(prompt_ids, chosen_ids):
        # Concaténer prompt + chosen
        full_ids = p_ids + c_ids
        
        # Padding
        pad_len = (max_prompt_len + max_chosen_len) - len(full_ids)
        padded = full_ids + [pad_id] * pad_len
        
        # Labels: -100 pour le prompt, vrai IDs pour chosen
        labels = [-100] * len(p_ids) + c_ids + [-100] * pad_len
        
        prompt_chosen_ids.append(padded)
        chosen_labels.append(labels)
    
    # Tensors pour rejected
    prompt_rejected_ids = []
    rejected_labels = []
    
    for p_ids, r_ids in zip(prompt_ids, rejected_ids):
        full_ids = p_ids + r_ids
        pad_len = (max_prompt_len + max_rejected_len) - len(full_ids)
        padded = full_ids + [pad_id] * pad_len
        labels = [-100] * len(p_ids) + r_ids + [-100] * pad_len
        
        prompt_rejected_ids.append(padded)
        rejected_labels.append(labels)
    
    return {
        'chosen_input_ids': torch.tensor(prompt_chosen_ids, dtype=torch.long),
        'chosen_labels': torch.tensor(chosen_labels, dtype=torch.long),
        'rejected_input_ids': torch.tensor(prompt_rejected_ids, dtype=torch.long),
        'rejected_labels': torch.tensor(rejected_labels, dtype=torch.long)
    }


# ============================================================================
# DPO TRAINER
# ============================================================================

class DPOTrainer:
    """
    Trainer DPO (Direct Preference Optimization)
    
    DPO optimise directement la policy en utilisant les préférences,
    sans besoin d'un reward model séparé.
    
    Loss DPO:
    L = -log(σ(β * [log π_θ(y_w|x) - log π_ref(y_w|x) 
                     - log π_θ(y_l|x) + log π_ref(y_l|x)]))
    
    Où:
    - y_w = chosen response
    - y_l = rejected response
    - π_θ = policy model
    - π_ref = reference model (frozen)
    - β = regularization coefficient
    - σ = sigmoid
    """
    
    def __init__(
        self,
        model,  # Policy model (sera optimisé)
        tokenizer,
        device: torch.device,
        config: DPOConfig,
        model_dir: Optional[str] = None
    ):
        self.policy_model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        self.model_dir = Path(model_dir) if model_dir else None
        
        # Créer reference model (frozen copy du policy)
        print("📋 Création du reference model (copie figée)...")
        self.ref_model = copy.deepcopy(model)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        print("✅ Reference model créé et figé")
        
        # Logger
        self.logger = logging.getLogger("DPOTrainer")
        self.logger.setLevel(logging.INFO)
        
        # Output dir
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Métriques
        self.train_stats = {
            'loss': [],
            'accuracy': [],  # % où chosen > rejected
            'chosen_rewards': [],
            'rejected_rewards': []
        }
        
        print(f"✅ DPOTrainer initialisé (β={config.beta})")
    
    def compute_log_probs(
        self,
        model,
        input_ids: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcule log P(y|x) pour une séquence
        
        Returns:
            Tensor de shape (batch_size,) avec log prob moyen par token
        """
        # Forward pass
        logits, _ = model(input_ids)
        
        # Shift pour autoregressive
        logits = logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
        
        # Log probs
        log_probs = F.log_softmax(logits, dim=-1)
        
        # IMPORTANT: Masquer d'abord les -100 avant gather
        mask = (labels != -100).float()
        
        # Remplacer -100 par 0 pour éviter l'erreur d'index
        labels_masked = labels.clone()
        labels_masked[labels == -100] = 0
        
        # Gather les log probs des tokens réels
        selected_log_probs = log_probs.gather(
            dim=-1,
            index=labels_masked.unsqueeze(-1)
        ).squeeze(-1)
        
        # Appliquer le masque
        masked_log_probs = selected_log_probs * mask
        seq_log_probs = masked_log_probs.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
        
        return seq_log_probs
    
    def dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Calcule la loss DPO
        
        Returns:
            loss, metrics_dict
        """
        # Rewards implicites
        policy_chosen_rewards = policy_chosen_logps - ref_chosen_logps
        policy_rejected_rewards = policy_rejected_logps - ref_rejected_logps
        
        # Logits pour DPO
        logits = self.config.beta * (policy_chosen_rewards - policy_rejected_rewards)
        
        # Loss: -log sigmoid(logits)
        loss = -F.logsigmoid(logits).mean()
        
        # Métriques
        with torch.no_grad():
            accuracy = (logits > 0).float().mean().item()
            chosen_rewards = policy_chosen_rewards.mean().item()
            rejected_rewards = policy_rejected_rewards.mean().item()
        
        metrics = {
            'accuracy': accuracy,
            'chosen_rewards': chosen_rewards,
            'rejected_rewards': rejected_rewards,
            'reward_margin': chosen_rewards - rejected_rewards
        }
        
        return loss, metrics
    
    def train_step(
        self,
        batch: Dict
    ) -> Tuple[torch.Tensor, Dict]:
        """Un step d'entraînement"""
        
        # Move to device
        chosen_ids = batch['chosen_input_ids'].to(self.device)
        chosen_labels = batch['chosen_labels'].to(self.device)
        rejected_ids = batch['rejected_input_ids'].to(self.device)
        rejected_labels = batch['rejected_labels'].to(self.device)
        
        # Policy log probs
        policy_chosen_logps = self.compute_log_probs(
            self.policy_model, chosen_ids, chosen_labels
        )
        policy_rejected_logps = self.compute_log_probs(
            self.policy_model, rejected_ids, rejected_labels
        )
        
        # Reference log probs (no grad)
        with torch.no_grad():
            ref_chosen_logps = self.compute_log_probs(
                self.ref_model, chosen_ids, chosen_labels
            )
            ref_rejected_logps = self.compute_log_probs(
                self.ref_model, rejected_ids, rejected_labels
            )
        
        # DPO loss
        loss, metrics = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps
        )
        
        return loss, metrics
    
    def train(self):
        """Lance l'entraînement DPO"""
        print("\n" + "="*70)
        print("🎯 DÉMARRAGE ENTRAÎNEMENT DPO")
        print("="*70)
        print(f"📊 Dataset: {self.config.dataset_name}")
        print(f"💾 Output: {self.config.output_dir}")
        print(f"🔧 Beta (KL coef): {self.config.beta}")
        print(f"📦 Batch size: {self.config.batch_size}")
        print(f"📚 Epochs: {self.config.num_epochs}")
        print("="*70 + "\n")
        
        # Préparer datasets
        print("📥 Chargement des datasets...")
        train_dataset = DPODataset(
            split="train",
            max_samples=self.config.max_samples_train,
            tokenizer=self.tokenizer,
            max_prompt_length=self.config.max_prompt_length,
            max_response_length=self.config.max_response_length
        )
        
        val_dataset = DPODataset(
            split="test",
            max_samples=self.config.max_samples_val,
            tokenizer=self.tokenizer,
            max_prompt_length=self.config.max_prompt_length,
            max_response_length=self.config.max_response_length
        )
        
        # DataLoaders
        pad_id = getattr(self.tokenizer, 'pad_token_id', 0) or 0
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=lambda b: dpo_collate_fn(b, pad_id=pad_id)
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=lambda b: dpo_collate_fn(b, pad_id=pad_id)
        )
        
        # Optimizer
        optimizer = AdamW(
            self.policy_model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Training loop
        self.policy_model.train()
        global_step = 0
        best_val_accuracy = 0.0
        
        for epoch in range(self.config.num_epochs):
            print(f"\n📍 Epoch {epoch + 1}/{self.config.num_epochs}")
            
            epoch_losses = []
            epoch_accuracies = []
            
            pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
            
            for batch_idx, batch in enumerate(pbar):
                # Forward & loss
                loss, metrics = self.train_step(batch)
                
                # Backward
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.policy_model.parameters(),
                        self.config.max_grad_norm
                    )
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                
                # Stats
                epoch_losses.append(loss.item() * self.config.gradient_accumulation_steps)
                epoch_accuracies.append(metrics['accuracy'])
                
                # Logging
                if global_step % self.config.logging_steps == 0:
                    pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'acc': f"{metrics['accuracy']:.2%}",
                        'margin': f"{metrics['reward_margin']:.3f}"
                    })
                
                # Eval
                if global_step % self.config.eval_steps == 0:
                    val_metrics = self._evaluate(val_loader)
                    self.policy_model.train()
                    
                    if val_metrics['accuracy'] > best_val_accuracy:
                        best_val_accuracy = val_metrics['accuracy']
                        self._save_checkpoint(epoch, global_step, is_best=True)
                
                # Save
                if global_step % self.config.save_steps == 0:
                    self._save_checkpoint(epoch, global_step)
            
            # Epoch summary
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            avg_acc = sum(epoch_accuracies) / len(epoch_accuracies)
            
            print(f"\n✓ Epoch {epoch+1} terminé:")
            print(f"  Loss moyenne: {avg_loss:.4f}")
            print(f"  Accuracy moyenne: {avg_acc:.2%}")
        
        # Sauvegarde finale
        self._save_checkpoint(epoch, global_step, is_final=True)
        
        print("\n" + "="*70)
        print("✅ ENTRAÎNEMENT DPO TERMINÉ")
        print("="*70)
        print(f"🎯 Meilleure accuracy validation: {best_val_accuracy:.2%}")
        print(f"💾 Modèle sauvegardé: {self.config.output_dir}")
        print("="*70 + "\n")
    
    @torch.no_grad()
    def _evaluate(self, val_loader: DataLoader) -> Dict:
        """Évalue le modèle"""
        self.policy_model.eval()
        
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        for batch in tqdm(val_loader, desc="Validation"):
            loss, metrics = self.train_step(batch)
            total_loss += loss.item()
            total_accuracy += metrics['accuracy']
            num_batches += 1
        
        avg_metrics = {
            'loss': total_loss / num_batches,
            'accuracy': total_accuracy / num_batches
        }
        
        print(f"\n📊 Validation: Loss={avg_metrics['loss']:.4f}, "
              f"Accuracy={avg_metrics['accuracy']:.2%}")
        
        return avg_metrics
    
    def _save_checkpoint(
        self,
        epoch: int,
        step: int,
        is_best: bool = False,
        is_final: bool = False
    ):
        """Sauvegarde un checkpoint"""
        if is_final:
            suffix = "final"
        elif is_best:
            suffix = "best"
        else:
            suffix = f"epoch{epoch}_step{step}"
        
        checkpoint_dir = Path(self.config.output_dir) / f"checkpoint_{suffix}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder policy model
        torch.save(
            self.policy_model.state_dict(),
            checkpoint_dir / "model.pt"
        )
        
        # Sauvegarder config
        with open(checkpoint_dir / "dpo_config.json", 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
        
        self.logger.info(f"💾 Checkpoint sauvegardé: {checkpoint_dir}")


# ============================================================================
# FONCTION D'ENTRAÎNEMENT RAPIDE
# ============================================================================

def train_with_dpo(
    model,
    tokenizer,
    device: torch.device,
    config: Optional[DPOConfig] = None,
    model_dir: Optional[str] = None
):
    """
    Fonction wrapper pour lancer DPO facilement
    
    Usage:
        from rlhf_dpo_module import train_with_dpo, DPOConfig
        
        config = DPOConfig(
            beta=0.1,
            learning_rate=5e-7,
            num_epochs=1
        )
        
        train_with_dpo(model, tokenizer, device, config)
    """
    if config is None:
        config = DPOConfig()
    
    trainer = DPOTrainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        config=config,
        model_dir=model_dir
    )
    
    trainer.train()
    
    return trainer