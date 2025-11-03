"""
Infrastructure complète pour entraîner Qwen2-0.5B en chatbot
Version corrigée compatible avec transformers >= 4.30 et trl >= 0.7

Installation requise:
    pip install torch transformers datasets accelerate peft bitsandbytes trl wandb

Usage:
    # Phase 1: Supervised Fine-Tuning
    python train_qwen_chatbot.py --phase sft --dataset alpaca --epochs 1
    
    # Test du modèle
    python train_qwen_chatbot.py --phase test --checkpoint ./checkpoints/sft_final
    
    # Créer dataset d'exemple
    python train_qwen_chatbot.py --phase create-example
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer  # Gardé pour compatibilité mais non utilisé dans trl 0.24+
import argparse
from pathlib import Path
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
import os


@dataclass
class TrainingConfig:
    """Configuration d'entraînement"""
    model_name: str = "Qwen/Qwen2-0.5B-Instruct"
    output_dir: str = "./checkpoints"
    
    # Hyperparamètres généraux
    learning_rate: float = 2e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    max_seq_length: int = 512
    warmup_steps: int = 100
    
    # LoRA config (pour fine-tuning efficace)
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = None
    
    # Logging
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    use_wandb: bool = False
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            # Modules typiques pour Qwen/LLaMA
            self.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]


class DatasetLoader:
    """Chargeur de datasets pour le fine-tuning"""
    
    DATASETS = {
        "alpaca": {
            "name": "tatsu-lab/alpaca",
            "format": "instruction",
            "description": "52K instructions en anglais"
        },
        "dolly": {
            "name": "databricks/databricks-dolly-15k",
            "format": "instruction",
            "description": "15K instructions diverses"
        },
        "oasst1": {
            "name": "OpenAssistant/oasst1",
            "format": "conversation",
            "description": "Conversations multi-tours"
        },
        "custom": {
            "name": "custom",
            "format": "custom",
            "description": "Dataset personnalisé (JSON)"
        }
    }
    
    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def load(self, dataset_name: str, custom_path: Optional[str] = None) -> Dataset:
        """
        Charge et prépare un dataset
        
        Args:
            dataset_name: Nom du dataset (alpaca, dolly, oasst1, custom)
            custom_path: Chemin vers dataset custom (si dataset_name='custom')
        """
        print(f"\n{'='*70}")
        print(f"📚 CHARGEMENT DU DATASET")
        print(f"{'='*70}")
        
        if dataset_name not in self.DATASETS:
            raise ValueError(f"Dataset inconnu: {dataset_name}. Disponibles: {list(self.DATASETS.keys())}")
        
        if dataset_name == "custom":
            if not custom_path:
                raise ValueError("custom_path requis pour dataset custom")
            dataset = self._load_custom(custom_path)
        else:
            dataset_info = self.DATASETS[dataset_name]
            print(f"Dataset: {dataset_info['description']}")
            dataset = load_dataset(dataset_info['name'], split='train')
            
            # Formater selon le type
            if dataset_info['format'] == 'instruction':
                dataset = dataset.map(self._format_instruction, batched=False)
            elif dataset_info['format'] == 'conversation':
                dataset = dataset.map(self._format_conversation, batched=False)
        
        print(f"✅ Dataset chargé: {len(dataset):,} exemples")
        print(f"{'='*70}\n")
        
        return dataset
    
    def _format_instruction(self, example: Dict) -> Dict:
        """
        Formate un exemple instruction → réponse
        Format Qwen: <|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>
        """
        # Adapter selon la structure du dataset
        if 'instruction' in example and 'output' in example:
            instruction = example['instruction']
            if 'input' in example and example['input']:
                instruction += f"\n{example['input']}"
            response = example['output']
        elif 'question' in example and 'answer' in example:
            instruction = example['question']
            response = example['answer']
        else:
            return example
        
        # Format Qwen
        text = (
            f"<|im_start|>user\n{instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n{response}<|im_end|>"
        )
        
        return {"text": text}
    
    def _format_conversation(self, example: Dict) -> Dict:
        """Formate une conversation multi-tours"""
        # Pour OASST1 ou conversations similaires
        if 'messages' in example:
            formatted = []
            for msg in example['messages']:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                formatted.append(f"<|im_start|>{role}\n{content}<|im_end|>")
            text = "\n".join(formatted)
        else:
            return example
        
        return {"text": text}
    
    def _load_custom(self, path: str) -> Dataset:
        """
        Charge un dataset custom au format JSON
        
        Format attendu:
        [
            {
                "instruction": "Question ou instruction",
                "output": "Réponse attendue"
            },
            ...
        ]
        """
        print(f"Chargement du dataset custom: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Formatter
        formatted_data = []
        for item in data:
            text = (
                f"<|im_start|>user\n{item['instruction']}<|im_end|>\n"
                f"<|im_start|>assistant\n{item['output']}<|im_end|>"
            )
            formatted_data.append({"text": text})
        
        return Dataset.from_list(formatted_data)


class QwenChatbotTrainer:
    """Entraîneur principal pour Qwen2-0.5B"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = config.device
        
        print(f"\n{'='*70}")
        print(f"🚀 INITIALISATION DE L'ENTRAÎNEMENT")
        print(f"{'='*70}")
        print(f"Modèle: {config.model_name}")
        print(f"Device: {self.device}")
        print(f"LoRA: {'✅' if config.use_lora else '❌'}")
        print(f"WandB: {'✅' if config.use_wandb else '❌'}")
        print(f"{'='*70}\n")
        
        # Créer le dossier de sortie
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Charger tokenizer
        print("🔤 Chargement du tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
        
        # Configuration du pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        print(f"✅ Tokenizer: {len(self.tokenizer):,} tokens")
        
        # Charger modèle
        print("🤖 Chargement du modèle...")
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            dtype=torch.float32,  # Changé de torch_dtype à dtype
            device_map=self.device,
            trust_remote_code=True
        )
        
        # Appliquer LoRA si activé
        if config.use_lora:
            print("⚡ Application de LoRA...")
            self.model = self._apply_lora()
            
            # IMPORTANT: S'assurer que le modèle est en mode training
            self.model.train()
        
        print(f"✅ Modèle chargé!")
        
        # Compter paramètres
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"📊 Paramètres totaux: {total_params:,} ({total_params/1e6:.1f}M)")
        print(f"📊 Paramètres entraînables: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
        print(f"📊 Pourcentage entraînable: {100 * trainable_params / total_params:.2f}%")
        
        # WandB
        if config.use_wandb:
            try:
                import wandb
                wandb.init(project="qwen-chatbot", config=vars(config))
            except ImportError:
                print("⚠️  WandB non installé, logging désactivé")
    
    def _apply_lora(self) -> nn.Module:
        """Applique LoRA au modèle pour un fine-tuning efficace"""
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Appliquer LoRA
        model = get_peft_model(self.model, lora_config)
        
        # S'assurer que les paramètres LoRA sont entraînables
        model.print_trainable_parameters()
        
        # Forcer l'activation des gradients pour les paramètres LoRA
        for name, param in model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = True
        
        return model
    
    def train_sft(self, dataset: Dataset):
        """
        Phase 1: Supervised Fine-Tuning (SFT)
        Entraîne le modèle sur des paires instruction-réponse
        """
        print(f"\n{'='*70}")
        print(f"📖 PHASE 1: SUPERVISED FINE-TUNING (SFT)")
        print(f"{'='*70}\n")
        
        # Préparer le dataset avec tokenization
        def tokenize_function(examples):
            # Tokenize les textes
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.max_seq_length,
                padding="max_length",
            )
            # Ajouter les labels (identiques aux input_ids pour du causal LM)
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        print("🔄 Tokenization du dataset...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset"
        )
        
        # Configuration de base (TrainingArguments standard)
        training_args = TrainingArguments(
            output_dir=f"{self.config.output_dir}/sft",
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=3,
            report_to="wandb" if self.config.use_wandb else "none",
            remove_unused_columns=False,
            gradient_checkpointing=False,  # Désactiver gradient checkpointing avec LoRA
            fp16=(self.device == "cuda"),
        )
        
        # Data collator pour causal language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, pas masked LM
        )
        
        # Trainer standard (pas SFTTrainer pour trl 0.24+)
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        # Entraîner
        print("🏋️ Démarrage de l'entraînement SFT...\n")
        trainer.train()
        
        # Sauvegarder
        final_path = f"{self.config.output_dir}/sft_final"
        trainer.save_model(final_path)
        self.tokenizer.save_pretrained(final_path)
        
        print(f"\n✅ SFT terminé! Modèle sauvegardé dans: {final_path}")
        
        return trainer
    
    def test_model(self, checkpoint_path: str):
        """Teste le modèle entraîné en mode interactif"""
        print(f"\n{'='*70}")
        print(f"🧪 TEST DU MODÈLE")
        print(f"{'='*70}")
        print(f"Chargement depuis: {checkpoint_path}\n")
        
        # Charger le modèle fine-tuné
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            dtype=torch.float32,
            device_map=self.device,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
        
        model.eval()
        
        print(f"{'='*70}")
        print(f"💬 MODE CHAT INTERACTIF")
        print(f"{'='*70}")
        print("Tapez /quit pour quitter")
        print("Tapez /help pour voir les commandes\n")
        
        while True:
            try:
                user_input = input("👤 Vous: ").strip()
                
                if user_input == "/quit":
                    print("👋 Au revoir!")
                    break
                
                if user_input == "/help":
                    print("\nCommandes disponibles:")
                    print("  /quit  - Quitter le chat")
                    print("  /help  - Afficher cette aide\n")
                    continue
                
                if not user_input:
                    continue
                
                # Format Qwen
                prompt = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
                
                # Tokenize
                inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
                
                # Générer
                with torch.no_grad():
                    outputs = model.generate(
                        inputs['input_ids'],
                        max_new_tokens=256,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                
                # Decoder
                response = tokenizer.decode(outputs[0], skip_special_tokens=False)
                
                # Extraire la réponse
                if "<|im_start|>assistant\n" in response:
                    assistant_response = response.split("<|im_start|>assistant\n")[-1]
                    assistant_response = assistant_response.replace("<|im_end|>", "").strip()
                else:
                    assistant_response = response
                
                print(f"🤖 Assistant: {assistant_response}\n")
                
            except KeyboardInterrupt:
                print("\n👋 Au revoir!")
                break
            except Exception as e:
                print(f"❌ Erreur: {e}\n")


def create_example_dataset():
    """Crée un dataset d'exemple pour tester"""
    examples = [
        {
            "instruction": "Qu'est-ce que l'intelligence artificielle?",
            "output": "L'intelligence artificielle (IA) est un domaine de l'informatique qui vise à créer des systèmes capables d'effectuer des tâches nécessitant normalement l'intelligence humaine, comme l'apprentissage, le raisonnement et la résolution de problèmes."
        },
        {
            "instruction": "Explique Python en termes simples.",
            "output": "Python est un langage de programmation facile à apprendre et très populaire. Il utilise une syntaxe claire et lisible qui ressemble presque à l'anglais, ce qui le rend idéal pour les débutants."
        },
        {
            "instruction": "Comment fonctionne un réseau de neurones?",
            "output": "Un réseau de neurones est inspiré du cerveau humain. Il contient des couches de neurones artificiels connectés qui traitent l'information. Chaque connexion a un poids qui s'ajuste pendant l'entraînement pour améliorer les prédictions."
        },
        {
            "instruction": "Qu'est-ce que le fine-tuning?",
            "output": "Le fine-tuning consiste à prendre un modèle pré-entraîné et à l'adapter à une tâche spécifique en continuant son entraînement sur de nouvelles données. C'est plus rapide et efficace que d'entraîner un modèle depuis zéro."
        },
        {
            "instruction": "Raconte-moi une blague.",
            "output": "Pourquoi les développeurs préfèrent-ils le dark mode? Parce que la lumière attire les bugs! 🐛"
        },
        {
            "instruction": "Qu'est-ce que LoRA?",
            "output": "LoRA (Low-Rank Adaptation) est une technique d'entraînement efficace qui n'entraîne qu'un petit nombre de paramètres additionnels au lieu de tout le modèle. Cela réduit considérablement les besoins en mémoire et en temps de calcul."
        },
        {
            "instruction": "Comment améliorer un modèle de langage?",
            "output": "On peut améliorer un modèle de langage via: 1) Le fine-tuning sur des données spécifiques, 2) L'augmentation de données, 3) L'ajustement des hyperparamètres, 4) L'utilisation de techniques comme RLHF pour l'aligner sur les préférences humaines."
        },
        {
            "instruction": "Explique-moi le machine learning.",
            "output": "Le machine learning est une branche de l'IA où les ordinateurs apprennent à partir de données sans être explicitement programmés. Le système identifie des patterns dans les données et fait des prédictions sur de nouvelles données."
        }
    ]
    
    # Sauvegarder
    output_path = "./example_dataset.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Dataset d'exemple créé: {output_path}")
    print(f"   Contient {len(examples)} exemples")
    print(f"\nPour l'utiliser:")
    print(f"   python train_qwen_chatbot.py --phase sft --dataset custom --dataset-path {output_path} --epochs 1")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Entraînement Qwen2-0.5B Chatbot")
    
    # Phase d'entraînement
    parser.add_argument(
        "--phase",
        type=str,
        required=True,
        choices=["sft", "test", "create-example"],
        help="Phase: sft (supervised), test, ou create-example"
    )
    
    # Dataset
    parser.add_argument("--dataset", type=str, default="custom", help="Dataset à utiliser")
    parser.add_argument("--dataset-path", type=str, help="Chemin vers dataset custom")
    
    # Checkpoint
    parser.add_argument("--checkpoint", type=str, help="Chemin vers checkpoint à charger")
    
    # Hyperparamètres
    parser.add_argument("--epochs", type=int, default=3, help="Nombre d'époques")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=512, help="Longueur max des séquences")
    
    # Options
    parser.add_argument("--no-lora", action="store_true", help="Désactiver LoRA")
    parser.add_argument("--wandb", action="store_true", help="Activer WandB logging")
    parser.add_argument("--output-dir", type=str, default="./checkpoints", help="Dossier de sortie")
    
    args = parser.parse_args()
    
    try:
        # Créer dataset d'exemple si demandé
        if args.phase == "create-example":
            create_example_dataset()
            return
        
        # Configuration
        config = TrainingConfig(
            output_dir=args.output_dir,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            max_seq_length=args.max_length,
            use_lora=not args.no_lora,
            use_wandb=args.wandb,
        )
        
        # Initialiser trainer
        trainer = QwenChatbotTrainer(config)
        
        # Phase de test
        if args.phase == "test":
            if not args.checkpoint:
                raise ValueError("--checkpoint requis pour le test")
            trainer.test_model(args.checkpoint)
            return
        
        # Charger dataset
        dataset_loader = DatasetLoader(trainer.tokenizer, max_length=config.max_seq_length)
        
        if args.phase == "sft":
            # Phase 1: SFT
            dataset = dataset_loader.load(args.dataset, args.dataset_path)
            trainer.train_sft(dataset)
        
        print("\n✅ Processus terminé!")
        
    except KeyboardInterrupt:
        print("\n\n👋 Interrompu par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
