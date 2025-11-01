"""
Système d'entraînement avec OASST1/2 + Instruction Tuning + RLHF
Sources : OASST1 + OASST2 (dialogues Hugging Face) + RLHF (Anthropic/hh-rlhf)

TOKENIZERS OPEN-SOURCE SUPPORTÉS (32k-50k vocab):
- gpt2: 50,257 tokens (OpenAI, complètement open-source)
- mistralai/Mistral-7B-v0.1: 32,000 tokens (NON-GATED)
- meta-llama/Llama-3.2-1B: 128,256 tokens (NON-GATED, plus récent)
- EleutherAI/gpt-neox-20b: 50,432 tokens (open-source)
"""

import os
import sys
import json
import time
from tqdm import tqdm
from typing import List, Dict, Optional
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model.HessGPT import HessGPT
from utils.instruction_tuning import (
    InstructionTemplates,
    convert_to_instruction_format,
    InstructionDatasetLoader
)
from utils.rlhf_module import RLHFTrainer, RLHFConfig

try:
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from huggingface_hub.errors import GatedRepoError
    HF_AVAILABLE = True
except ImportError:
    print("⚠️ 'datasets' et 'transformers' non installés.")
    print("Installez avec: pip install datasets transformers")
    HF_AVAILABLE = False


# =============================================================================
# TOKENIZERS OPEN-SOURCE RECOMMANDÉS (32k-50k vocab, NON-GATED)
# =============================================================================
RECOMMENDED_TOKENIZERS = {
    "gpt2": {
        "vocab_size": 50257,
        "description": "GPT-2 OpenAI (50k vocab, totalement open-source)",
        "gated": False
    },
    "mistralai/Mistral-7B-v0.1": {
        "vocab_size": 32000,
        "description": "Mistral 7B (32k vocab, NON-GATED)",
        "gated": False
    },
    "EleutherAI/gpt-neox-20b": {
        "vocab_size": 50432,
        "description": "GPT-NeoX (50k vocab, EleutherAI open-source)",
        "gated": False
    },
    "meta-llama/Llama-3.2-1B": {
        "vocab_size": 128256,
        "description": "Llama 3.2 (128k vocab, NON-GATED, plus récent)",
        "gated": False
    }
}

# Tokenizer par défaut (le plus safe et performant pour 32k-50k)
DEFAULT_TOKENIZER = "mistralai/Mistral-7B-v0.1"


def print_available_tokenizers():
    """Affiche les tokenizers disponibles"""
    print("\n" + "="*70)
    print("📋 TOKENIZERS OPEN-SOURCE DISPONIBLES (32k-50k vocab)")
    print("="*70)
    for name, info in RECOMMENDED_TOKENIZERS.items():
        print(f"\n🔹 {name}")
        print(f"   Vocab: {info['vocab_size']:,} tokens")
        print(f"   {info['description']}")
        print(f"   Gated: {'❌ OUI' if info['gated'] else '✅ NON'}")
    print("\n" + "="*70)
    print(f"💡 Par défaut: {DEFAULT_TOKENIZER}")
    print("="*70 + "\n")


def load_hf_tokenizer(tokenizer_name=None, use_fast=True):
    """
    Charge un tokenizer Hugging Face (32k-50k vocab, NON-GATED)
    
    Args:
        tokenizer_name: Nom du tokenizer (None = utilise DEFAULT_TOKENIZER)
        use_fast: Utiliser la version rapide (Rust)
    
    Returns:
        Tokenizer HuggingFace avec méthodes wrapper
    """
    if not HF_AVAILABLE:
        raise ImportError("Installez transformers: pip install transformers")
    
    # Utiliser le tokenizer par défaut si non spécifié
    if tokenizer_name is None:
        tokenizer_name = DEFAULT_TOKENIZER
        print(f"ℹ️  Utilisation du tokenizer par défaut: {tokenizer_name}")
    
    # Vérifier si le tokenizer est dans nos recommandations
    if tokenizer_name not in RECOMMENDED_TOKENIZERS:
        print(f"⚠️  ATTENTION: '{tokenizer_name}' n'est pas dans la liste recommandée")
        print_available_tokenizers()
        response = input(f"Continuer avec '{tokenizer_name}' ? (y/N): ")
        if response.lower() != 'y':
            print("❌ Chargement annulé")
            sys.exit(1)
    
    print(f"\n🔤 Chargement du tokenizer depuis HuggingFace: {tokenizer_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=use_fast,
            trust_remote_code=True
        )
    except (GatedRepoError, OSError) as e:
        if "gated" in str(e).lower() or "restricted" in str(e).lower():
            print(f"\n❌ ERREUR: Le modèle '{tokenizer_name}' est à accès restreint (GATED)!")
            print("\n📋 Solutions:")
            print("   1. Utilisez un tokenizer NON-GATED:")
            print_available_tokenizers()
            print("\n   2. Ou authentifiez-vous:")
            print("      - Créez un token sur https://huggingface.co/settings/tokens")
            print("      - Exécutez: huggingface-cli login")
            print(f"      - Acceptez les conditions: https://huggingface.co/{tokenizer_name}")
            raise
        else:
            raise
    
    # Assurer la compatibilité avec le code existant
    if not hasattr(tokenizer, 'encoder'):
        def encoder_wrapper(text):
            return tokenizer.encode(text, add_special_tokens=False)
        tokenizer.encoder = encoder_wrapper
    
    if not hasattr(tokenizer, 'decoder'):
        def decoder_wrapper(ids):
            return tokenizer.decode(ids, skip_special_tokens=False)
        tokenizer.decoder = decoder_wrapper
    
    # Définir les tokens spéciaux si manquants
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    vocab_size = len(tokenizer)
    print(f"✅ Tokenizer chargé: {vocab_size:,} tokens")
    print(f"   - PAD: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"   - EOS: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    if tokenizer.bos_token:
        print(f"   - BOS: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
    
    # Vérifier la plage de vocab
    if vocab_size < 30000:
        print(f"⚠️  Vocab size ({vocab_size:,}) < 30k - Performance sous-optimale")
    elif 30000 <= vocab_size <= 60000:
        print(f"✅ Vocab size optimal pour entraînement (32k-50k)")
    else:
        print(f"ℹ️  Vocab size élevé ({vocab_size:,}) - Mémoire accrue")
    
    return tokenizer


class OASSTDialogueLoader:
    """Loader pour OASST1 et OASST2"""

    def __init__(self, version='oasst1', language='en', max_samples=None):
        self.version = version
        self.language = language
        self.max_samples = max_samples
        self.dataset = None

        if not HF_AVAILABLE:
            print("⚠️ Hugging Face datasets non disponible")
            return

        self._load_dataset()

    def _load_dataset(self):
        print(f"\n📦 Chargement du dataset {self.version.upper()} depuis Hugging Face...")
        try:
            if self.version == 'oasst1':
                dataset_name = "OpenAssistant/oasst1"
            else:  # oasst2
                dataset_name = "OpenAssistant/oasst2"

            self.dataset = load_dataset(dataset_name, split="train")

            if self.language != 'all':
                print(f"🔍 Filtrage pour langue: {self.language}")
                self.dataset = self.dataset.filter(
                    lambda x: x.get('lang', 'en') == self.language
                )

            if self.max_samples:
                self.dataset = self.dataset.select(range(min(self.max_samples, len(self.dataset))))

            print(f"✅ {len(self.dataset)} conversations chargées depuis {self.version.upper()}")

        except Exception as e:
            print(f"❌ Erreur lors du chargement {self.version}: {e}")
            self.dataset = None

    def extract_dialogues(self) -> List[Dict]:
        """Extrait les paires question/réponse"""
        if self.dataset is None:
            return self._get_fallback_dialogues()

        dialogues = []
        print(f"\n💬 Extraction des dialogues {self.version.upper()}...")

        for item in tqdm(self.dataset, desc=f"Parsing {self.version.upper()}"):
            if item.get('role') == 'prompter' and item.get('text'):
                prompt = item['text']
                message_id = item.get('message_id')

                if message_id:
                    for potential_response in self.dataset:
                        if (potential_response.get('parent_id') == message_id and 
                            potential_response.get('role') == 'assistant'):
                            response = potential_response.get('text', '')
                            if response:
                                dialogues.append({
                                    'human': prompt.strip(),
                                    'assistant': response.strip(),
                                    'source': self.version
                                })
                            break

        print(f"✅ {len(dialogues)} paires extraites depuis {self.version.upper()}")
        return dialogues

    def _get_fallback_dialogues(self) -> List[Dict]:
        """Dialogues de fallback si le dataset n'est pas disponible"""
        fallback = [
            ("Hello", "Hello! How can I help you today?"),
            ("What is machine learning?", "Machine learning is a branch of AI that enables systems to learn from data."),
            ("Explain neural networks", "Neural networks are computing systems inspired by biological neural networks."),
        ]
        return [{"human": h, "assistant": a, "source": "fallback"} for h, a in fallback]


class InstructionTunedDataset(Dataset):
    """Dataset avec instruction tuning appliqué"""

    def __init__(self, pairs, tokenizer, max_length=512, instruction_template="chat_bot"):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.instruction_template = instruction_template

        print(f"\n🎯 Application de l'instruction tuning (template: {instruction_template})")
        self.formatted_pairs = convert_to_instruction_format(
            pairs,
            template_name=instruction_template
        )
        print(f"✅ {len(self.formatted_pairs)} exemples formatés")

    def __len__(self):
        return len(self.formatted_pairs)

    def __getitem__(self, idx):
        formatted_text = self.formatted_pairs[idx]['formatted_text']
        h = self.pairs[idx]['human'].strip()
        a = self.pairs[idx]['assistant'].strip()

        prefix = f"Human: {h}\nBot:"
        
        if hasattr(self.tokenizer, 'encode'):
            ids_prefix = self.tokenizer.encode(prefix, add_special_tokens=False)
            ids_all = self.tokenizer.encode(formatted_text, add_special_tokens=False)
        else:
            ids_prefix = self.tokenizer.encoder(prefix)
            ids_all = self.tokenizer.encoder(formatted_text)

        if len(ids_all) > self.max_length:
            ids_all = ids_all[-self.max_length:]

        if hasattr(self.tokenizer, 'encode'):
            ids_assistant = self.tokenizer.encode(a, add_special_tokens=False)
        else:
            ids_assistant = self.tokenizer.encoder(a)
        
        assist_start = max(0, len(ids_all) - len(ids_assistant))
        
        return {
            "input_ids": torch.tensor(ids_all, dtype=torch.long),
            "assist_start": assist_start
        }


def collate_fn(batch, pad_id=0):
    """Collate function pour le DataLoader"""
    input_ids_list = [b["input_ids"] for b in batch]
    assist_starts = [b["assist_start"] for b in batch]
    max_len = max([t.size(0) for t in input_ids_list])

    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, ids in enumerate(input_ids_list):
        L = ids.size(0)
        input_ids[i, :L] = ids
        attention_mask[i, :L] = 1
        start = assist_starts[i]
        labels[i, start:L] = input_ids[i, start:L]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class OASSTTrainer:
    """Trainer utilisant OASST1/2 + Instruction Tuning + RLHF"""

    def __init__(
        self,
        model_dir,
        tokenizer_name=None,  # None = utilise DEFAULT_TOKENIZER
        device=None,
        language='en',
        instruction_template="chat_bot",
        custom_data_dir=None
    ):
        self.model_dir = model_dir
        self.tokenizer_name = tokenizer_name or DEFAULT_TOKENIZER
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.language = language
        self.instruction_template = instruction_template
        self.custom_data_dir = custom_data_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data'
        )

        os.makedirs(model_dir, exist_ok=True)

        self.model, self.tokenizer, self.config = self._load_or_init_model()

        self.history_file = os.path.join(model_dir, "training_history.json")
        self.history = self._load_history()

        print(f"\n✅ Trainer initialisé avec instruction tuning (template: {instruction_template})")

    def _load_or_init_model(self):
        cfg_path = os.path.join(self.model_dir, "config.json")
        model_path = os.path.join(self.model_dir, "model.pt")

        tokenizer = load_hf_tokenizer(self.tokenizer_name)
        vocab_size = len(tokenizer)

        if os.path.exists(cfg_path):
            with open(cfg_path, 'r') as f:
                cfg = json.load(f)
            
            if cfg["vocab_size"] != vocab_size:
                print(f"⚠️ Mise à jour vocab_size: {cfg['vocab_size']} → {vocab_size}")
                cfg["vocab_size"] = vocab_size
                with open(cfg_path, 'w') as f:
                    json.dump(cfg, f, indent=2)
        else:
            cfg = {
                "vocab_size": vocab_size,
                "embed_dim": 512,
                "num_heads": 8,
                "num_layers": 6,
                "max_seq_len": 512,
                "tokenizer_name": self.tokenizer_name
            }
            with open(cfg_path, 'w') as f:
                json.dump(cfg, f, indent=2)
            print(f"🆕 Configuration créée avec vocab_size={vocab_size}")

        model = HessGPT(
            vocab_size=cfg["vocab_size"],
            embed_dim=cfg["embed_dim"],
            num_heads=cfg["num_heads"],
            num_layers=cfg["num_layers"],
            max_seq_len=cfg["max_seq_len"]
        )

        if os.path.exists(model_path):
            print(f"✅ Chargement du modèle existant : {model_path}")
            try:
                state = torch.load(model_path, map_location=self.device, weights_only=True)
            except TypeError:
                state = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state)
        else:
            print("🆕 Initialisation d'un nouveau modèle")

        model.to(self.device)
        return model, tokenizer, cfg

    def _load_history(self):
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return {
            "cycles": [],
            "total_examples_trained": 0,
            "instruction_template_used": self.instruction_template,
            "rlhf_cycles": 0,
            "tokenizer_name": self.tokenizer_name
        }

    def _save_history(self):
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)

    def load_custom_data(self) -> List[Dict]:
        """Charge les données personnalisées depuis data/"""
        custom_data = []

        if not os.path.exists(self.custom_data_dir):
            print(f"⚠️ Dossier de données personnalisées non trouvé: {self.custom_data_dir}")
            return custom_data

        print(f"\n📂 Recherche de fichiers personnalisés dans: {self.custom_data_dir}")

        for filename in os.listdir(self.custom_data_dir):
            if filename.endswith(('.json', '.jsonl', '.csv')):
                filepath = os.path.join(self.custom_data_dir, filename)
                try:
                    data = InstructionDatasetLoader.load_dataset(filepath)
                    custom_data.extend(data)
                    print(f"✅ {len(data)} exemples chargés depuis {filename}")
                except Exception as e:
                    print(f"⚠️ Erreur lors du chargement de {filename}: {e}")

        if custom_data:
            print(f"✅ Total: {len(custom_data)} exemples personnalisés chargés")

        return custom_data

    def generate_dataset(
        self,
        num_oasst1=500,
        num_oasst2=500,
        repeat_important=3
    ):
        """Génère le dataset depuis OASST1 et OASST2 (anglais)"""
        print("\n" + "="*60)
        print("🔄 GÉNÉRATION DATASET OASST1/2 (ENGLISH)")
        print("="*60)

        dataset = []
        dialogues1 = []
        dialogues2 = []

        if num_oasst1 > 0:
            print("\n💬 Source 1: OASST1 (English)")
            loader1 = OASSTDialogueLoader('oasst1', self.language, num_oasst1)
            dialogues1 = loader1.extract_dialogues()
            dataset.extend(dialogues1 * repeat_important)

        if num_oasst2 > 0:
            print("\n💬 Source 2: OASST2 (English)")
            loader2 = OASSTDialogueLoader('oasst2', self.language, num_oasst2)
            dialogues2 = loader2.extract_dialogues()
            dataset.extend(dialogues2 * repeat_important)

        random.shuffle(dataset)

        print(f"\n" + "="*60)
        print(f"✅ DATASET TOTAL: {len(dataset)} exemples")
        print(f"   - OASST1: {len(dialogues1) * repeat_important}")
        print(f"   - OASST2: {len(dialogues2) * repeat_important}")
        print(f"🎯 Instruction Template: {self.instruction_template}")
        print(f"🔤 Tokenizer: {self.tokenizer_name} ({len(self.tokenizer):,} tokens)")
        print("="*60)

        return dataset

    def train_one_cycle(
        self,
        num_oasst1=500,
        num_oasst2=500,
        epochs=3,
        batch_size=8,
        lr=5e-5
    ):
        """Entraînement supervisé avec OASST1/2"""
        print("\n" + "="*70)
        print("🚀 CYCLE D'ENTRAÎNEMENT SUPERVISÉ (OASST1/2 - ENGLISH)")
        print("="*70)

        dataset_pairs = self.generate_dataset(
            num_oasst1, num_oasst2, repeat_important=3
        )

        if not dataset_pairs:
            print("❌ Dataset vide, abandon du cycle")
            return

        dataset = InstructionTunedDataset(
            dataset_pairs,
            self.tokenizer,
            max_length=self.config["max_seq_len"],
            instruction_template=self.instruction_template
        )

        pad_id = getattr(self.tokenizer, 'pad_token_id', 0) or 0

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_fn(b, pad_id=pad_id)
        )

        optimizer = AdamW(self.model.parameters(), lr=lr)
        loss_fn = CrossEntropyLoss(ignore_index=-100)

        self.model.train()

        total_loss = 0
        step = 0

        print(f"\n⏳ Entraînement sur {len(dataset)} exemples, {epochs} époques")

        for epoch in range(epochs):
            epoch_loss = 0
            pbar = tqdm(dataloader, desc=f"Époque {epoch+1}/{epochs}")

            for batch in pbar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits, _ = self.model(input_ids)

                loss = loss_fn(
                    logits.view(-1, self.config["vocab_size"]),
                    labels.view(-1)
                )

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                epoch_loss += loss.item()
                step += 1

                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_epoch_loss = epoch_loss / len(dataloader)
            print(f"✓ Époque {epoch+1} terminée - Loss moyenne: {avg_epoch_loss:.4f}")

        avg_loss = total_loss / step

        torch.save(self.model.state_dict(), os.path.join(self.model_dir, "model.pt"))
        print(f"\n✅ Modèle sauvegardé: {self.model_dir}/model.pt")

        cycle_info = {
            "cycle": len(self.history["cycles"]) + 1,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "examples": len(dataset),
            "epochs": epochs,
            "avg_loss": avg_loss,
            "instruction_template": self.instruction_template,
            "tokenizer_name": self.tokenizer_name
        }
        self.history["cycles"].append(cycle_info)
        self.history["total_examples_trained"] += len(dataset)
        self._save_history()

        print("\n" + "="*70)
        print(f"✅ CYCLE SUPERVISÉ TERMINÉ")
        print(f"   Loss moyenne: {avg_loss:.4f}")
        print(f"   Total exemples: {self.history['total_examples_trained']}")
        print("="*70)

    def train_with_rlhf(
        self,
        max_samples=5000,
        epochs=1,
        batch_size=4,
        lr=1.41e-5
    ):
        """Entraînement RLHF après le supervisé"""
        print("\n" + "="*70)
        print("🎯 LANCEMENT ENTRAÎNEMENT RLHF")
        print("="*70)

        rlhf_config = RLHFConfig(
            dataset_name="Anthropic/hh-rlhf",
            max_samples_train=max_samples,
            max_samples_val=int(max_samples * 0.1),
            batch_size=batch_size,
            mini_batch_size=max(1, batch_size // 4),
            num_train_epochs=epochs,
            learning_rate=lr,
            output_dir=os.path.join(self.model_dir, "rlhf_output")
        )

        rlhf_trainer = RLHFTrainer(
            base_model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            rlhf_config=rlhf_config,
            model_dir=self.model_dir
        )

        rlhf_trainer.train_with_rlhf()

        self.history["rlhf_cycles"] += 1
        self._save_history()

        print(f"\n✅ RLHF terminé ({self.history['rlhf_cycles']} cycles)")


def main():
    print("\n" + "="*70)
    print("🤖 SYSTÈME D'ENTRAÎNEMENT OASST1/2 + RLHF + INSTRUCTION TUNING")
    print("="*70)
    
    # Afficher les tokenizers disponibles
    print_available_tokenizers()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Device: {device}")

    model_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "saved_models",
        "my_llm"
    )
    print(f"📁 Model directory: {model_dir}")

    # OPTION 1: Mistral 7B (32k vocab, NON-GATED, recommandé)
    tokenizer_name = "mistralai/Mistral-7B-v0.1"
    
    # OPTION 2: GPT-2 (50k vocab, totalement open-source)
    # tokenizer_name = "gpt2"
    
    # OPTION 3: GPT-NeoX (50k vocab, EleutherAI)
    # tokenizer_name = "EleutherAI/gpt-neox-20b"
    
    # OPTION 4: Llama 3.2 (128k vocab, NON-GATED, plus récent)
    # tokenizer_name = "meta-llama/Llama-3.2-1B"
    
    # OPTION 5: Laisser None pour utiliser DEFAULT_TOKENIZER
    # tokenizer_name = None

    print(f"🔤 Tokenizer choisi: {tokenizer_name or DEFAULT_TOKENIZER}")

    trainer = OASSTTrainer(
        model_dir=model_dir,
        tokenizer_name=tokenizer_name,
        device=device,
        language='en',
        instruction_template="chat_bot"
    )

    print("\n🎯 Phase 1: Entraînement supervisé (OASST1/2)")
    print("   - 500 dialogues OASST1 (English)")
    print("   - 500 dialogues OASST2 (English)")
    print("   - 3 époques")
    print(f"   - Tokenizer: {tokenizer_name or DEFAULT_TOKENIZER}")

    trainer.train_one_cycle(
        num_oasst1=500,
        num_oasst2=500,
        epochs=3,
        batch_size=4,
        lr=5e-5
    )

    print("\n🎯 Phase 2: RLHF (Anthropic/hh-rlhf)")
    print("   - 5000 samples")
    print("   - 1 époque")

    trainer.train_with_rlhf(
        max_samples=5000,
        epochs=1,
        batch_size=4,
        lr=1.41e-5
    )

    print("\n✅ Entraînement complet terminé!")


if __name__ == "__main__":
    main()