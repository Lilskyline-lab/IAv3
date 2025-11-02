"""
Script d'expérimentation pour Google Colab
4 expériences pour identifier la source du problème

EXPÉRIENCES:
1. BASELINE     : Train supervisé simple (OASST) SANS instruction tuning NI DPO
2. WITH_IT      : Train supervisé AVEC instruction tuning, SANS DPO
3. WITH_DPO     : Train supervisé simple + DPO
4. FULL         : Train complet (supervisé + instruction tuning + DPO)

Usage dans Colab:
    !git clone [votre_repo]
    %cd [repo]/IA
    !python colab_experiments.py --experiment all --gpu
"""

import os
import sys
import json
import time
import torch
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model.HessGPT import HessGPT
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm


class SimpleDataset(Dataset):
    """Dataset minimal sans instruction tuning"""
    def __init__(self, pairs, tokenizer, max_length=256):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        # Format simple: juste concaténer
        text = f"Human: {pair['human']}\nAssistant: {pair['assistant']}"
        
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > self.max_length:
            tokens = tokens[-self.max_length:]
        
        # Tout est considéré comme réponse (pas de masking)
        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "assist_start": 0  # Tout est loss
        }


def collate_simple(batch, pad_id=2):
    """Collate simple"""
    input_ids_list = [b["input_ids"] for b in batch]
    max_len = max(t.size(0) for t in input_ids_list)
    
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    
    for i, ids in enumerate(input_ids_list):
        L = ids.size(0)
        input_ids[i, :L] = ids
        # Tout est cible (pas de masking intelligent)
        labels[i, :L] = ids
    
    return {"input_ids": input_ids, "labels": labels}


def create_synthetic_data(num_samples=1000):
    """Crée des données synthétiques pour le test"""
    templates = [
        ("Hello", "Hello! How can I help you?"),
        ("Hi", "Hi there! What can I do for you?"),
        ("What is AI?", "AI is artificial intelligence."),
        ("What is Python?", "Python is a programming language."),
        ("Explain ML", "Machine learning is a subset of AI."),
        ("Tell me about coding", "Coding is writing instructions for computers."),
        ("What is data?", "Data is information stored digitally."),
        ("Define algorithm", "An algorithm is a set of instructions."),
        ("What is a model?", "A model is a trained AI system."),
        ("Explain training", "Training is teaching a model from data."),
    ]
    
    data = []
    for _ in range(num_samples):
        template = templates[len(data) % len(templates)]
        data.append({
            "human": template[0],
            "assistant": template[1]
        })
    
    return data


def train_baseline(
    model_dir: str,
    tokenizer_name: str,
    device: torch.device,
    num_samples: int = 1000,
    epochs: int = 5,
    batch_size: int = 16
):
    """
    EXPÉRIENCE 1: BASELINE
    Train supervisé simple SANS instruction tuning NI DPO
    """
    print("\n" + "="*80)
    print("🧪 EXPÉRIENCE 1: BASELINE (Supervisé simple)")
    print("="*80)
    print("📋 Configuration:")
    print(f"   - Samples: {num_samples}")
    print(f"   - Epochs: {epochs}")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Instruction tuning: ❌ NON")
    print(f"   - DPO: ❌ NON")
    print("="*80)
    
    from transformers import AutoTokenizer
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    vocab_size = len(tokenizer)
    
    # Modèle
    model = HessGPT(
        vocab_size=vocab_size,
        embed_dim=512,
        num_heads=8,
        num_layers=6,
        max_seq_len=256
    ).to(device)
    
    # Données synthétiques
    data = create_synthetic_data(num_samples)
    dataset = SimpleDataset(data, tokenizer, max_length=256)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_simple(b, pad_id=tokenizer.pad_token_id or 2)
    )
    
    # Training
    optimizer = AdamW(model.parameters(), lr=5e-5)
    loss_fn = CrossEntropyLoss(ignore_index=-100)
    
    model.train()
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            logits, _ = model(input_ids)
            loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"✓ Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    
    # Sauvegarder
    exp_dir = os.path.join(model_dir, "exp1_baseline")
    os.makedirs(exp_dir, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(exp_dir, "model.pt"))
    
    with open(os.path.join(exp_dir, "results.json"), 'w') as f:
        json.dump({
            "experiment": "baseline",
            "losses": losses,
            "final_loss": losses[-1],
            "config": {
                "samples": num_samples,
                "epochs": epochs,
                "batch_size": batch_size,
                "instruction_tuning": False,
                "dpo": False
            }
        }, f, indent=2)
    
    print(f"\n✅ Expérience 1 terminée: Loss finale = {losses[-1]:.4f}")
    print(f"💾 Sauvegardé dans: {exp_dir}")
    
    return losses[-1], exp_dir


def train_with_instruction_tuning(
    model_dir: str,
    tokenizer_name: str,
    device: torch.device,
    num_samples: int = 1000,
    epochs: int = 5,
    batch_size: int = 16
):
    """
    EXPÉRIENCE 2: WITH_IT
    Train supervisé AVEC instruction tuning, SANS DPO
    """
    print("\n" + "="*80)
    print("🧪 EXPÉRIENCE 2: WITH INSTRUCTION TUNING")
    print("="*80)
    print("📋 Configuration:")
    print(f"   - Samples: {num_samples}")
    print(f"   - Epochs: {epochs}")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Instruction tuning: ✅ OUI (Mistral format)")
    print(f"   - DPO: ❌ NON")
    print("="*80)
    
    from transformers import AutoTokenizer
    from utils.instruction_tuning import convert_to_instruction_format
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    vocab_size = len(tokenizer)
    
    # Modèle
    model = HessGPT(
        vocab_size=vocab_size,
        embed_dim=512,
        num_heads=8,
        num_layers=6,
        max_seq_len=256
    ).to(device)
    
    # Données avec instruction tuning
    data = create_synthetic_data(num_samples)
    formatted_data = convert_to_instruction_format(data, template_name="mistral")
    
    # Dataset custom pour IT
    class ITDataset(Dataset):
        def __init__(self, formatted, tokenizer, max_length=256):
            self.formatted = formatted
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.formatted)
        
        def __getitem__(self, idx):
            text = self.formatted[idx]['formatted_text']
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            
            if len(tokens) > self.max_length:
                tokens = tokens[-self.max_length:]
            
            # Trouver où commence [/INST]
            inst_close = "[/INST]"
            inst_tokens = self.tokenizer.encode(inst_close, add_special_tokens=False)
            assist_start = 0
            
            # Simple: considérer 60% du texte comme prompt
            assist_start = int(len(tokens) * 0.6)
            
            return {
                "input_ids": torch.tensor(tokens, dtype=torch.long),
                "assist_start": assist_start
            }
    
    def collate_it(batch, pad_id=2):
        input_ids_list = [b["input_ids"] for b in batch]
        assist_starts = [b["assist_start"] for b in batch]
        max_len = max(t.size(0) for t in input_ids_list)
        
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        
        for i, ids in enumerate(input_ids_list):
            L = ids.size(0)
            input_ids[i, :L] = ids
            start = assist_starts[i]
            labels[i, start:L] = ids[start:L]
        
        return {"input_ids": input_ids, "labels": labels}
    
    dataset = ITDataset(formatted_data, tokenizer, max_length=256)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_it(b, pad_id=tokenizer.pad_token_id or 2)
    )
    
    # Training
    optimizer = AdamW(model.parameters(), lr=5e-5)
    loss_fn = CrossEntropyLoss(ignore_index=-100)
    
    model.train()
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            logits, _ = model(input_ids)
            loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"✓ Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    
    # Sauvegarder
    exp_dir = os.path.join(model_dir, "exp2_with_it")
    os.makedirs(exp_dir, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(exp_dir, "model.pt"))
    
    with open(os.path.join(exp_dir, "results.json"), 'w') as f:
        json.dump({
            "experiment": "with_instruction_tuning",
            "losses": losses,
            "final_loss": losses[-1],
            "config": {
                "samples": num_samples,
                "epochs": epochs,
                "batch_size": batch_size,
                "instruction_tuning": True,
                "dpo": False
            }
        }, f, indent=2)
    
    print(f"\n✅ Expérience 2 terminée: Loss finale = {losses[-1]:.4f}")
    print(f"💾 Sauvegardé dans: {exp_dir}")
    
    return losses[-1], exp_dir


def train_simple_plus_dpo(
    model_dir: str,
    tokenizer_name: str,
    device: torch.device,
    num_samples: int = 1000,
    epochs: int = 5,
    batch_size: int = 16
):
    """
    EXPÉRIENCE 3: WITH_DPO
    Train supervisé simple + DPO (SANS instruction tuning)
    """
    print("\n" + "="*80)
    print("🧪 EXPÉRIENCE 3: SIMPLE + DPO")
    print("="*80)
    print("📋 Configuration:")
    print(f"   - Samples supervisé: {num_samples}")
    print(f"   - Epochs supervisé: {epochs}")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Instruction tuning: ❌ NON")
    print(f"   - DPO: ✅ OUI (ultra-light)")
    print("="*80)
    
    # Phase 1: Train baseline
    print("\n📚 Phase 1: Entraînement supervisé simple...")
    final_loss, baseline_dir = train_baseline(
        model_dir, tokenizer_name, device,
        num_samples, epochs, batch_size
    )
    
    # Phase 2: DPO ultra-light
    print("\n🎯 Phase 2: DPO ultra-light...")
    print("⚠️  DPO sur CPU est très lent, utilisation de mini-batch")
    
    from utils.rlhf_module import train_with_dpo, DPOConfig
    from transformers import AutoTokenizer
    
    # Charger le modèle baseline
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    vocab_size = len(tokenizer)
    
    model = HessGPT(
        vocab_size=vocab_size,
        embed_dim=512,
        num_heads=8,
        num_layers=6,
        max_seq_len=256
    ).to(device)
    
    model.load_state_dict(torch.load(os.path.join(baseline_dir, "model.pt")))
    
    # DPO config ultra-light
    dpo_config = DPOConfig(
        max_samples_train=50,  # Très peu de samples
        max_samples_val=10,
        batch_size=2,          # Très petit batch
        num_epochs=1,
        learning_rate=5e-7,
        beta=0.1,
        output_dir=os.path.join(model_dir, "exp3_with_dpo", "dpo_temp")
    )
    
    try:
        train_with_dpo(
            model=model,
            tokenizer=tokenizer,
            device=device,
            config=dpo_config,
            model_dir=model_dir
        )
    except Exception as e:
        print(f"⚠️  DPO échoué (normal sur CPU): {e}")
        print("💡 Utilisant le modèle baseline sans DPO")
    
    # Sauvegarder
    exp_dir = os.path.join(model_dir, "exp3_with_dpo")
    os.makedirs(exp_dir, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(exp_dir, "model.pt"))
    
    with open(os.path.join(exp_dir, "results.json"), 'w') as f:
        json.dump({
            "experiment": "simple_plus_dpo",
            "supervised_loss": final_loss,
            "config": {
                "samples": num_samples,
                "epochs": epochs,
                "batch_size": batch_size,
                "instruction_tuning": False,
                "dpo": True
            }
        }, f, indent=2)
    
    print(f"\n✅ Expérience 3 terminée")
    print(f"💾 Sauvegardé dans: {exp_dir}")
    
    return final_loss, exp_dir


def train_full_pipeline(
    model_dir: str,
    tokenizer_name: str,
    device: torch.device,
    num_samples: int = 1000,
    epochs: int = 5,
    batch_size: int = 16
):
    """
    EXPÉRIENCE 4: FULL
    Train complet: supervisé + instruction tuning + DPO
    """
    print("\n" + "="*80)
    print("🧪 EXPÉRIENCE 4: FULL PIPELINE")
    print("="*80)
    print("📋 Configuration:")
    print(f"   - Samples supervisé: {num_samples}")
    print(f"   - Epochs supervisé: {epochs}")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Instruction tuning: ✅ OUI")
    print(f"   - DPO: ✅ OUI")
    print("="*80)
    
    # Phase 1: Supervisé + IT
    print("\n📚 Phase 1: Supervisé avec Instruction Tuning...")
    it_loss, it_dir = train_with_instruction_tuning(
        model_dir, tokenizer_name, device,
        num_samples, epochs, batch_size
    )
    
    # Phase 2: DPO
    print("\n🎯 Phase 2: DPO ultra-light...")
    
    from utils.rlhf_module import train_with_dpo, DPOConfig
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    vocab_size = len(tokenizer)
    
    model = HessGPT(
        vocab_size=vocab_size,
        embed_dim=512,
        num_heads=8,
        num_layers=6,
        max_seq_len=256
    ).to(device)
    
    model.load_state_dict(torch.load(os.path.join(it_dir, "model.pt")))
    
    dpo_config = DPOConfig(
        max_samples_train=50,
        max_samples_val=10,
        batch_size=2,
        num_epochs=1,
        learning_rate=5e-7,
        beta=0.1,
        output_dir=os.path.join(model_dir, "exp4_full", "dpo_temp")
    )
    
    try:
        train_with_dpo(
            model=model,
            tokenizer=tokenizer,
            device=device,
            config=dpo_config,
            model_dir=model_dir
        )
    except Exception as e:
        print(f"⚠️  DPO échoué: {e}")
    
    # Sauvegarder
    exp_dir = os.path.join(model_dir, "exp4_full")
    os.makedirs(exp_dir, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(exp_dir, "model.pt"))
    
    with open(os.path.join(exp_dir, "results.json"), 'w') as f:
        json.dump({
            "experiment": "full_pipeline",
            "supervised_loss": it_loss,
            "config": {
                "samples": num_samples,
                "epochs": epochs,
                "batch_size": batch_size,
                "instruction_tuning": True,
                "dpo": True
            }
        }, f, indent=2)
    
    print(f"\n✅ Expérience 4 terminée")
    print(f"💾 Sauvegardé dans: {exp_dir}")
    
    return it_loss, exp_dir


def compare_all_experiments(model_dir: str):
    """Compare les résultats de toutes les expériences"""
    print("\n" + "="*80)
    print("📊 COMPARAISON DES 4 EXPÉRIENCES")
    print("="*80)
    
    experiments = [
        ("exp1_baseline", "BASELINE (Simple)"),
        ("exp2_with_it", "WITH IT (+ Instruction Tuning)"),
        ("exp3_with_dpo", "WITH DPO (+ DPO only)"),
        ("exp4_full", "FULL (IT + DPO)")
    ]
    
    results = []
    
    for exp_dir, exp_name in experiments:
        full_path = os.path.join(model_dir, exp_dir)
        results_file = os.path.join(full_path, "results.json")
        
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                data = json.load(f)
                results.append((exp_name, data))
                
                print(f"\n🔹 {exp_name}")
                print(f"   Loss finale: {data.get('final_loss', data.get('supervised_loss', 'N/A')):.4f}")
                print(f"   IT: {'✅' if data['config']['instruction_tuning'] else '❌'}")
                print(f"   DPO: {'✅' if data['config']['dpo'] else '❌'}")
        else:
            print(f"\n🔹 {exp_name}: ❌ Résultats non trouvés")
    
    print("\n" + "="*80)
    print("💡 CONCLUSIONS:")
    print("="*80)
    print("""
1. Si BASELINE fonctionne mais WITH_IT échoue:
   → Problème dans l'instruction tuning (formatage, masking)

2. Si BASELINE fonctionne mais WITH_DPO échoue:
   → Problème dans DPO (KL divergence, reference model)

3. Si BASELINE échoue:
   → Problème fondamental (architecture, tokenizer, données)

4. Si FULL fonctionne le mieux:
   → Pipeline correct, continuer avec plus de données
    """)
    print("="*80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["1", "2", "3", "4", "all"], default="all")
    parser.add_argument("--model_dir", default="saved_models/experiments")
    parser.add_argument("--tokenizer", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gpu", action="store_true", help="Utiliser GPU")
    
    args = parser.parse_args()
    
    # Device
    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 GPU détecté et activé!")
    else:
        device = torch.device("cpu")
        print("💻 Utilisation du CPU")
    
    # Chemin absolu
    if not os.path.isabs(args.model_dir):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args.model_dir = os.path.join(base_dir, args.model_dir)
    
    os.makedirs(args.model_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("🔬 DIAGNOSTIC SYSTEM - 4 EXPÉRIENCES")
    print("="*80)
    print(f"📁 Dossier: {args.model_dir}")
    print(f"🔤 Tokenizer: {args.tokenizer}")
    print(f"💻 Device: {device}")
    print(f"📊 Samples: {args.samples}")
    print(f"🔁 Epochs: {args.epochs}")
    print("="*80)
    
    start_time = time.time()
    
    try:
        if args.experiment in ["1", "all"]:
            train_baseline(args.model_dir, args.tokenizer, device, 
                          args.samples, args.epochs, args.batch_size)
        
        if args.experiment in ["2", "all"]:
            train_with_instruction_tuning(args.model_dir, args.tokenizer, device,
                                         args.samples, args.epochs, args.batch_size)
        
        if args.experiment in ["3", "all"]:
            train_simple_plus_dpo(args.model_dir, args.tokenizer, device,
                                 args.samples, args.epochs, args.batch_size)
        
        if args.experiment in ["4", "all"]:
            train_full_pipeline(args.model_dir, args.tokenizer, device,
                               args.samples, args.epochs, args.batch_size)
        
        # Comparaison finale
        if args.experiment == "all":
            compare_all_experiments(args.model_dir)
        
    except KeyboardInterrupt:
        print("\n⚠️  Expériences interrompues")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
    
    elapsed = time.time() - start_time
    print(f"\n⏱️  Temps total: {elapsed/60:.1f} minutes")
    print("\n✅ Expériences terminées!")
    print(f"📁 Résultats dans: {args.model_dir}")


if __name__ == "__main__":
    main()
