"""
Script de réentraînement progressif avec diagnostic
Augmente graduellement la complexité et les données
"""

import os
import sys
import torch
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.FineTuning import OASSTTrainer, DPOConfig


def diagnose_model(trainer):
    """Diagnostic rapide du modèle"""
    print("\n" + "="*70)
    print("🔍 DIAGNOSTIC DU MODÈLE")
    print("="*70)
    
    # Charger l'historique
    history = trainer.history
    
    print(f"📊 Cycles d'entraînement: {len(history['cycles'])}")
    print(f"📚 Total exemples vus: {history['total_examples_trained']}")
    print(f"🎯 DPO cycles: {history['dpo_cycles']}")
    
    if history['cycles']:
        last_cycle = history['cycles'][-1]
        print(f"\n📈 Dernier cycle:")
        print(f"   Loss: {last_cycle['avg_loss']:.4f}")
        print(f"   Exemples: {last_cycle['examples']}")
        print(f"   Époques: {last_cycle['epochs']}")
    
    # Évaluation
    print(f"\n💡 Recommandations:")
    
    if history['total_examples_trained'] < 10000:
        print("   ⚠️  CRITIQUE: Moins de 10k exemples")
        print("      → Entraînez avec au moins 10k-20k exemples")
        recommendation = "train_more"
    elif history['total_examples_trained'] < 50000:
        print("   ⚠️  Entraînement léger (< 50k exemples)")
        print("      → Continuez l'entraînement supervisé")
        recommendation = "continue"
    else:
        print("   ✅ Entraînement supervisé suffisant")
        if history['dpo_cycles'] == 0:
            print("      → Passez au DPO pour l'alignment")
            recommendation = "dpo"
        else:
            print("      → Modèle bien entraîné!")
            recommendation = "done"
    
    print("="*70)
    return recommendation


def test_generation(trainer, prompts):
    """Test rapide de génération"""
    print("\n" + "="*70)
    print("🧪 TEST DE GÉNÉRATION")
    print("="*70)
    
    trainer.model.eval()
    
    for prompt in prompts:
        print(f"\n👤 Prompt: {prompt}")
        print("🤖 Génération: ", end='', flush=True)
        
        # Encoder
        formatted = f"[INST] {prompt} [/INST]"
        input_ids = trainer.tokenizer.encode(formatted, add_special_tokens=False)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(trainer.device)
        
        # Générer
        with torch.no_grad():
            generated = []
            current = input_tensor
            
            for _ in range(50):  # 50 tokens max
                logits, _ = trainer.model(current)
                next_logits = logits[0, -1, :]
                
                # Temperature + sampling
                next_logits = next_logits / 0.7
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                if next_token.item() == trainer.tokenizer.eos_token_id:
                    break
                
                generated.append(next_token.item())
                current = torch.cat([current, next_token.unsqueeze(0)], dim=1)
            
            if generated:
                response = trainer.tokenizer.decode(generated, skip_special_tokens=True)
                print(response[:200])
            else:
                print("[AUCUNE GÉNÉRATION]")
    
    print("\n" + "="*70)


def progressive_training(trainer, phase):
    """Entraînement progressif selon la phase"""
    
    if phase == "phase1":
        print("\n" + "="*70)
        print("📚 PHASE 1: Foundation (10k exemples)")
        print("="*70)
        trainer.train_one_cycle(
            num_oasst1=5000,
            num_oasst2=5000,
            epochs=3,
            batch_size=8,
            lr=5e-5
        )
    
    elif phase == "phase2":
        print("\n" + "="*70)
        print("📚 PHASE 2: Renforcement (20k exemples)")
        print("="*70)
        trainer.train_one_cycle(
            num_oasst1=10000,
            num_oasst2=10000,
            epochs=4,
            batch_size=8,
            lr=3e-5  # LR plus petit
        )
    
    elif phase == "phase3":
        print("\n" + "="*70)
        print("🎯 PHASE 3: DPO Alignment")
        print("="*70)
        trainer.train_with_dpo(
            max_samples=10000,
            epochs=2,
            batch_size=4,
            lr=5e-7,
            beta=0.1
        )
    
    elif phase == "phase4":
        print("\n" + "="*70)
        print("🎯 PHASE 4: DPO Fine-tuning")
        print("="*70)
        trainer.train_with_dpo(
            max_samples=20000,
            epochs=1,
            batch_size=4,
            lr=1e-7,  # Très petit LR pour fine-tuning
            beta=0.15
        )


def main():
    print("\n" + "="*70)
    print("🔄 RÉENTRAÎNEMENT PROGRESSIF")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Device: {device}")
    
    model_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "saved_models",
        "my_llm"
    )
    
    tokenizer_name = "mistralai/Mistral-7B-v0.1"
    
    # Créer le trainer
    trainer = OASSTTrainer(
        model_dir=model_dir,
        tokenizer_name=tokenizer_name,
        device=device,
        language='en',
        instruction_template=None
    )
    
    # Diagnostic
    recommendation = diagnose_model(trainer)
    
    # Test de génération avant
    print("\n📊 TEST AVANT ENTRAÎNEMENT:")
    test_prompts = [
        "Hello!",
        "What is AI?",
        "Tell me a joke"
    ]
    test_generation(trainer, test_prompts)
    
    # Demander confirmation
    print("\n" + "="*70)
    print("🤔 PLAN D'ENTRAÎNEMENT RECOMMANDÉ:")
    print("="*70)
    
    if recommendation == "train_more":
        print("✅ Phase 1: Foundation (10k exemples, 3 époques)")
        print("✅ Phase 2: Renforcement (20k exemples, 4 époques)")
        print("✅ Phase 3: DPO Alignment (10k exemples, 2 époques)")
        print("✅ Phase 4: DPO Fine-tuning (20k exemples, 1 époque)")
        phases = ["phase1", "phase2", "phase3", "phase4"]
    
    elif recommendation == "continue":
        print("✅ Phase 2: Renforcement (20k exemples, 4 époques)")
        print("✅ Phase 3: DPO Alignment (10k exemples, 2 époques)")
        print("✅ Phase 4: DPO Fine-tuning (20k exemples, 1 époque)")
        phases = ["phase2", "phase3", "phase4"]
    
    elif recommendation == "dpo":
        print("✅ Phase 3: DPO Alignment (10k exemples, 2 époques)")
        print("✅ Phase 4: DPO Fine-tuning (20k exemples, 1 époque)")
        phases = ["phase3", "phase4"]
    
    else:
        print("✅ Modèle déjà bien entraîné!")
        print("💡 Vous pouvez faire du fine-tuning additionnel si nécessaire")
        phases = []
    
    if phases:
        print(f"\n⏱️  Temps estimé: {len(phases) * 30}-{len(phases) * 60} minutes")
        print("="*70)
        
        response = input("\n🚀 Lancer l'entraînement progressif? (y/N): ")
        
        if response.lower() == 'y':
            for i, phase in enumerate(phases, 1):
                print(f"\n{'='*70}")
                print(f"🎯 ÉTAPE {i}/{len(phases)}")
                print('='*70)
                
                progressive_training(trainer, phase)
                
                # Test intermédiaire
                if i < len(phases):
                    print(f"\n📊 TEST APRÈS PHASE {i}:")
                    test_generation(trainer, ["Hello!", "What is AI?"])
            
            # Test final
            print("\n" + "="*70)
            print("🎉 ENTRAÎNEMENT PROGRESSIF TERMINÉ!")
            print("="*70)
            print("\n📊 TEST FINAL:")
            test_generation(trainer, test_prompts)
            
            print("\n💡 Testez en mode interactif avec:")
            print("   python test.py --mode interactive --template mistral")
        else:
            print("\n❌ Entraînement annulé")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()