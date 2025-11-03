"""
Script pour charger un modèle pré-entraîné léger et tester la génération
RAPIDE, PAS D'ENTRAÎNEMENT, JUSTE TEST

Modèles supportés (< 1B params):
- TinyLlama-1.1B (~550MB)
- Qwen2-0.5B (~250MB)
- Phi-1.5 (~1.3B)
- OPT-350M (~350MB)

Usage:
    python load_pretrained_test.py --model tinyllama
    python load_pretrained_test.py --model qwen
    python load_pretrained_test.py --model opt
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from typing import Optional, Set


class PretrainedWrapper:
    """Wrapper pour utiliser des modèles Hugging Face pré-entraînés"""
    
    MODELS = {
        "tinyllama": {
            "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "size": "1.1B params (~550MB)",
            "description": "TinyLlama - Petit modèle Llama pré-entraîné"
        },
        "qwen": {
            "name": "Qwen/Qwen2-0.5B-Instruct",
            "size": "0.5B params (~250MB)",
            "description": "Qwen2 0.5B - Très léger et rapide"
        },
        "opt": {
            "name": "facebook/opt-350m",
            "size": "350M params (~350MB)",
            "description": "OPT-350M - Facebook's OPT"
        },
        "pythia": {
            "name": "EleutherAI/pythia-410m",
            "size": "410M params (~410MB)",
            "description": "Pythia 410M - EleutherAI"
        }
    }
    
    def __init__(self, model_key: str = "qwen", device: str = "cpu"):
        """
        Args:
            model_key: Clé du modèle à charger (voir MODELS)
            device: 'cpu' ou 'cuda'
        """
        self.device = device
        self.model_key = model_key
        
        if model_key not in self.MODELS:
            raise ValueError(f"Modèle inconnu: {model_key}. Disponibles: {list(self.MODELS.keys())}")
        
        self.model_name = self.MODELS[model_key]["name"]
        
        print(f"\n{'='*70}")
        print(f"📦 CHARGEMENT DU MODÈLE PRÉ-ENTRAÎNÉ")
        print(f"{'='*70}")
        print(f"Modèle: {self.MODELS[model_key]['description']}")
        print(f"Taille: {self.MODELS[model_key]['size']}")
        print(f"Device: {device}")
        print(f"{'='*70}\n")
        
        # Charger tokenizer
        print("🔤 Chargement du tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Configuration du pad_token pour éviter l'erreur d'attention mask
        # Utiliser un token différent de eos_token
        if self.tokenizer.pad_token is None:
            # Essayer d'utiliser unk_token
            if self.tokenizer.unk_token is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
                print(f"   Pad token défini comme unk_token: {self.tokenizer.pad_token}")
            else:
                # Ajouter un nouveau token spécial
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                print(f"   Nouveau pad token ajouté: [PAD]")
        
        # S'assurer que pad_token != eos_token
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            print(f"   ⚠️  pad_token == eos_token détecté, correction...")
            if self.tokenizer.unk_token is not None and self.tokenizer.unk_token_id != self.tokenizer.eos_token_id:
                self.tokenizer.pad_token = self.tokenizer.unk_token
                print(f"   Pad token redéfini comme unk_token")
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                print(f"   Nouveau pad token ajouté: [PAD]")
        
        print(f"✅ Tokenizer chargé: {len(self.tokenizer):,} tokens")
        print(f"   - EOS token: {self.tokenizer.eos_token} (ID: {self.tokenizer.eos_token_id})")
        print(f"   - PAD token: {self.tokenizer.pad_token} (ID: {self.tokenizer.pad_token_id})")
        print(f"   - BOS token: {self.tokenizer.bos_token} (ID: {self.tokenizer.bos_token_id})")
        
        # Charger modèle
        print("🤖 Chargement du modèle...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,  # Float32 pour CPU
            low_cpu_mem_usage=True
        )
        
        self.model.to(device)
        self.model.eval()
        
        print(f"✅ Modèle chargé avec succès!")
        
        # Compter les paramètres
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"📊 Paramètres totaux: {total_params:,} ({total_params/1e6:.1f}M)")
        
        # Extraire valid_token_ids
        self.valid_token_ids = self._extract_valid_token_ids()
    
    def _extract_valid_token_ids(self) -> Set[int]:
        """Extrait les IDs valides du tokenizer"""
        if hasattr(self.tokenizer, 'get_vocab'):
            vocab = self.tokenizer.get_vocab()
            valid_ids = set(vocab.values())
        else:
            valid_ids = set(range(len(self.tokenizer)))
        
        # Ajouter tokens spéciaux
        special_ids = {
            self.tokenizer.bos_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
            self.tokenizer.unk_token_id
        }
        valid_ids.update(id for id in special_ids if id is not None)
        
        return valid_ids
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.2,
        do_sample: bool = True
    ) -> str:
        """
        Génère du texte avec le modèle pré-entraîné
        
        Args:
            prompt: Texte de départ
            max_new_tokens: Nombre de tokens à générer
            temperature: Contrôle la créativité (0.1-2.0)
            top_p: Nucleus sampling
            top_k: Top-k sampling
            repetition_penalty: Pénalité pour répétitions
            do_sample: Si False, utilise greedy decoding
        
        Returns:
            Texte généré
        """
        # Tokenize avec attention_mask explicite
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_attention_mask=True  # Créer explicitement le mask
        ).to(self.device)
        
        print(f"\n🔍 Prompt: {prompt}")
        print(f"📏 Input tokens: {inputs['input_ids'].shape[1]}")
        
        # Générer avec attention_mask
        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],  # Passer le mask explicitement
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                # Protections anti-répétitions
                no_repeat_ngram_size=3,  # Empêche répétition de 3-grams
            )
        
        # Decoder
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extraire seulement la partie générée
        response = generated_text[len(prompt):].strip()
        
        print(f"📝 Réponse: {response}")
        print(f"📏 Output tokens: {outputs.shape[1]}")
        
        return response
    
    def interactive_chat(self):
        """Mode chat interactif"""
        print(f"\n{'='*70}")
        print(f"💬 MODE CHAT INTERACTIF")
        print(f"{'='*70}")
        print("⚠️  NOTE: Ce modèle n'est PAS entraîné pour la conversation!")
        print("Il génère du texte aléatoire basé sur ses données d'entraînement.")
        print("Pour un vrai chatbot, il faut fine-tuner le modèle.")
        print(f"{'='*70}")
        print("Commandes:")
        print("  /temp <val>  - Changer température")
        print("  /penalty <val> - Changer repetition penalty")
        print("  /quit        - Quitter")
        print(f"{'='*70}\n")
        
        temperature = 0.7
        repetition_penalty = 1.2
        
        while True:
            try:
                user_input = input("👤 Vous: ").strip()
                
                if not user_input:
                    continue
                
                if user_input == "/quit":
                    print("👋 Au revoir!")
                    break
                
                if user_input.startswith("/temp"):
                    try:
                        temperature = float(user_input.split()[1])
                        print(f"✅ Température: {temperature}")
                    except:
                        print("❌ Usage: /temp 0.7")
                    continue
                
                if user_input.startswith("/penalty"):
                    try:
                        repetition_penalty = float(user_input.split()[1])
                        print(f"✅ Repetition penalty: {repetition_penalty}")
                    except:
                        print("❌ Usage: /penalty 1.5")
                    continue
                
                print("🤖 Assistant: ", end="", flush=True)
                
                response = self.generate(
                    user_input,
                    max_new_tokens=100,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty
                )
                
                print(response)
                
            except KeyboardInterrupt:
                print("\n👋 Au revoir!")
                break
            except Exception as e:
                print(f"\n❌ Erreur: {e}")
    
    def batch_test(self):
        """Test avec plusieurs prompts"""
        print(f"\n{'='*70}")
        print(f"🧪 TEST BATCH")
        print(f"{'='*70}\n")
        
        test_prompts = [
            "Hello, how are you?",
            "What is artificial intelligence?",
            "Explain Python programming in simple terms.",
            "Tell me a short joke.",
            "What is 2+2?",
            "Write a haiku about coding.",
            "Describe a cat in 3 words.",
            "What is the capital of France?"
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n[{i}/{len(test_prompts)}] {'='*50}")
            
            response = self.generate(
                prompt,
                max_new_tokens=50,
                temperature=0.7,
                repetition_penalty=1.2
            )
            
            # Vérifier répétitions
            words = response.split()
            has_repetition = False
            if len(words) >= 6:
                for j in range(len(words) - 5):
                    if words[j] == words[j+1] == words[j+2]:
                        has_repetition = True
                        break
            
            if has_repetition:
                print("⚠️  Répétition détectée!")
            else:
                print("✅ Pas de répétition")
        
        print(f"\n{'='*70}")
        print("✅ Test batch terminé!")
        print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Test de génération avec modèles pré-entraînés")
    parser.add_argument(
        "--model",
        type=str,
        default="qwen",
        choices=["tinyllama", "qwen", "opt", "pythia"],
        help="Modèle à charger"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="interactive",
        choices=["interactive", "batch"],
        help="Mode: interactive ou batch test"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device (cpu ou cuda)"
    )
    
    args = parser.parse_args()
    
    # Vérifier CUDA
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA non disponible, utilisation du CPU")
        args.device = "cpu"
    
    try:
        # Charger modèle
        wrapper = PretrainedWrapper(args.model, args.device)
        
        # Lancer le mode choisi
        if args.mode == "interactive":
            wrapper.interactive_chat()
        else:
            wrapper.batch_test()
    
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()