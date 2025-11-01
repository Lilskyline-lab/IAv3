"""
Script de test interactif pour le modèle HessGPT
Supporte plusieurs templates de prompt et modes d'interaction
"""

import os
import sys
import json
import torch
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model.HessGPT import HessGPT

try:
    from transformers import AutoTokenizer
    HF_AVAILABLE = True
except ImportError:
    print("⚠️ transformers requis: pip install transformers")
    sys.exit(1)


class TokenizerWrapper:
    """Wrapper pour gérer les erreurs de décodage"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # Exposer les attributs importants
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.bos_token_id = getattr(tokenizer, 'bos_token_id', None)
        self.vocab_size = tokenizer.vocab_size
    
    def encode(self, *args, **kwargs):
        return self.tokenizer.encode(*args, **kwargs)
    
    def decode(self, *args, **kwargs):
        try:
            result = self.tokenizer.decode(*args, **kwargs)
            # Nettoyer les caractères problématiques
            return result.encode('utf-8', errors='ignore').decode('utf-8')
        except Exception as e:
            print(f"⚠️ Erreur de décodage: {e}")
            return "[ERREUR_DÉCODAGE]"
    
    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)
    
    def __getattr__(self, name):
        # Déléguer tous les autres attributs au tokenizer original
        return getattr(self.tokenizer, name)


class ChatBot:
    """Bot de chat avec support de différents templates"""
    
    TEMPLATES = {
        'mistral': {
            'user': '[INST] {prompt} [/INST]',
            'assistant': '{response}',
            'system': None
        },
        'llama': {
            'user': '<s>[INST] {prompt} [/INST]',
            'assistant': '{response}</s>',
            'system': '<<SYS>>{system}<</SYS>>'
        },
        'chatml': {
            'user': '<|im_start|>user\n{prompt}<|im_end|>\n',
            'assistant': '<|im_start|>assistant\n{response}<|im_end|>\n',
            'system': '<|im_start|>system\n{system}<|im_end|>\n'
        },
        'simple': {
            'user': 'User: {prompt}\n',
            'assistant': 'Assistant: {response}\n',
            'system': 'System: {system}\n'
        }
    }
    
    def __init__(self, model_dir: str, template: str = 'mistral'):
        self.model_dir = model_dir
        self.template_name = template
        self.template = self.TEMPLATES.get(template, self.TEMPLATES['mistral'])
        
        print("\n" + "="*70)
        print("🤖 CHARGEMENT DU MODÈLE")
        print("="*70)
        
        # Charger config
        self.config = self._load_config()
        print("📋 Configuration chargée:")
        print(f"   Vocab size: {self.config['vocab_size']:,}")
        print(f"   Embed dim: {self.config['embed_dim']}")
        print(f"   Num layers: {self.config['num_layers']}")
        print(f"   Tokenizer: {self.config.get('tokenizer_name', 'gpt2')}")
        
        # Charger tokenizer
        self.tokenizer = self._load_tokenizer()
        
        # Charger modèle
        self.model = self._load_model()
        print("✅ Modèle chargé avec succès!\n")
        
        # Historique de conversation
        self.history = []
    
    def _load_config(self) -> dict:
        """Charge la configuration du modèle"""
        config_path = os.path.join(self.model_dir, "config.json")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config non trouvée: {config_path}")
        
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _load_tokenizer(self):
        """Charge le tokenizer avec wrapper pour la gestion d'erreurs"""
        tokenizer_name = self.config.get('tokenizer_name', 'gpt2')
        print(f"🔤 Chargement tokenizer: {tokenizer_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Retourner le wrapper au lieu du tokenizer brut
        return TokenizerWrapper(tokenizer)
    
    def _load_model(self) -> HessGPT:
        """Charge le modèle"""
        model_path = os.path.join(self.model_dir, "model.pt")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modèle non trouvé: {model_path}")
        
        print(f"📦 Chargement du modèle: {model_path}")
        
        # Créer le modèle
        model = HessGPT(
            vocab_size=self.config['vocab_size'],
            embed_dim=self.config['embed_dim'],
            num_heads=self.config['num_heads'],
            num_layers=self.config['num_layers'],
            max_seq_len=self.config['max_seq_len']
        )
        
        # Charger les poids
        try:
            state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        except TypeError:
            state_dict = torch.load(model_path, map_location='cpu')
        
        model.load_state_dict(state_dict)
        model.eval()
        
        return model
    
    def format_prompt(self, user_message: str, system_message: str = None) -> str:
        """Formate le prompt selon le template"""
        prompt = ""
        
        # Ajouter le message système si présent
        if system_message and self.template['system']:
            prompt += self.template['system'].format(system=system_message)
        
        # Ajouter l'historique
        for user_msg, assistant_msg in self.history:
            prompt += self.template['user'].format(prompt=user_msg)
            prompt += self.template['assistant'].format(response=assistant_msg)
        
        # Ajouter le message actuel
        prompt += self.template['user'].format(prompt=user_message)
        
        return prompt
    
    def generate(
        self,
        prompt: str,
        max_length: int = 200,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        stop_sequences: list = None
    ) -> str:
        """Génère une réponse"""
        
        # Encoder le prompt
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_tensor = torch.tensor([input_ids], dtype=torch.long)
        
        # Tokens générés
        generated = []
        
        # Pour le repetition penalty
        seen_tokens = set(input_ids)
        
        with torch.no_grad():
            current = input_tensor
            
            for step in range(max_length):
                # Forward pass
                logits, _ = self.model(current)
                next_token_logits = logits[0, -1, :]
                
                # Appliquer repetition penalty
                if repetition_penalty != 1.0:
                    for token_id in seen_tokens:
                        if next_token_logits[token_id] < 0:
                            next_token_logits[token_id] *= repetition_penalty
                        else:
                            next_token_logits[token_id] /= repetition_penalty
                
                # Temperature
                next_token_logits = next_token_logits / temperature
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Vérifier fin de séquence
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                generated.append(next_token.item())
                seen_tokens.add(next_token.item())
                
                # Mettre à jour le contexte
                current = torch.cat([current, next_token.unsqueeze(0)], dim=1)
                
                # Vérifier les stop sequences
                if stop_sequences:
                    current_text = self.tokenizer.decode(generated, skip_special_tokens=True)
                    if any(stop_seq in current_text for stop_seq in stop_sequences):
                        break
        
        # Décoder
        if generated:
            response = self.tokenizer.decode(generated, skip_special_tokens=True)
        else:
            response = ""
        
        return response.strip()
    
    def chat(
        self,
        message: str,
        system_message: str = None,
        max_length: int = 200,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Interface de chat avec historique"""
        
        # Formater le prompt complet
        full_prompt = self.format_prompt(message, system_message)
        
        # Générer la réponse
        response = self.generate(
            full_prompt,
            max_length=max_length,
            temperature=temperature,
            **kwargs
        )
        
        # Ajouter à l'historique
        self.history.append((message, response))
        
        return response
    
    def clear_history(self):
        """Efface l'historique de conversation"""
        self.history = []
        print("🗑️  Historique effacé")
    
    def show_history(self):
        """Affiche l'historique"""
        if not self.history:
            print("📭 Aucun historique")
            return
        
        print("\n" + "="*70)
        print("📜 HISTORIQUE DE CONVERSATION")
        print("="*70)
        
        for i, (user_msg, assistant_msg) in enumerate(self.history, 1):
            print(f"\n[{i}] 👤 User: {user_msg}")
            print(f"    🤖 Assistant: {assistant_msg}")
        
        print("\n" + "="*70)


def interactive_mode(bot: ChatBot):
    """Mode interactif de chat"""
    print("\n" + "="*70)
    print("💬 MODE INTERACTIF")
    print("="*70)
    print("Commandes disponibles:")
    print("  /help     - Afficher l'aide")
    print("  /clear    - Effacer l'historique")
    print("  /history  - Voir l'historique")
    print("  /settings - Modifier les paramètres")
    print("  /quit     - Quitter")
    print("="*70 + "\n")
    
    # Paramètres par défaut
    settings = {
        'temperature': 0.7,
        'max_length': 200,
        'top_k': 50,
        'top_p': 0.9,
        'repetition_penalty': 1.2
    }
    
    while True:
        try:
            user_input = input("👤 Vous: ").strip()
            
            if not user_input:
                continue
            
            # Commandes
            if user_input.startswith('/'):
                cmd = user_input.lower()
                
                if cmd == '/quit':
                    print("\n👋 Au revoir!")
                    break
                
                elif cmd == '/clear':
                    bot.clear_history()
                    continue
                
                elif cmd == '/history':
                    bot.show_history()
                    continue
                
                elif cmd == '/help':
                    print("\n📖 Aide:")
                    print("  Tapez simplement votre message pour discuter")
                    print("  Utilisez les commandes / pour des actions spéciales")
                    continue
                
                elif cmd == '/settings':
                    print("\n⚙️  Paramètres actuels:")
                    for key, value in settings.items():
                        print(f"  {key}: {value}")
                    
                    print("\nModifier (laissez vide pour garder) :")
                    for key in settings.keys():
                        new_val = input(f"  {key} [{settings[key]}]: ").strip()
                        if new_val:
                            try:
                                settings[key] = float(new_val)
                            except ValueError:
                                print(f"  ⚠️ Valeur invalide pour {key}")
                    continue
                
                else:
                    print("❌ Commande inconnue. Tapez /help pour l'aide")
                    continue
            
            # Générer réponse
            print("🤖 Assistant: ", end='', flush=True)
            
            response = bot.chat(
                user_input,
                temperature=settings['temperature'],
                max_length=settings['max_length'],
                top_k=settings['top_k'],
                top_p=settings['top_p'],
                repetition_penalty=settings['repetition_penalty']
            )
            
            print(response)
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 Au revoir!")
            break
        except Exception as e:
            print(f"\n❌ Erreur: {e}")
            import traceback
            traceback.print_exc()


def benchmark_mode(bot: ChatBot):
    """Mode benchmark avec questions prédéfinies"""
    print("\n" + "="*70)
    print("📊 MODE BENCHMARK")
    print("="*70)
    
    test_prompts = [
        "Hello! How are you?",
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a short poem about AI.",
        "What's 2+2?",
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[{i}/{len(test_prompts)}] 👤 Prompt: {prompt}")
        print("🤖 Réponse: ", end='', flush=True)
        
        response = bot.chat(prompt, max_length=150)
        print(response)
        print("-" * 70)
    
    print("\n✅ Benchmark terminé!")


def single_prompt_mode(bot: ChatBot, prompt: str):
    """Mode prompt unique"""
    print("\n" + "="*70)
    print("💬 GÉNÉRATION UNIQUE")
    print("="*70)
    print(f"👤 Prompt: {prompt}\n")
    
    response = bot.chat(prompt)
    
    print(f"🤖 Réponse: {response}\n")
    print("="*70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test du modèle HessGPT")
    parser.add_argument(
        "--model_dir",
        default="saved_models/my_llm",
        help="Dossier contenant le modèle"
    )
    parser.add_argument(
        "--template",
        choices=['mistral', 'llama', 'chatml', 'simple'],
        default='mistral',
        help="Template de prompt à utiliser"
    )
    parser.add_argument(
        "--mode",
        choices=['interactive', 'benchmark', 'single'],
        default='interactive',
        help="Mode d'exécution"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Prompt pour le mode single"
    )
    
    args = parser.parse_args()
    
    # Chemin absolu
    if not os.path.isabs(args.model_dir):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args.model_dir = os.path.join(base_dir, args.model_dir)
    
    # Vérifier que le dossier existe
    if not os.path.exists(args.model_dir):
        print(f"❌ Dossier non trouvé: {args.model_dir}")
        return
    
    # Créer le bot
    try:
        bot = ChatBot(args.model_dir, template=args.template)
    except Exception as e:
        print(f"\n❌ Erreur lors du chargement: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Lancer le mode approprié
    if args.mode == 'interactive':
        interactive_mode(bot)
    elif args.mode == 'benchmark':
        benchmark_mode(bot)
    elif args.mode == 'single':
        if not args.prompt:
            print("❌ --prompt requis pour le mode single")
            return
        single_prompt_mode(bot, args.prompt)


if __name__ == "__main__":
    main()