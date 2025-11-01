"""
Wrapper pour utiliser les tokenizers Hugging Face avec l'interface MYBPE
Compatible avec GPT-2, Llama, Mistral, etc.
"""
from transformers import AutoTokenizer
from huggingface_hub.errors import GatedRepoError
import pickle
import os

class HFTokenizerWrapper:
    """Wrapper qui imite l'interface MYBPE pour les tokenizers HuggingFace"""
    
    def __init__(self, model_name="gpt2", vocab_size=None):
        """
        Args:
            model_name: Nom du modèle HuggingFace (gpt2, meta-llama/Llama-2-7b-hf, mistralai/Mistral-7B-v0.1)
            vocab_size: Ignoré, juste pour compatibilité avec MYBPE
        """
        self.model_name = model_name
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except (GatedRepoError, OSError) as e:
            if "gated" in str(e).lower() or "restricted" in str(e).lower():
                print(f"\n❌ ERREUR: Le modèle '{model_name}' est à accès restreint!")
                print("\n📋 Solutions possibles:")
                print("   1. Authentification Hugging Face:")
                print("      - Créez un token sur https://huggingface.co/settings/tokens")
                print("      - Exécutez: huggingface-cli login")
                print(f"      - Acceptez les conditions: https://huggingface.co/{model_name}")
                print("\n   2. Utilisez un modèle non-gated:")
                print("      - meta-llama/Llama-3.2-1B (32k vocab)")
                print("      - mistralai/Mistral-7B-v0.1 (32k vocab)")
                print("      - gpt2 (50k vocab)")
                raise
            else:
                raise
        
        # Assurer qu'il y a un pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.vocab_size = len(self.tokenizer)
        
        # Créer un mapping vocabulaire compatible
        self.voc = {i: self.tokenizer.convert_ids_to_tokens(i) for i in range(self.vocab_size)}
        
        print(f"✅ Tokenizer HuggingFace chargé: {model_name}")
        print(f"📊 Vocabulaire: {self.vocab_size:,} tokens")
    
    @property
    def vocab(self):
        """Pour compatibilité avec le code existant"""
        return self.voc
    
    def encoder(self, text):
        """
        Encode du texte en IDs (compatible avec MYBPE.encoder)
        
        Args:
            text: String à encoder
        
        Returns:
            List[int]: Liste d'IDs
        """
        if isinstance(text, str):
            return self.tokenizer.encode(text, add_special_tokens=False)
        return text
    
    def decoder(self, ids):
        """
        Décode des IDs en texte (compatible avec MYBPE.decoder)
        
        Args:
            ids: List[int] ou Tensor
        
        Returns:
            str: Texte décodé
        """
        if hasattr(ids, 'tolist'):  # Si c'est un tensor
            ids = ids.tolist()
        return self.tokenizer.decode(ids, skip_special_tokens=True)
    
    def load_tokenizer(self, path, verbose=True):
        """Pour compatibilité - Le tokenizer est déjà chargé"""
        if verbose:
            print(f"ℹ️  Tokenizer HuggingFace déjà chargé ({self.model_name})")
            print(f"📊 Vocab size: {self.vocab_size:,}")
    
    def save_tokenizer(self, path):
        """Sauvegarde les infos du tokenizer"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        data = {
            'model_name': self.model_name,
            'vocab_size': self.vocab_size,
            'tokenizer_type': 'huggingface'
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        # Sauvegarder aussi le tokenizer HF
        tokenizer_dir = path.replace('.bin', '_hf')
        self.tokenizer.save_pretrained(tokenizer_dir)
        print(f"💾 Tokenizer sauvegardé: {path}")
        print(f"💾 HuggingFace files: {tokenizer_dir}/")
    
    @staticmethod
    def load_from_hf(path):
        """Charge un tokenizer HF depuis un fichier .bin"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        model_name = data.get('model_name', 'gpt2')
        return HFTokenizerWrapper(model_name)


# Modèles recommandés (non-gated)
RECOMMENDED_MODELS = {
    'gpt2': {'vocab': '50k', 'description': 'GPT-2 OpenAI (classique)'},
    'gpt2-large': {'vocab': '50k', 'description': 'GPT-2 Large'},
    'mistralai/Mistral-7B-v0.1': {'vocab': '32k', 'description': 'Mistral 7B'},
    'meta-llama/Llama-3.2-1B': {'vocab': '128k', 'description': 'Llama 3.2 (récent)'},
}


def download_and_save_tokenizer(model_name="gpt2", save_path="tokenizer_50k_hf.bin"):
    """
    Télécharge un tokenizer HuggingFace et le sauvegarde
    
    Exemples:
        - "gpt2" (50k vocab)
        - "meta-llama/Llama-3.2-1B" (128k vocab)
        - "mistralai/Mistral-7B-v0.1" (32k vocab)
    """
    print(f"\n📥 Téléchargement du tokenizer: {model_name}")
    
    tokenizer = HFTokenizerWrapper(model_name)
    tokenizer.save_tokenizer(save_path)
    
    print(f"\n✅ Tokenizer prêt à l'emploi!")
    print(f"💡 Utilisez-le dans votre code:")
    print(f"   from hf_tokenizer_wrapper import HFTokenizerWrapper")
    print(f"   tokenizer = HFTokenizerWrapper.load_from_hf('{save_path}')")
    
    return tokenizer


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Télécharger un tokenizer HuggingFace")
    parser.add_argument("--model", default="gpt2", help="Nom du modèle HF")
    parser.add_argument("--save", default="tokenizer_50k_hf.bin", help="Chemin de sauvegarde")
    parser.add_argument("--test", action="store_true", help="Tester l'encodage/décodage")
    parser.add_argument("--list", action="store_true", help="Lister les modèles recommandés")
    
    args = parser.parse_args()
    
    if args.list:
        print("\n" + "="*60)
        print("📋 MODÈLES RECOMMANDÉS (NON-GATED)")
        print("="*60)
        for model, info in RECOMMENDED_MODELS.items():
            print(f"\n🔹 {model}")
            print(f"   Vocab: {info['vocab']} | {info['description']}")
        print("\n" + "="*60)
        exit(0)
    
    tokenizer = download_and_save_tokenizer(args.model, args.save)
    
    if args.test:
        print("\n" + "="*60)
        print("🧪 TEST DU TOKENIZER")
        print("="*60)
        
        test_texts = [
            "Hello, how are you?",
            "Bonjour, comment allez-vous ?",
            "Human: What is AI?\nBot: AI is artificial intelligence."
        ]
        
        for text in test_texts:
            print(f"\n📝 Texte: {repr(text)}")
            ids = tokenizer.encoder(text)
            decoded = tokenizer.decoder(ids)
            print(f"   🔢 IDs: {ids[:10]}... ({len(ids)} tokens)")
            print(f"   ✅ Décodé: {repr(decoded)}")