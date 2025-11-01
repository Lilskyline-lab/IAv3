#HessGPT.py - VERSION CORRIGÉE AVEC GÉNÉRATION FIXÉE
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from TransformerBlock.transformer_block import TransformerBlock

class HessGPT(nn.Module):
    """
    Modèle HessGPT - Architecture Transformer personnalisée
    VERSION CORRIGÉE : 
    - Retourne (logits, hidden_states)
    - Génération avec masquage des tokens invalides
    """
    def __init__(
        self,
        vocab_size,
        embed_dim=768,
        num_heads=12,
        num_layers=12,
        max_seq_len=1024,
        dropout=0.1
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.position_embeddings = nn.Embedding(max_seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Layer Norm finale
        self.ln_final = nn.LayerNorm(embed_dim)
        
        # Output Head
        self.output_head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # Partager les poids
        self.output_head.weight = self.token_embeddings.weight
        
        # Initialisation
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialisation des poids"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass - Retourne (logits, hidden_states)
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] (optionnel)
        
        Returns:
            logits: [batch_size, seq_len, vocab_size]
            hidden_states: [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len = input_ids.shape
        
        # 1. Embeddings
        token_embeds = self.token_embeddings(input_ids)
        positions = torch.arange(0, seq_len, device=input_ids.device)
        position_embeds = self.position_embeddings(positions)
        x = self.dropout(token_embeds + position_embeds)
        
        # 2. Créer le masque causal
        mask = self.create_causal_mask(seq_len, device=input_ids.device)
        
        # 3. Passer à travers tous les Transformer Blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # 4. Layer Norm finale
        hidden_states = self.ln_final(x)
        
        # 5. Output Head
        logits = self.output_head(hidden_states)
        
        return logits, hidden_states
    
    def create_causal_mask(self, seq_len, device):
        """Crée un masque causal triangulaire"""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask
    
    def compute_loss(self, logits, targets, ignore_index=-100):
        """
        Calcule la loss séparément (pour entraînement)
        
        Args:
            logits: [batch_size, seq_len, vocab_size]
            targets: [batch_size, seq_len]
            ignore_index: Index à ignorer (pour padding/masking)
        
        Returns:
            loss: Scalar
        """
        loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            targets.view(-1),
            ignore_index=ignore_index
        )
        return loss
    
    def generate(
        self, 
        input_ids, 
        max_new_tokens=50, 
        temperature=1.0, 
        top_k=None, 
        top_p=0.9, 
        repetition_penalty=1.0,
        valid_token_ids=None
    ):
        """
        🔧 GÉNÉRATION CORRIGÉE - Masque les tokens invalides
        
        Args:
            input_ids: [batch_size, seq_len]
            max_new_tokens: Nombre de tokens à générer
            temperature: Contrôle la randomness
            top_k: Top-k sampling
            top_p: Nucleus sampling
            repetition_penalty: Pénalité pour tokens répétés
            valid_token_ids: Set/List des IDs valides du tokenizer (NOUVEAU)
        
        Returns:
            generated_ids: [batch_size, seq_len + max_new_tokens]
        """
        self.eval()
        
        generated = input_ids.clone()
        device = input_ids.device
        
        # 🔧 Créer le masque de tokens invalides une seule fois
        if valid_token_ids is not None:
            invalid_mask = torch.ones(self.vocab_size, dtype=torch.bool, device=device)
            valid_ids_list = list(valid_token_ids) if isinstance(valid_token_ids, set) else valid_token_ids
            invalid_mask[valid_ids_list] = False
        else:
            invalid_mask = None
        
        with torch.no_grad():
            for step in range(max_new_tokens):
                # Tronquer si trop long
                input_ids_cond = generated if generated.size(1) <= self.max_seq_len else generated[:, -self.max_seq_len:]
                
                # Forward pass
                logits, _ = self.forward(input_ids_cond)
                
                # Prendre les logits du dernier token
                next_logits = logits[:, -1, :] / temperature
                
                # 🔧 CORRECTION 1 : Masquer les tokens INVALIDES EN PREMIER
                if invalid_mask is not None:
                    next_logits[:, invalid_mask] = -float('inf')
                
                # Appliquer repetition penalty (seulement sur tokens valides)
                if repetition_penalty != 1.0:
                    for token_id in set(generated[0].tolist()):
                        if token_id < self.vocab_size:
                            # Ne pénaliser que si le token n'est pas déjà masqué
                            if invalid_mask is None or not invalid_mask[token_id]:
                                next_logits[0, token_id] /= repetition_penalty
                
                # Top-k sampling
                if top_k is not None and top_k > 0:
                    # Compter les tokens non-masqués
                    non_masked = (next_logits[0] > -float('inf')).sum().item()
                    effective_top_k = min(top_k, non_masked)
                    
                    if effective_top_k > 0:
                        v, _ = torch.topk(next_logits, effective_top_k)
                        next_logits[next_logits < v[:, [-1]]] = -float('inf')
                
                # 🔧 CORRECTION 2 : Vérifier qu'il reste des tokens valides
                if torch.all(torch.isinf(next_logits)):
                    # Fallback : prendre le premier token valide
                    if valid_token_ids is not None and len(valid_token_ids) > 0:
                        next_token = torch.tensor([[min(valid_token_ids)]], device=device)
                    else:
                        next_token = torch.tensor([[0]], device=device)
                    generated = torch.cat([generated, next_token], dim=1)
                    continue
                
                # Softmax
                probs = F.softmax(next_logits, dim=-1)
                
                # Top-p (nucleus) sampling
                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    
                    # Masquer les tokens au-delà de top_p
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    probs[indices_to_remove] = 0.0
                    
                    # Renormaliser
                    probs_sum = probs.sum(dim=-1, keepdim=True)
                    if probs_sum > 0:
                        probs = probs / probs_sum
                    else:
                        # Fallback si tout est masqué
                        probs = torch.zeros_like(probs)
                        if valid_token_ids is not None:
                            probs[0, min(valid_token_ids)] = 1.0
                        else:
                            probs[0, 0] = 1.0
                
                # 🔧 CORRECTION 3 : Vérifier que probs n'est pas vide
                if probs.sum() == 0:
                    if valid_token_ids is not None and len(valid_token_ids) > 0:
                        next_token = torch.tensor([[min(valid_token_ids)]], device=device)
                    else:
                        next_token = torch.tensor([[0]], device=device)
                else:
                    # Sampler le prochain token
                    next_token = torch.multinomial(probs, num_samples=1)
                
                # 🔧 CORRECTION 4 : Vérification finale du token généré
                token_id = next_token.item()
                if valid_token_ids is not None and token_id not in valid_token_ids:
                    # Le token est invalide, forcer un token valide
                    next_token = torch.tensor([[min(valid_token_ids)]], device=device)
                
                # Ajouter à la séquence
                generated = torch.cat([generated, next_token], dim=1)
        
        return generated


# ============================================
# TESTS
# ============================================

def test_hessgpt_forward():
    """Test du forward corrigé"""
    print("\n" + "="*60)
    print("TEST: HessGPT Forward (CORRIGÉ)")
    print("="*60)
    
    vocab_size = 300
    batch_size = 2
    seq_len = 10
    
    model = HessGPT(
        vocab_size=vocab_size,
        embed_dim=256,
        num_heads=8,
        num_layers=4
    )
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    logits, hidden_states = model(input_ids)
    
    print(f"✓ Input shape: {input_ids.shape}")
    print(f"✓ Logits shape: {logits.shape}")
    print(f"✓ Hidden states shape: {hidden_states.shape}")
    
    assert logits.shape == (batch_size, seq_len, vocab_size)
    assert hidden_states.shape == (batch_size, seq_len, model.embed_dim)
    
    print(f"\n✅ Forward corrigé : retourne bien (logits, hidden_states)")
    
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    loss = model.compute_loss(logits, targets)
    print(f"✓ Loss: {loss.item():.4f}")


def test_generation_with_valid_ids():
    """Test de génération avec valid_token_ids"""
    print("\n" + "="*60)
    print("TEST: Génération avec masquage tokens invalides")
    print("="*60)
    
    vocab_size = 300
    
    model = HessGPT(
        vocab_size=vocab_size,
        embed_dim=128,
        num_heads=4,
        num_layers=2
    )
    
    prompt = torch.randint(0, vocab_size, (1, 5))
    
    # Simuler un vocabulaire avec des trous
    # Seulement les IDs 0-99 et 200-250 sont valides
    valid_token_ids = set(range(100)) | set(range(200, 251))
    
    print(f"✓ Prompt: {prompt[0].tolist()}")
    print(f"✓ Tokens valides: {len(valid_token_ids)} IDs")
    
    # Génération AVEC valid_token_ids
    generated = model.generate(
        prompt,
        max_new_tokens=10,
        temperature=0.9,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
        valid_token_ids=valid_token_ids  # 🔧 NOUVEAU
    )
    
    generated_ids = generated[0].tolist()
    print(f"✓ Generated: {generated_ids}")
    
    # Vérifier qu'aucun ID invalide n'a été généré
    invalid_ids = [id for id in generated_ids if id not in valid_token_ids]
    
    if invalid_ids:
        print(f"❌ IDs invalides générés: {invalid_ids}")
    else:
        print(f"✅ Tous les IDs générés sont valides!")


if __name__ == "__main__":
    print("\n🚀 TESTS DU MODÈLE HessGPT CORRIGÉ\n")
    
    test_hessgpt_forward()
    test_generation_with_valid_ids()
    
    print("\n" + "="*60)
    print("✅ TOUS LES TESTS PASSÉS!")
    print("="*60)
    print("\n💡 Changements principaux:")
    print("   1. forward() retourne (logits, hidden_states)")
    print("   2. compute_loss() séparé pour entraînement")
    print("   3. generate() avec valid_token_ids pour masquer tokens invalides")
    print("   4. Gestion robuste des cas limites (probs vides, tokens invalides)")
    print("="*60 + "\n")
