#app.py

import os
import json
import torch
import io
import sys
import threading
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

sys.path.append('./IA/training/Model')
sys.path.append('./IA/training/TransformerBlock')
sys.path.append('./IA/training/Attention')
sys.path.append('./IA/training/FeedForward')
sys.path.append('./IA/training/Embeddings_Layer')
sys.path.append('./IA/Tokenizer')

from HessGPT import HessGPT
from tokenizerv2 import MYBPE

app = Flask(__name__)
CORS(app)

model = None
tokenizer = None
config = None
device = None
model_loaded = False
load_lock = threading.Lock()

def silence_output():
    old_out = sys.stdout
    old_err = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    return old_out, old_err

def restore_output(old):
    out, err = old
    sys.stdout = out
    sys.stderr = err

def load_model_and_tokenizer():
    global model, tokenizer, config, device, model_loaded

    if model_loaded:
        return

    with load_lock:
        if model_loaded:
            return

        model_dir = "./IA/saved_models/my_llm"
        tokenizer_path = "./IA/Tokenizer/tokenizer_20k_production.bin"
        device = torch.device("cpu")

        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        print("🔤 Chargement du tokenizer...")
        tokenizer = MYBPE(vocab_size=config.get("vocab_size", 20000))
        old = silence_output()
        try:
            tokenizer.load_tokenizer(tokenizer_path, verbose=False)
        finally:
            restore_output(old)
        print("✅ Tokenizer chargé")

        print("🤖 Initialisation du modèle...")
        model = HessGPT(
            vocab_size=config["vocab_size"],
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            max_seq_len=config["max_seq_len"]
        )

        model_file = os.path.join(model_dir, "model.pt")
        if os.path.exists(model_file):
            try:
                state = torch.load(model_file, map_location=device, weights_only=True)
            except TypeError:
                state = torch.load(model_file, map_location=device)
            
            # Vérifier la compatibilité
            model_params = sum(p.numel() for p in model.parameters())
            loaded_params = sum(p.numel() for p in state.values() if isinstance(p, torch.Tensor))
            
            print(f"✅ Modèle chargé depuis {model_file}")
            print(f"   📊 Paramètres modèle: {model_params:,}")
            print(f"   📊 Paramètres chargés: {loaded_params:,}")
            
            model.load_state_dict(state)
            
            # Vérifier si les poids sont initialisés (pas tous à zéro)
            first_param = next(iter(model.parameters()))
            print(f"   🔍 Premier poids (sample): {first_param.flatten()[:5]}")
        else:
            print(f"⚠️ Fichier model.pt non trouvé. Le modèle utilisera des poids aléatoires.")

        model.to(device)
        model.eval()
        model_loaded = True
        print("✅ Modèle prêt!")

def generate_response(prompt, max_new_tokens=40, temperature=0.9, top_k=0, top_p=0.9, repetition_penalty=1.3):
    old = silence_output()
    try:
        tokens = tokenizer.encoder(prompt)
    finally:
        restore_output(old)

    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    generated_ids = input_ids[0].tolist()
    
    print(f"\n🎬 Génération démarrée:")
    print(f"   📝 Prompt tokens: {len(tokens)}")
    print(f"   🎯 Max new tokens: {max_new_tokens}")

    with torch.no_grad():
        for step in range(max_new_tokens):
            # Préparer l'input (garder seulement les derniers max_seq_len tokens)
            context_tokens = generated_ids[-model.max_seq_len:]
            inp = torch.tensor([context_tokens], dtype=torch.long, device=device)
            
            # Forward pass - CORRECTION: le modèle retourne (logits, loss)
            # Pendant l'inférence, loss sera None
            output = model(inp)
            
            # Gérer le cas où model() retourne un tuple ou juste logits
            if isinstance(output, tuple):
                logits = output[0]  # Premier élément est logits
            else:
                logits = output
            
            # Vérifier la forme des logits
            if logits.dim() != 3:
                print(f"⚠️ Forme de logits invalide: {logits.shape}")
                break
            
            # Extraire les logits pour le dernier token
            next_logits = logits[0, -1, :].clone()
            
            # Debug: afficher les statistiques des logits
            if step == 0:
                print(f"   📊 Logits stats: min={next_logits.min():.2f}, max={next_logits.max():.2f}, mean={next_logits.mean():.2f}")
                print(f"   🔍 Logits contient NaN: {torch.isnan(next_logits).any()}")
                print(f"   🔍 Logits contient Inf: {torch.isinf(next_logits).any()}")
            
            # Vérifier les NaN/Inf
            if torch.isnan(next_logits).any() or torch.isinf(next_logits).any():
                print(f"⚠️ Logits invalides détectés au step {step}")
                break
            
            # Appliquer la pénalité de répétition
            if repetition_penalty != 1.0:
                for token_id in set(generated_ids):
                    if 0 <= token_id < len(next_logits):
                        if next_logits[token_id] > 0:
                            next_logits[token_id] /= repetition_penalty
                        else:
                            next_logits[token_id] *= repetition_penalty
            
            # Appliquer la température
            if temperature > 0 and temperature != 1.0:
                next_logits = next_logits / temperature
            
            # Calculer les probabilités
            probs = torch.softmax(next_logits, dim=-1)
            
            # Vérifier que les probs sont valides
            if torch.isnan(probs).any() or (probs.sum() == 0):
                print(f"⚠️ Probabilités invalides au step {step}")
                break
            
            # Sampling avec top-k ou top-p
            if top_k is not None and top_k > 0:
                # Top-k sampling
                k = min(top_k, probs.size(-1))
                top_probs, top_indices = torch.topk(probs, k)
                top_probs = top_probs / top_probs.sum()  # Renormaliser
                sampled_idx = torch.multinomial(top_probs, num_samples=1)
                next_id = int(top_indices[sampled_idx].item())
            elif top_p is not None and 0.0 < top_p < 1.0:
                # Top-p (nucleus) sampling
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # Trouver le cutoff
                mask = cumsum_probs <= top_p
                if not mask.any():
                    mask[0] = True  # Garder au moins le top token
                
                # Sélectionner les tokens
                filtered_probs = sorted_probs * mask.float()
                filtered_probs = filtered_probs / filtered_probs.sum()
                
                sampled_idx = torch.multinomial(filtered_probs, num_samples=1)
                next_id = int(sorted_indices[sampled_idx].item())
            else:
                # Sampling sans filtrage
                next_id = int(torch.multinomial(probs, num_samples=1).item())
            
            # Debug: afficher le token sélectionné
            if step < 3:  # Afficher les 3 premiers tokens
                old = silence_output()
                try:
                    token_text = tokenizer.decoder([next_id])
                finally:
                    restore_output(old)
                print(f"   🎲 Step {step}: token_id={next_id}, text='{token_text}', prob={probs[next_id]:.4f}")
            
            # Ajouter le token généré
            generated_ids.append(next_id)
            
            # Vérifier les conditions d'arrêt (token de fin, etc.)
            # Vous pouvez ajouter une vérification pour un token EOS si vous en avez un
            # if next_id == tokenizer.eos_token_id:
            #     break

    # Décoder le texte généré
    old = silence_output()
    try:
        text = tokenizer.decoder(generated_ids)
    finally:
        restore_output(old)
    
    print(f"✅ Génération terminée: {len(generated_ids) - len(tokens)} nouveaux tokens")

    # Nettoyer la réponse
    if "Bot:" in text:
        response = text.split("Bot:")[-1].strip()
    elif prompt in text:
        response = text[len(prompt):].strip()
    else:
        response = text.strip()
    
    # Nettoyer les réponses vides ou invalides
    if not response or len(response) < 2:
        response = "[Le modèle n'a pas généré de réponse valide]"
    
    return response

@app.route('/')
def index():
    return render_template('index.html')

def ensure_model_loaded():
    if not model_loaded:
        load_model_and_tokenizer()

@app.route('/api/chat', methods=['POST'])
def chat():
    ensure_model_loaded()

    try:
        data = request.json
        user_message = data.get('message', '')
        max_tokens = data.get('max_tokens', 40)
        temperature = data.get('temperature', 0.7)
        top_k = data.get('top_k', 50)
        top_p = data.get('top_p', 0.95)
        repetition_penalty = data.get('repetition_penalty', 1.1)

        if not user_message:
            return jsonify({'error': 'Message vide'}), 400

        prompt = f"Human: {user_message}\nBot:"

        response = generate_response(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )

        return jsonify({
            'response': response,
            'success': True
        })

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"❌ Erreur dans /api/chat:")
        print(error_trace)
        
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/api/config', methods=['GET'])
def get_config():
    ensure_model_loaded()

    return jsonify({
        'vocab_size': config.get('vocab_size'),
        'embed_dim': config.get('embed_dim'),
        'num_heads': config.get('num_heads'),
        'num_layers': config.get('num_layers'),
        'max_seq_len': config.get('max_seq_len')
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Démarrage de l'application...")
    print("="*60 + "\n")

    load_model_and_tokenizer()

    print("\n" + "="*60)
    print("✅ Serveur prêt!")
    print("="*60 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=False)
