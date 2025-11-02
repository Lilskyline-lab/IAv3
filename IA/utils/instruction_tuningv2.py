"""
Module d'Instruction Tuning MODERNE pour LLM (2025)
VERSION CORRIGÉE avec Loss Masking Proper
"""

import json
import csv
import os
import torch
from typing import List, Dict, Union, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum


class InstructionFormat(Enum):
    """Formats d'instruction modernes (2025)"""
    CHATML = "chatml"
    LLAMA3 = "llama3"
    MISTRAL = "mistral"
    ALPACA = "alpaca"
    CHAT_SIMPLE = "chat_simple"
    VICUNA = "vicuna"
    ZEPHYR = "zephyr"
    CUSTOM = "custom"


@dataclass
class Message:
    """Représente un message dans une conversation"""
    role: str
    content: str
    
    def __post_init__(self):
        valid_roles = {"system", "user", "assistant", "human", "bot"}
        if self.role.lower() not in valid_roles:
            raise ValueError(f"Role invalide: {self.role}. Attendu: {valid_roles}")


@dataclass
class Conversation:
    """Représente une conversation multi-tours"""
    messages: List[Message]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, role: str, content: str):
        self.messages.append(Message(role, content))
    
    def to_dict(self) -> Dict:
        return {
            "messages": [{"role": m.role, "content": m.content} for m in self.messages],
            "metadata": self.metadata
        }


@dataclass
class InstructionTemplate:
    """Template moderne pour formater les instructions"""
    name: str
    format_type: InstructionFormat
    system_prefix: str = ""
    user_prefix: str = ""
    user_suffix: str = ""
    assistant_prefix: str = ""
    assistant_suffix: str = ""
    separator: str = ""
    end_token: str = ""
    supports_system: bool = True
    supports_multi_turn: bool = True
    
    # NOUVEAU : Permet de marquer les zones à masquer
    mask_prompt: bool = True  # Masquer le prompt de l'utilisateur
    mask_system: bool = True  # Masquer le message système
    
    def format_message(self, role: str, content: str) -> str:
        """Formate un seul message"""
        role = role.lower()
        
        if role in ("system",):
            return f"{self.system_prefix}{content}"
        elif role in ("user", "human"):
            return f"{self.user_prefix}{content}{self.user_suffix}"
        elif role in ("assistant", "bot"):
            return f"{self.assistant_prefix}{content}{self.assistant_suffix}"
        else:
            raise ValueError(f"Role non supporté: {role}")
    
    def format_conversation(self, messages: List[Message]) -> str:
        """Formate une conversation complète"""
        formatted_parts = []
        
        for msg in messages:
            formatted = self.format_message(msg.role, msg.content)
            formatted_parts.append(formatted)
        
        result = self.separator.join(formatted_parts)
        if self.end_token:
            result += self.end_token
        
        return result
    
    def create_labels_mask(self, tokenizer, text: str, messages: List[Message]) -> Tuple[List[int], List[int]]:
        """
        Crée input_ids et labels avec masquage proper
        
        Returns:
            input_ids: Tous les tokens
            labels: Tokens avec -100 pour masquer les prompts
        """
        # Tokenize le texte complet
        input_ids = tokenizer.encode(text, add_special_tokens=False)
        labels = input_ids.copy()
        
        # Construire le texte progressivement pour trouver les positions
        current_pos = 0
        current_text = ""
        
        for msg in messages:
            # Formater le message
            formatted_msg = self.format_message(msg.role, msg.content)
            msg_start = current_pos
            
            # Tokenize le message
            msg_tokens = tokenizer.encode(formatted_msg, add_special_tokens=False)
            msg_end = msg_start + len(msg_tokens)
            
            # Masquer selon le rôle
            if msg.role.lower() in ("system",) and self.mask_system:
                # Masquer tout le message système
                for i in range(msg_start, min(msg_end, len(labels))):
                    labels[i] = -100
            
            elif msg.role.lower() in ("user", "human") and self.mask_prompt:
                # Masquer tout le prompt utilisateur (incluant [INST] et [/INST])
                for i in range(msg_start, min(msg_end, len(labels))):
                    labels[i] = -100
            
            # Assistant: on garde tout (pas de masque)
            
            current_pos = msg_end
            current_text += formatted_msg
        
        return input_ids, labels


class ModernInstructionTemplates:
    """Collection de templates modernes (2025)"""
    
    @staticmethod
    def get_chatml_template() -> InstructionTemplate:
        """Format ChatML (OpenAI standard)"""
        return InstructionTemplate(
            name="chatml",
            format_type=InstructionFormat.CHATML,
            system_prefix="<|im_start|>system\n",
            user_prefix="<|im_start|>user\n",
            user_suffix="<|im_end|>\n",
            assistant_prefix="<|im_start|>assistant\n",
            assistant_suffix="<|im_end|>\n",
            separator="",
            supports_system=True,
            supports_multi_turn=True,
            mask_prompt=True,
            mask_system=True
        )
    
    @staticmethod
    def get_llama3_template() -> InstructionTemplate:
        """Format Llama 3+ (Meta)"""
        return InstructionTemplate(
            name="llama3",
            format_type=InstructionFormat.LLAMA3,
            system_prefix="<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
            user_prefix="<|start_header_id|>user<|end_header_id|>\n\n",
            user_suffix="<|eot_id|>",
            assistant_prefix="<|start_header_id|>assistant<|end_header_id|>\n\n",
            assistant_suffix="<|eot_id|>",
            separator="",
            supports_system=True,
            supports_multi_turn=True,
            mask_prompt=True,
            mask_system=True
        )
    
    @staticmethod
    def get_mistral_instruct_template() -> InstructionTemplate:
        """
        Format Mistral Instruct - VERSION CORRIGÉE
        [INST] Hello [/INST] Hi there!
        
        CRITIQUE : On masque TOUT le prompt incluant [INST] et [/INST]
        """
        return InstructionTemplate(
            name="mistral_instruct",
            format_type=InstructionFormat.MISTRAL,
            system_prefix="",
            user_prefix="[INST] ",
            user_suffix=" [/INST]",  # Le ] sera masqué !
            assistant_prefix=" ",
            assistant_suffix="</s>",
            separator="",
            supports_system=False,
            supports_multi_turn=True,
            mask_prompt=True,  # CRITIQUE : Masque tout le prompt
            mask_system=False
        )
    
    @staticmethod
    def get_zephyr_template() -> InstructionTemplate:
        """Format Zephyr (HuggingFaceH4)"""
        return InstructionTemplate(
            name="zephyr",
            format_type=InstructionFormat.ZEPHYR,
            system_prefix="<|system|>\n",
            user_prefix="<|user|>\n",
            user_suffix="</s>\n",
            assistant_prefix="<|assistant|>\n",
            assistant_suffix="</s>\n",
            separator="",
            supports_system=True,
            supports_multi_turn=True,
            mask_prompt=True,
            mask_system=True
        )
    
    @staticmethod
    def get_chat_simple_template() -> InstructionTemplate:
        """Format simple Human/Assistant"""
        return InstructionTemplate(
            name="chat_simple",
            format_type=InstructionFormat.CHAT_SIMPLE,
            system_prefix="System: ",
            user_prefix="Human: ",
            user_suffix="\n",
            assistant_prefix="Assistant: ",
            assistant_suffix="\n",
            separator="",
            supports_system=True,
            supports_multi_turn=True,
            mask_prompt=True,
            mask_system=True
        )
    
    @staticmethod
    def get_chat_bot_template() -> InstructionTemplate:
        """Format Human/Bot (legacy)"""
        return InstructionTemplate(
            name="chat_bot",
            format_type=InstructionFormat.CHAT_SIMPLE,
            system_prefix="System: ",
            user_prefix="Human: ",
            user_suffix="\n",
            assistant_prefix="Bot: ",
            assistant_suffix="\n",
            separator="",
            supports_system=True,
            supports_multi_turn=True,
            mask_prompt=True,
            mask_system=True
        )
    
    @staticmethod
    def get_alpaca_template() -> InstructionTemplate:
        """Format Alpaca (legacy)"""
        return InstructionTemplate(
            name="alpaca",
            format_type=InstructionFormat.ALPACA,
            system_prefix="Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n",
            user_prefix="### Instruction:\n",
            user_suffix="\n\n",
            assistant_prefix="### Response:\n",
            assistant_suffix="",
            separator="",
            supports_system=False,
            supports_multi_turn=False,
            mask_prompt=True,
            mask_system=False
        )


class InstructionDataFormatter:
    """Formatter moderne avec support multi-tours et loss masking"""
    
    def __init__(
        self,
        template: Optional[InstructionTemplate] = None,
        template_name: str = "chatml",
        tokenizer = None
    ):
        """
        Args:
            template: Template personnalisé
            template_name: Nom du template prédéfini
            tokenizer: Tokenizer pour le masquage
        """
        if template:
            self.template = template
        else:
            self.template = self._get_template_by_name(template_name)
        
        self.tokenizer = tokenizer
        print(f"📝 Formatter initialisé avec template: {self.template.name}")
        if tokenizer:
            print(f"   ✅ Loss masking activé (masque prompts: {self.template.mask_prompt})")
    
    def _get_template_by_name(self, name: str) -> InstructionTemplate:
        """Récupère un template par son nom"""
        templates = {
            "chatml": ModernInstructionTemplates.get_chatml_template(),
            "llama3": ModernInstructionTemplates.get_llama3_template(),
            "mistral": ModernInstructionTemplates.get_mistral_instruct_template(),
            "zephyr": ModernInstructionTemplates.get_zephyr_template(),
            "chat_simple": ModernInstructionTemplates.get_chat_simple_template(),
            "chat_bot": ModernInstructionTemplates.get_chat_bot_template(),
            "alpaca": ModernInstructionTemplates.get_alpaca_template(),
        }
        
        if name not in templates:
            print(f"⚠️  Template '{name}' inconnu, utilisation de 'chatml' par défaut")
            return templates["chatml"]
        
        return templates[name]
    
    def format_single_turn(self, user_message: str, assistant_message: str, system_message: str = "") -> str:
        """Formate une conversation simple (un seul tour)"""
        messages = []
        
        if system_message and self.template.supports_system:
            messages.append(Message("system", system_message))
        
        messages.append(Message("user", user_message))
        messages.append(Message("assistant", assistant_message))
        
        return self.template.format_conversation(messages)
    
    def format_single_turn_with_labels(
        self, 
        user_message: str, 
        assistant_message: str, 
        system_message: str = ""
    ) -> Dict[str, Any]:
        """
        Formate avec création des labels pour le masquage
        
        Returns:
            {
                "text": str,
                "input_ids": List[int],
                "labels": List[int]  # Avec -100 pour masquer
            }
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer requis pour créer les labels")
        
        messages = []
        
        if system_message and self.template.supports_system:
            messages.append(Message("system", system_message))
        
        messages.append(Message("user", user_message))
        messages.append(Message("assistant", assistant_message))
        
        text = self.template.format_conversation(messages)
        input_ids, labels = self.template.create_labels_mask(self.tokenizer, text, messages)
        
        return {
            "text": text,
            "input_ids": input_ids,
            "labels": labels
        }
    
    def format_multi_turn(self, conversation: Conversation) -> str:
        """Formate une conversation multi-tours"""
        if not self.template.supports_multi_turn:
            print(f"⚠️  Template {self.template.name} ne supporte pas multi-turn")
            if len(conversation.messages) >= 2:
                last_user = None
                last_assistant = None
                for msg in reversed(conversation.messages):
                    if msg.role in ("assistant", "bot") and not last_assistant:
                        last_assistant = msg
                    if msg.role in ("user", "human") and not last_user:
                        last_user = msg
                
                if last_user and last_assistant:
                    return self.format_single_turn(last_user.content, last_assistant.content)
        
        return self.template.format_conversation(conversation.messages)
    
    def format_multi_turn_with_labels(self, conversation: Conversation) -> Dict[str, Any]:
        """Formate multi-tour avec labels"""
        if not self.tokenizer:
            raise ValueError("Tokenizer requis")
        
        text = self.format_multi_turn(conversation)
        input_ids, labels = self.template.create_labels_mask(
            self.tokenizer, text, conversation.messages
        )
        
        return {
            "text": text,
            "input_ids": input_ids,
            "labels": labels
        }
    
    def format_from_dict(self, data: Dict, with_labels: bool = False) -> Union[str, Dict]:
        """
        Formate depuis un dictionnaire
        
        Args:
            data: Dictionnaire de données
            with_labels: Si True, retourne avec input_ids et labels
        """
        # Format multi-tour
        if "messages" in data:
            messages = [Message(m["role"], m["content"]) for m in data["messages"]]
            conv = Conversation(messages)
            
            if with_labels:
                return self.format_multi_turn_with_labels(conv)
            return self.format_multi_turn(conv)
        
        # Format simple
        if "human" in data and "assistant" in data:
            system = data.get("system", "")
            
            if with_labels:
                return self.format_single_turn_with_labels(
                    data["human"], data["assistant"], system
                )
            return self.format_single_turn(data["human"], data["assistant"], system)
        
        # Alias: user/bot
        if "user" in data and "assistant" in data:
            system = data.get("system", "")
            
            if with_labels:
                return self.format_single_turn_with_labels(
                    data["user"], data["assistant"], system
                )
            return self.format_single_turn(data["user"], data["assistant"], system)
        
        # Format Alpaca
        if "instruction" in data:
            instruction = data["instruction"]
            input_text = data.get("input", "")
            output = data.get("output", "")
            
            user_msg = f"{instruction}\n\nInput: {input_text}" if input_text else instruction
            
            if with_labels:
                return self.format_single_turn_with_labels(user_msg, output)
            return self.format_single_turn(user_msg, output)
        
        raise ValueError(f"Format non reconnu: {list(data.keys())}")


class InstructionDatasetLoader:
    """Loader avec support moderne"""
    
    @staticmethod
    def load_json(file_path: str) -> List[Dict]:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            if 'data' in data:
                return data['data']
            elif 'conversations' in data:
                return data['conversations']
        
        raise ValueError("Format JSON invalide")
    
    @staticmethod
    def load_jsonl(file_path: str) -> List[Dict]:
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    
    @staticmethod
    def load_csv(file_path: str) -> List[Dict]:
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(dict(row))
        return data
    
    @staticmethod
    def load_dataset(file_path: str) -> List[Dict]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Fichier non trouvé: {file_path}")
        
        ext = os.path.splitext(file_path)[1].lower()
        
        loaders = {
            '.json': InstructionDatasetLoader.load_json,
            '.jsonl': InstructionDatasetLoader.load_jsonl,
            '.csv': InstructionDatasetLoader.load_csv,
        }
        
        if ext not in loaders:
            raise ValueError(f"Format non supporté: {ext}")
        
        return loaders[ext](file_path)
    
    @staticmethod
    def save_jsonl(data: List[Dict], file_path: str):
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')


def convert_to_instruction_format(
    data: List[Dict],
    template_name: str = "chatml",
    system_message: str = "",
    tokenizer = None,
    with_labels: bool = False
) -> List[Dict]:
    """
    Convertit des données au format instruction moderne
    
    Args:
        data: Liste de dictionnaires
        template_name: Template à utiliser
        system_message: Message système optionnel
        tokenizer: Tokenizer pour le masquage
        with_labels: Si True, crée input_ids et labels
    
    Returns:
        Liste avec "formatted_text" (et optionnellement "input_ids"/"labels")
    """
    formatter = InstructionDataFormatter(
        template_name=template_name,
        tokenizer=tokenizer
    )
    
    result = []
    errors = 0
    
    for item in data:
        try:
            if system_message and "system" not in item and "messages" not in item:
                item = {**item, "system": system_message}
            
            formatted = formatter.format_from_dict(item, with_labels=with_labels)
            
            result_item = item.copy()
            
            if with_labels and isinstance(formatted, dict):
                result_item["formatted_text"] = formatted["text"]
                result_item["input_ids"] = formatted["input_ids"]
                result_item["labels"] = formatted["labels"]
            else:
                result_item["formatted_text"] = formatted
            
            result.append(result_item)
        
        except Exception as e:
            errors += 1
            print(f"⚠️  Erreur formatage: {e}")
            continue
    
    if errors > 0:
        print(f"⚠️  {errors}/{len(data)} exemples ont échoué")
    
    print(f"✅ {len(result)}/{len(data)} exemples formatés avec succès")
    
    return result


if __name__ == "__main__":
    print("="*70)
    print("MODULE D'INSTRUCTION TUNING MODERNE - VERSION CORRIGÉE")
    print("="*70)
    
    # Test du masquage
    print("\n🔍 TEST DU LOSS MASKING")
    print("="*70)
    
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    
    sample = {
        "human": "Hello!",
        "assistant": "Hi there! How can I help you today?"
    }
    
    formatter = InstructionDataFormatter(
        template_name="mistral",
        tokenizer=tokenizer
    )
    
    result = formatter.format_from_dict(sample, with_labels=True)
    
    print(f"\n📝 Text:\n{result['text']}")
    print(f"\n🔢 Input IDs (premiers 20): {result['input_ids'][:20]}")
    print(f"\n🎯 Labels (premiers 20): {result['labels'][:20]}")
    print(f"\n✅ Tokens masqués (-100): {result['labels'].count(-100)}/{len(result['labels'])}")
    
    print("\n" + "="*70)