"""
Module d'Instruction Tuning MODERNE pour LLM (2025)
Supporte les derniers formats : ChatML, Llama3, Mistral Instruct, etc.
Gestion multi-tours, tokens spéciaux, et validation robuste
"""

import json
import csv
import os
from typing import List, Dict, Union, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum


class InstructionFormat(Enum):
    """Formats d'instruction modernes (2025)"""
    CHATML = "chatml"  # Format OpenAI/ChatGPT (le plus standard)
    LLAMA3 = "llama3"  # Meta Llama 3+ avec <|begin_of_text|>
    MISTRAL = "mistral"  # Mistral Instruct avec [INST]
    ALPACA = "alpaca"  # Legacy Alpaca
    CHAT_SIMPLE = "chat_simple"  # Human/Assistant simple
    VICUNA = "vicuna"  # Legacy Vicuna
    ZEPHYR = "zephyr"  # HuggingFaceH4 Zephyr
    CUSTOM = "custom"


@dataclass
class Message:
    """Représente un message dans une conversation"""
    role: str  # "system", "user", "assistant"
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
        """Ajoute un message à la conversation"""
        self.messages.append(Message(role, content))
    
    def to_dict(self) -> Dict:
        """Convertit en dictionnaire"""
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


class ModernInstructionTemplates:
    """Collection de templates modernes (2025)"""
    
    @staticmethod
    def get_chatml_template() -> InstructionTemplate:
        """
        Format ChatML (OpenAI standard)
        <|im_start|>system
        You are a helpful assistant.<|im_end|>
        <|im_start|>user
        Hello!<|im_end|>
        <|im_start|>assistant
        Hi! How can I help?<|im_end|>
        """
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
            supports_multi_turn=True
        )
    
    @staticmethod
    def get_llama3_template() -> InstructionTemplate:
        """
        Format Llama 3+ (Meta)
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are helpful.<|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        Hello<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        Hi!<|eot_id|>
        """
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
            supports_multi_turn=True
        )
    
    @staticmethod
    def get_mistral_instruct_template() -> InstructionTemplate:
        """
        Format Mistral Instruct
        [INST] Hello [/INST] Hi there!
        """
        return InstructionTemplate(
            name="mistral_instruct",
            format_type=InstructionFormat.MISTRAL,
            system_prefix="",  # Mistral inclut le system dans le premier [INST]
            user_prefix="[INST] ",
            user_suffix=" [/INST]",
            assistant_prefix=" ",
            assistant_suffix="</s>",
            separator="",
            supports_system=False,  # System intégré dans user
            supports_multi_turn=True
        )
    
    @staticmethod
    def get_zephyr_template() -> InstructionTemplate:
        """
        Format Zephyr (HuggingFaceH4)
        <|system|>
        You are helpful.</s>
        <|user|>
        Hello</s>
        <|assistant|>
        Hi!</s>
        """
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
            supports_multi_turn=True
        )
    
    @staticmethod
    def get_chat_simple_template() -> InstructionTemplate:
        """
        Format simple Human/Assistant
        Human: Hello
        Assistant: Hi there!
        """
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
            supports_multi_turn=True
        )
    
    @staticmethod
    def get_chat_bot_template() -> InstructionTemplate:
        """Format Human/Bot (legacy, compatible avec ancien code)"""
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
            supports_multi_turn=True
        )
    
    @staticmethod
    def get_alpaca_template() -> InstructionTemplate:
        """Format Alpaca (legacy)"""
        # Note: Alpaca ne supporte pas vraiment multi-turn
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
            supports_multi_turn=False
        )


class InstructionDataFormatter:
    """Formatter moderne avec support multi-tours"""
    
    def __init__(
        self,
        template: Optional[InstructionTemplate] = None,
        template_name: str = "chatml"
    ):
        """
        Args:
            template: Template personnalisé (prioritaire)
            template_name: Nom du template prédéfini si template=None
        """
        if template:
            self.template = template
        else:
            self.template = self._get_template_by_name(template_name)
        
        print(f"📝 Formatter initialisé avec template: {self.template.name}")
    
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
    
    def format_multi_turn(self, conversation: Conversation) -> str:
        """Formate une conversation multi-tours"""
        if not self.template.supports_multi_turn:
            print(f"⚠️  Template {self.template.name} ne supporte pas multi-turn")
            # Prendre seulement le dernier tour
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
    
    def format_from_dict(self, data: Dict) -> str:
        """
        Formate depuis un dictionnaire
        
        Formats supportés:
        1. Simple: {"human": "...", "assistant": "..."}
        2. Avec system: {"human": "...", "assistant": "...", "system": "..."}
        3. Multi-tour: {"messages": [{"role": "...", "content": "..."}, ...]}
        4. Alpaca: {"instruction": "...", "input": "...", "output": "..."}
        """
        # Format 3: Multi-tour
        if "messages" in data:
            messages = [Message(m["role"], m["content"]) for m in data["messages"]]
            conv = Conversation(messages)
            return self.format_multi_turn(conv)
        
        # Format 1 & 2: Simple avec/sans system
        if "human" in data and "assistant" in data:
            system = data.get("system", "")
            return self.format_single_turn(data["human"], data["assistant"], system)
        
        # Alias: user/bot
        if "user" in data and "assistant" in data:
            system = data.get("system", "")
            return self.format_single_turn(data["user"], data["assistant"], system)
        
        # Format 4: Alpaca
        if "instruction" in data:
            instruction = data["instruction"]
            input_text = data.get("input", "")
            output = data.get("output", "")
            
            if input_text:
                user_msg = f"{instruction}\n\nInput: {input_text}"
            else:
                user_msg = instruction
            
            return self.format_single_turn(user_msg, output)
        
        raise ValueError(f"Format de données non reconnu: {list(data.keys())}")


class InstructionDatasetLoader:
    """Loader avec support moderne"""
    
    @staticmethod
    def load_json(file_path: str) -> List[Dict]:
        """Charge JSON"""
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
        """Charge JSONL"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    
    @staticmethod
    def load_csv(file_path: str) -> List[Dict]:
        """Charge CSV"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(dict(row))
        return data
    
    @staticmethod
    def load_dataset(file_path: str) -> List[Dict]:
        """Auto-détecte et charge"""
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
        """Sauvegarde en JSONL"""
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')


def convert_to_instruction_format(
    data: List[Dict],
    template_name: str = "chatml",
    system_message: str = ""
) -> List[Dict]:
    """
    Convertit des données au format instruction moderne
    
    Args:
        data: Liste de dictionnaires
        template_name: "chatml", "llama3", "mistral", "chat_simple", etc.
        system_message: Message système optionnel à ajouter
    
    Returns:
        Liste avec le champ "formatted_text" ajouté
    """
    formatter = InstructionDataFormatter(template_name=template_name)
    
    result = []
    errors = 0
    
    for item in data:
        try:
            # Ajouter system_message si fourni et pas déjà présent
            if system_message and "system" not in item and "messages" not in item:
                item = {**item, "system": system_message}
            
            formatted_text = formatter.format_from_dict(item)
            
            result_item = item.copy()
            result_item["formatted_text"] = formatted_text
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
    print("MODULE D'INSTRUCTION TUNING MODERNE (2025)")
    print("="*70)
    
    # Exemples de données
    sample_single = {
        "human": "Qu'est-ce que l'IA ?",
        "assistant": "L'IA (Intelligence Artificielle) est un domaine de l'informatique..."
    }
    
    sample_multi = {
        "messages": [
            {"role": "system", "content": "Tu es un assistant serviable."},
            {"role": "user", "content": "Bonjour!"},
            {"role": "assistant", "content": "Bonjour! Comment puis-je vous aider?"},
            {"role": "user", "content": "Parle-moi de l'IA"},
            {"role": "assistant", "content": "L'IA est fascinante..."}
        ]
    }
    
    # Test des différents templates
    templates = ["chatml", "llama3", "mistral", "chat_simple"]
    
    print("\n📝 TEST: Conversation simple")
    print("="*70)
    for template_name in templates:
        print(f"\n🔹 Template: {template_name}")
        print("-" * 50)
        formatted = convert_to_instruction_format([sample_single], template_name)
        if formatted:
            print(formatted[0]["formatted_text"][:200] + "...")
    
    print("\n\n📝 TEST: Conversation multi-tours")
    print("="*70)
    for template_name in templates:
        print(f"\n🔹 Template: {template_name}")
        print("-" * 50)
        formatted = convert_to_instruction_format([sample_multi], template_name)
        if formatted:
            print(formatted[0]["formatted_text"][:300] + "...")
    
    print("\n" + "="*70)
    print("✅ MODULE PRÊT!")
    print("="*70)