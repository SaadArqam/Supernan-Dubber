import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

class TranslatorHub:
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct", device=None):
        """
        Loads Qwen2.5-1.5B-Instruct, an incredibly smart open-weight LLM that 
        understands context far better than standard translation models (NLLB/mBART).
        This guarantees non-literal, conversational, natural-sounding Hindi dubs.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading contextual translation LLM ({model_name}) on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        ).to(self.device)

    def translate_to_hindi(self, text: str, detected_lang: str) -> str:
        """
        Uses an LLM prompt to instruct the model to perform high-quality, 
        context-aware dubbing translation into strictly Devanagari Hindi.
        """
        print(f"Translating from {detected_lang} contextually → Hindi...")
        if not text or not text.strip():
            return ""

        system_prompt = (
            "You are an expert movie dubbing translator. "
            "Your task is to translate the provided text into highly natural, conversational Hindi. "
            "Do NOT provide literal word-for-word translations. Ensure the tone matches spoken Hindi. "
            "Respond ONLY with the Hindi translation in Devanagari script. Do not include quotes, explanations, or english words."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]

        text_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text_input], return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=256,
                temperature=0.3, # Low temp for translation accuracy
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Extract only the newly generated tokens
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        result = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return result.strip()

# Provide a backward-compatible functional wrapper for dub_video.py
_translator_instance = None

def translate_to_hindi(text: str, detected_lang: str) -> str:
    global _translator_instance
    if _translator_instance is None:
        _translator_instance = TranslatorHub()
        
    return _translator_instance.translate_to_hindi(text, detected_lang)