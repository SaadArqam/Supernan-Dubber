import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

class TranslatorHub:
    def __init__(self, model_name="facebook/nllb-200-1.3B", device=None):
        """
        Loads NLLB-200 1.3B.
        AI4Bharat models are currently gated behind HuggingFace Auth tokens.
        NLLB 1.3B is fully public, ungated, and perfectly capable of high-fidelity Hindi translation.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading translation model {model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.target_lang_nllb = "hin_Deva"

        self.WHISPER_TO_NLLB = {
            "en": "eng_Latn", "kn": "kan_Knda", "ta": "tam_Taml", "te": "tel_Telu",
            "ml": "mal_Mlym", "mr": "mar_Deva", "gu": "guj_Gujr", "pa": "pan_Guru",
            "bn": "ben_Beng", "ur": "urd_Arab", "or": "ory_Orya", "as": "asm_Beng",
            "fr": "fra_Latn", "de": "deu_Latn", "es": "spa_Latn", "it": "ita_Latn",
            "pt": "por_Latn", "ru": "rus_Cyrl", "zh": "zho_Hans", "ja": "jpn_Jpan",
            "ko": "kor_Hang", "ar": "arb_Arab", "tr": "tur_Latn", "hi": "hin_Deva",
        }

    def translate_to_hindi(self, text: str, detected_lang: str) -> str:
        """
        Translates text with hard-enforced Hindi Devanagari output bounding.
        """
        print(f"Translating from {detected_lang} → Hindi...")
        if not text or not text.strip():
            return ""

        source_lang_nllb = self.WHISPER_TO_NLLB.get(detected_lang.lower(), "eng_Latn")
        self.tokenizer.src_lang = source_lang_nllb

        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            translated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(self.target_lang_nllb),
                max_length=512,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=3,
                temperature=0.7,
                do_sample=False
            )

        result = self.tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        return result

# Provide a backward-compatible functional wrapper for dub_video.py
_translator_instance = None

def translate_to_hindi(text: str, detected_lang: str) -> str:
    global _translator_instance
    if _translator_instance is None:
        _translator_instance = TranslatorHub()
        
    return _translator_instance.translate_to_hindi(text, detected_lang)