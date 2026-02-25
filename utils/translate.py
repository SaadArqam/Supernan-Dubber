import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class TranslationModel:
    def __init__(self, model_name="facebook/nllb-200-1.3B", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Translation Model: {model_name} on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        
        self.target_lang_nllb = "hin_Deva"
        
        self.lang_map = {
            "en": "eng_Latn", "kn": "kan_Knda", "ta": "tam_Taml", "te": "tel_Telu",
            "ml": "mal_Mlym", "mr": "mar_Deva", "gu": "guj_Gujr", "pa": "pan_Guru",
            "bn": "ben_Beng", "ur": "urd_Arab", "or": "ory_Orya", "as": "asm_Beng",
            "fr": "fra_Latn", "de": "deu_Latn", "es": "spa_Latn", "it": "ita_Latn",
            "pt": "por_Latn", "ru": "rus_Cyrl", "zh": "zho_Hans", "ja": "jpn_Jpan",
            "ko": "kor_Hang", "ar": "arb_Arab", "tr": "tur_Latn", "hi": "hin_Deva",
        }

    def translate(self, text: str, source_lang_whisper: str) -> str:
        if not text.strip():
            return ""

        src_lang = self.lang_map.get(source_lang_whisper, "eng_Latn")
        self.tokenizer.src_lang = src_lang
        
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        
        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(self.target_lang_nllb)

        print(f"Translating from '{src_lang}' to '{self.target_lang_nllb}'...")
        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=512,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
            
        result = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        print(f"Translation complete.")
        return result.strip()


_translator_instance = None

def translate_to_hindi(text: str, detected_lang: str) -> str:
    global _translator_instance
    if _translator_instance is None:
        _translator_instance = TranslationModel()
    return _translator_instance.translate(text, detected_lang)