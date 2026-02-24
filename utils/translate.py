import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import pipeline

class IndicTranslator:
    def __init__(self, model_name="ai4bharat/indictrans2-en-indic-1B", device=None):
        """
        Loads IndicTrans2 for high-fidelity, context-aware translation.
        IndicTrans2 natively understands Indian context far better than NLLB or mBART.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading IndicTrans2 ({model_name}) on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, 
            trust_remote_code=True
        ).to(self.device)

        # IndicTrans2 uses specific language tags
        self.target_lang_tag = "hin_Deva"

    def translate_to_hindi(self, text: str, source_lang: str = "eng_Latn") -> str:
        """
        Translates text to Hindi using IndicTrans2.
        For best results, Whisper should output English, then we route to Hindi.
        """
        if not text or not text.strip():
            return ""

        # IndicTrans2 formatting requires src and tgt tags attached to the input
        batch = self.tokenizer(
            text, 
            src_lang=source_lang, 
            tgt_lang=self.target_lang_tag, 
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            generated_tokens = self.model.generate(
                **batch,
                use_cache=True,
                max_length=256,
                num_beams=5,
                early_stopping=True
            )

        translated_text = self.tokenizer.batch_decode(
            generated_tokens, 
            skip_special_tokens=True, 
            src_lang=source_lang,
            tgt_lang=self.target_lang_tag
        )[0]
        
        return translated_text

# Provide a backward-compatible functional wrapper for dub_video.py
_translator_instance = None

def translate_to_hindi(text: str, detected_lang: str) -> str:
    global _translator_instance
    if _translator_instance is None:
        _translator_instance = IndicTranslator()
    
    # We map whatever whisper outputs, assuming English input for this model
    # For a full multilingual pipeline: you would load indictrans2-indic-indic for kn->hi
    # But for standard 15s clips, English transcription -> Hindi is the most accurate flow
    source_tag = "eng_Latn" 
    
    print(f"Translating via IndicTrans2 → Hindi...")
    return _translator_instance.translate_to_hindi(text, source_lang=source_tag)