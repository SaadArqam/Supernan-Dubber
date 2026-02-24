import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

model_name = "facebook/nllb-200-1.3B"
print(f"Loading translation model {model_name}...")
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
target_lang_nllb = "hin_Deva"

WHISPER_TO_NLLB = {
    "en": "eng_Latn", "kn": "kan_Knda", "ta": "tam_Taml", "te": "tel_Telu",
    "ml": "mal_Mlym", "mr": "mar_Deva", "gu": "guj_Gujr", "pa": "pan_Guru",
    "bn": "ben_Beng", "ur": "urd_Arab", "or": "ory_Orya", "as": "asm_Beng",
    "fr": "fra_Latn", "de": "deu_Latn", "es": "spa_Latn", "it": "ita_Latn",
    "pt": "por_Latn", "ru": "rus_Cyrl", "zh": "zho_Hans", "ja": "jpn_Jpan",
    "ko": "kor_Hang", "ar": "arb_Arab", "tr": "tur_Latn", "hi": "hin_Deva",
}

def translate_to_hindi(text, detected_lang):
    print(f"Translating from {detected_lang} → Hindi...")
    if not text or not text.strip():
        return ""

    source_lang_nllb = WHISPER_TO_NLLB.get(detected_lang.lower(), "eng_Latn")
    tokenizer.src_lang = source_lang_nllb

    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=512
    ).to(device)

    with torch.no_grad():
        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id[target_lang_nllb],
            max_length=512,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=3,
            temperature=0.7,
            do_sample=False
        )

    result = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return result