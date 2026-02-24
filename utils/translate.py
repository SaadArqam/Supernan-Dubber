from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_name = "facebook/nllb-200-distilled-600M"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

lang_map = {
    "en": "eng_Latn",
    "kn": "kan_Knda",
    "hi": "hin_Deva",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "ml": "mal_Mlym",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn"
}

def translate_to_hindi(text, detected_lang):
    print(f"Translating from {detected_lang} → Hindi...")

    src_lang = lang_map.get(detected_lang, "eng_Latn")

    tokenizer.src_lang = src_lang

    encoded = tokenizer(text, return_tensors="pt")

    hindi_token_id = tokenizer.convert_tokens_to_ids("hin_Deva")

    generated_tokens = model.generate(
        **encoded,
        forced_bos_token_id=hindi_token_id,
        max_length=512
    )

    result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    return result[0]