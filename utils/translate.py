from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

model_name = "facebook/mbart-large-50-many-to-many-mmt"

tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

lang_map = {
    "en": "en_XX",
    "kn": "kn_IN",
    "hi": "hi_IN",
    "ta": "ta_IN",
    "te": "te_IN",
    "ml": "ml_IN",
    "fr": "fr_XX",
    "de": "de_DE",
    "es": "es_XX"
}

def translate_to_hindi(text, detected_lang):
    print(f"Translating from {detected_lang} → Hindi...")

    src_lang = lang_map.get(detected_lang, "en_XX")

    tokenizer.src_lang = src_lang

    encoded = tokenizer(text, return_tensors="pt")

    generated_tokens = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.lang_code_to_id["hi_IN"],
        max_length=512
    )

    result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    return result[0]