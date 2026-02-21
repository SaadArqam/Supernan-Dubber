from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch

MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"

tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_NAME)
model = MBartForConditionalGeneration.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def translate_kn_to_hi(text):
    tokenizer.src_lang = "kn_IN"

    encoded = tokenizer(text, return_tensors="pt").to(device)

    generated_tokens = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.lang_code_to_id["hi_IN"],
        max_length=512
    )

    translated_text = tokenizer.batch_decode(
        generated_tokens, skip_special_tokens=True
    )[0]

    return translated_text