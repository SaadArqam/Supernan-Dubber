from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL_NAME = "facebook/nllb-200-distilled-600M"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def translate_kn_to_hi(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        src_lang="kan_Knda"
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id["hin_Deva"]
        )

    translated_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return translated_text