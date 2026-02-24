import torch
import scipy.io.wavfile as wavfile
from transformers import VitsModel, AutoTokenizer

class HindiTTS:
    def __init__(self, model_name="facebook/mms-tts-hin", device=None):
        """
        Uses standard MMS-Hindi (Massively Multilingual Speech) developed by Meta.
        It runs purely on HuggingFace Transformers logic, eliminating the extremely
        volatile 'coqui-tts' dependency conflicts (numpy/librosa version issues).
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading TTS Model: {model_name} on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = VitsModel.from_pretrained(model_name).to(self.device)

    def generate_audio(self, text: str, output_path="hindi.wav"):
        if not text or not text.strip():
            text = "क्षमा करें, कोई आवाज़ नहीं मिली।"
            
        print("Generating Hindi speech waveform...")
        
        # Tokenize the Devanagari Hindi text
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # Generate raw waveform
        with torch.no_grad():
            output = self.model(**inputs).waveform
            
        audio_data = output.cpu().numpy().squeeze()
        
        # Save securely using scipy
        wavfile.write(output_path, self.model.config.sampling_rate, audio_data)
        print(f"✅ TTS output saved successfully to {output_path}")

# Singleton wrapper
_tts_idx = None
def generate_hindi_audio(text: str, output_path: str):
    global _tts_idx
    if _tts_idx is None:
        _tts_idx = HindiTTS()
    _tts_idx.generate_audio(text, output_path)