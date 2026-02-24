import os
import torch
from TTS.api import TTS

class XTTSVoiceCloner:
    def __init__(self, model_name="tts_models/multilingual/multi-dataset/xtts_v2", device=None):
        """
        Loads Coqui XTTS-v2 for zero-shot voice cloning.
        Requires agreeing to Coqui terms on first download in Colab.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading XTTS Voice Cloner ({model_name}) on {self.device}...")
        
        # Load the XTTS model
        self.tts = TTS(model_name).to(self.device)
        self.language = "hi" # Hindi output

    def generate_hindi_audio(self, text: str, output_path: str, reference_audio_path: str):
        """
        Clones the voice from reference_audio_path and speaks the Hindi text.
        """
        if not os.path.exists(reference_audio_path):
            raise FileNotFoundError(f"Reference audio not found at: {reference_audio_path}")
            
        print(f"Cloning voice from {reference_audio_path}...")
        
        # Generate speech with voice cloning
        self.tts.tts_to_file(
            text=text,
            file_path=output_path,
            speaker_wav=reference_audio_path,
            language=self.language
        )
        print(f"Voice cloned audio saved to {output_path}")

# Provide a backward-compatible functional wrapper for dub_video.py
_cloner_instance = None

def generate_hindi_audio(text: str, output_path: str, reference_audio_path: str = "clip.wav"):
    global _cloner_instance
    if _cloner_instance is None:
        _cloner_instance = XTTSVoiceCloner()
        
    _cloner_instance.generate_hindi_audio(text, output_path, reference_audio_path)