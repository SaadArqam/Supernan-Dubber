import subprocess

class HindiFemaleTTS:
    def __init__(self, voice_name="hi-IN-SwaraNeural"):
        """
        Uses Microsoft's Azure Edge-TTS engine for highly natural Hindi voices.
        Unlike local HuggingFace TTS models (MMS, Bark, XTTS) which either default
        to male voices, hallucinate, or cause extreme pip dependency conflicts in Colab,
        Edge-TTS provides a flawless, production-grade female neural voice instantly
        with zero dependencies required other than `edge-tts`.
        """
        self.voice_name = voice_name
        print(f"Initializing Edge-TTS Engine with female voice: {self.voice_name}...")

    def generate_audio(self, text: str, output_path="hindi.wav"):
        if not text or not text.strip():
            text = "क्षमा करें, कोई आवाज़ नहीं मिली।"
            
        print("Generating natural female Hindi speech...")
        
        # Edge-TTS runs purely via subprocess to avoid any asyncio loop conflicts in Jupyter/Colab
        cmd = [
            "edge-tts",
            "--voice", self.voice_name,
            "--text", text,
            "--write-media", output_path
        ]
        
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"✅ Female TTS output saved successfully to {output_path}")

# Singleton wrapper
_tts_idx = None
def generate_hindi_audio(text: str, output_path: str):
    global _tts_idx
    if _tts_idx is None:
        _tts_idx = HindiFemaleTTS()
    _tts_idx.generate_audio(text, output_path)