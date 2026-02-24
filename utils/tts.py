import subprocess
import sys

class EdgeTTS:
    def __init__(self, gender="female"):
        """
        Loads Microsoft's Azure Edge-TTS engine and assigns the voice according to the detected gender.
        """
        # Assign best available neural voices mapped to gender
        if gender.lower() == "female":
            self.voice_name = "hi-IN-SwaraNeural"
        else:
            self.voice_name = "hi-IN-MadhurNeural"
            
        print(f"Initializing Edge-TTS Engine with {gender} voice: {self.voice_name}...")

    def generate_audio(self, text: str, output_path="hindi.wav"):
        if not text or not text.strip():
            text = "क्षमा करें, कोई आवाज़ नहीं मिली।"
            
        print(f"Generating natural {self.voice_name} speech...")
        
        # Edge-TTS runs purely via subprocess to avoid any asyncio loop conflicts in Jupyter/Colab.
        # Calling 'python -m edge_tts' completely bypasses the Colab/Jupyter FileNotFoundError 
        # that happens when the CLI executable gets lost in the linux PATH.
        cmd = [
            sys.executable, "-m", "edge_tts",
            "--voice", self.voice_name,
            "--text", text,
            "--write-media", output_path
        ]
        
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"✅ TTS output saved successfully to {output_path}")

# Singleton dictionary wrapper to maintain instances
_tts_instances = {}

def generate_hindi_audio(text: str, output_path: str, gender: str = "female"):
    global _tts_instances
    if gender not in _tts_instances:
        _tts_instances[gender] = EdgeTTS(gender=gender)
        
    _tts_instances[gender].generate_audio(text, output_path)