import subprocess
import sys

class EdgeTTS:
    def __init__(self, gender="female"):
        if gender.lower() == "female":
            self.voice_name = "hi-IN-SwaraNeural"
        else:
            self.voice_name = "hi-IN-MadhurNeural"
            
        print(f"Initializing Edge-TTS Engine with {gender} voice: {self.voice_name}...")

    def generate_audio(self, text: str, output_path="hindi.wav"):
        if not text or not text.strip():
            text = "क्षमा करें, कोई आवाज़ नहीं मिली।"
            
        print(f"Generating natural {self.voice_name} speech...")
        
        import shutil
        edge_tts_path = shutil.which("edge-tts") or "edge-tts"
        
        with open("tts_temp.txt", "w", encoding="utf-8") as f:
            f.write(text)
            
        cmd = [
            edge_tts_path,
            "--voice", self.voice_name,
            "-f", "tts_temp.txt",
            "--write-media", output_path
        ]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode('utf-8') if e.stderr else "Unknown Error"
            print(f"\nEdge-TTS Error: {error_msg}")
            raise RuntimeError(f"Edge-TTS failed to generate audio. Is it installed? (pip install edge-tts)")
            
        print(f"TTS output saved successfully to {output_path}")

_tts_instances = {}

def generate_hindi_audio(text: str, output_path: str, gender: str = "female"):
    global _tts_instances
    if gender not in _tts_instances:
        _tts_instances[gender] = EdgeTTS(gender=gender)
        
    _tts_instances[gender].generate_audio(text, output_path)