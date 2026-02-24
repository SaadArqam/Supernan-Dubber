import os
import torch

class XTTSVoiceCloner:
    def __init__(self, model_name="tts_models/multilingual/multi-dataset/xtts_v2", device=None):
        """
        Loads Coqui XTTS-v2 for zero-shot voice cloning.
        Requires agreeing to Coqui terms on first download in Colab.
        """
        try:
            from TTS.api import TTS
        except ImportError:
            raise ImportError(
                "Coqui TTS is not installed or incompatible.\n"
                "Google Colab now uses Python 3.12, so you must install the community fork:\n"
                "!pip install coqui-tts"
            )
            
        # Programmatically agree to the Coqui TTS terms of service 
        # By doing this, we bypass the interactive prompt in Colab which hangs execution
        os.environ["COQUI_TOS_AGREED"] = "1"
        
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading XTTS Voice Cloner ({model_name}) on {self.device}...")
        
        # Load the XTTS model
        self.tts = TTS(model_name).to(self.device)
        self.language = "hi" # Hindi output

    def generate_hindi_audio(self, text: str, output_path: str, reference_audio_path: str):
        """
        Clones the voice from reference_audio_path and speaks the Hindi text.
        Splits text into chunks to prevent XTTS audio degradation/cracking on long inputs.
        """
        import re
        import soundfile as sf
        import numpy as np

        if not os.path.exists(reference_audio_path):
            raise FileNotFoundError(f"Reference audio not found at: {reference_audio_path}")
            
        print(f"Cloning voice from {reference_audio_path}...")
        
        if not text or not text.strip():
            print("Warning: Received empty text for TTS. Falling back to default audio safety phrase.")
            text = "क्षमा करें, मैं इस ऑडियो को नहीं समझ पाया।"
            
        # Split by Hindi/English punctuation to prevent XTTS overflow cracking
        # Matches |, ., !, ?, ,, or newlines
        chunks = re.split(r'([।\.!\?,\n])', text)
        
        # Re-attach punctuation to the chunks
        sentences = []
        for i in range(0, len(chunks)-1, 2):
            sentences.append((chunks[i] + chunks[i+1]).strip())
        if len(chunks) % 2 != 0 and chunks[-1].strip():
            sentences.append(chunks[-1].strip())
            
        print(f"Split text into {len(sentences)} chunks to prevent audio cracking...")

        all_wavs = []
        sample_rate = 24000 # Default XTTS output SR
        
        for idx, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
            
            temp_out = f"temp_xtts_chunk_{idx}.wav"
            print(f"Generating chunk {idx+1}/{len(sentences)}: {sentence}")
            
            self.tts.tts_to_file(
                text=sentence,
                file_path=temp_out,
                speaker_wav=reference_audio_path,
                language=self.language
            )
            
            # Read generated chunk and append to list
            data, sr = sf.read(temp_out)
            sample_rate = sr
            all_wavs.append(data)
            
            # Cleanup temp file
            os.remove(temp_out)

        # Stitch all audio chunks together seamlessly
        if all_wavs:
            final_audio = np.concatenate(all_wavs)
            sf.write(output_path, final_audio, sample_rate)
            print(f"Voice cloned audio stitched and saved safely to {output_path}")
        else:
            print("Failed to generate any valid audio chunks.")

# Provide a backward-compatible functional wrapper for dub_video.py
_cloner_instance = None

def generate_hindi_audio(text: str, output_path: str, reference_audio_path: str = "clip.wav"):
    global _cloner_instance
    if _cloner_instance is None:
        _cloner_instance = XTTSVoiceCloner()
        
    _cloner_instance.generate_hindi_audio(text, output_path, reference_audio_path)