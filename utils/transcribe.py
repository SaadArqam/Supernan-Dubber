from faster_whisper import WhisperModel

def transcribe_audio(audio_path):
    """
    Transcribes the audio using faster-whisper.
    Uses large-v3-turbo for incredibly fast but highly accurate transcription.
    Automatically detects the spoken language.
    """
    print("Loading ASR Model: faster-whisper (large-v3-turbo)...")
    
    # We use large-v3-turbo because it bridges the speed of 'medium' with the accuracy of 'large'
    model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
    
    print("Transcribing and detecting language...")
    segments, info = model.transcribe(audio_path, beam_size=5)
    
    # Join the generator segments into a single cohesive string
    text = " ".join([segment.text for segment in segments])
    detected_lang = info.language
    
    print(f"✅ Transcription complete. Detected Source Language: '{detected_lang}'")
    
    return text.strip(), detected_lang