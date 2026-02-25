from faster_whisper import WhisperModel

def transcribe_audio(audio_path):
    print("Loading ASR Model: faster-whisper (large-v3-turbo)...")
    model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
    
    print("Transcribing and detecting language...")
    segments, info = model.transcribe(audio_path, beam_size=5)
    
    text = " ".join([segment.text for segment in segments])
    detected_lang = info.language
    
    print(f"Transcription complete. Detected Source Language: '{detected_lang}'")
    
    return text.strip(), detected_lang