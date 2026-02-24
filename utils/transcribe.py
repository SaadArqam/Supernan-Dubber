from faster_whisper import WhisperModel

model = WhisperModel("medium", compute_type="float16")

def transcribe_audio(audio_path):
    segments, info = model.transcribe(audio_path)

    text = ""
    for segment in segments:
        text += segment.text + " "

    detected_language = info.language

    print(f"Detected language: {detected_language}")

    return text.strip(), detected_language