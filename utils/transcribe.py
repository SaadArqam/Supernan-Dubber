from faster_whisper import WhisperModel


def transcribe_audio(audio_path):
    model = WhisperModel("medium", compute_type="float32")

    segments, info = model.transcribe(
    audio_path,
    task="transcribe",
    vad_filter=False
)

    print("Detected language:", info.language)

    full_text = ""
    for segment in segments:
        print(f"[{segment.start:.2f}s → {segment.end:.2f}s] {segment.text}")
        full_text += segment.text + " "

    return full_text.strip()