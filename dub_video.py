from utils.extract import extract_clip, extract_audio
from utils.transcribe import transcribe_audio
from utils.translate import translate_kn_to_hi

INPUT_VIDEO = "input.mp4"
CLIP_VIDEO = "clip.mp4"
CLIP_AUDIO = "clip.wav"

def main():
    print("Extracting 15-second clip...")
    extract_clip(INPUT_VIDEO, CLIP_VIDEO)

    print("Extracting audio...")
    extract_audio(CLIP_VIDEO, CLIP_AUDIO)

    print("Transcribing audio...")
    transcript = transcribe_audio(CLIP_AUDIO)

    print("\n--- TRANSCRIPT ---")
    print(transcript)

    with open("transcript.txt", "w") as f:
        f.write(transcript)

    print("Transcript saved.")
    print("Translating to Hindi...")
    hindi_text = translate_kn_to_hi(transcript)

    print("\n--- HINDI TRANSLATION ---")
    print(hindi_text)

    with open("hindi.txt", "w") as f:
        f.write(hindi_text)

if __name__ == "__main__":
    main()