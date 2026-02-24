from utils.extract import extract_clip, extract_audio, merge_audio
from utils.transcribe import transcribe_audio
from utils.translate import translate_kn_to_hi
from utils.tts import generate_hindi_audio

INPUT_VIDEO = "input.mp4"
CLIP_VIDEO = "clip.mp4"
CLIP_AUDIO = "clip.wav"
HINDI_AUDIO = "hindi.wav"
DUBBED_VIDEO = "dubbed_output.mp4"


def main():
    print("Extracting 15-second clip...")
    extract_clip(INPUT_VIDEO, CLIP_VIDEO)

    print("Extracting audio...")
    extract_audio(CLIP_VIDEO, CLIP_AUDIO)

    print("Transcribing audio...")
    transcript = transcribe_audio(CLIP_AUDIO)

    print("\n--- TRANSCRIPT ---")
    print(transcript)

    with open("transcript.txt", "w", encoding="utf-8") as f:
        f.write(transcript)

    print("Transcript saved.")

    print("Translating to Hindi...")
    hindi_text = translate_kn_to_hi(transcript)

    print("\n--- HINDI TRANSLATION ---")
    print(hindi_text)

    with open("hindi.txt", "w", encoding="utf-8") as f:
        f.write(hindi_text)

    print("Hindi text saved.")

    print("Generating Hindi speech...")
    generate_hindi_audio(hindi_text, HINDI_AUDIO)

    print("Merging Hindi audio with video...")
    merge_audio(CLIP_VIDEO, HINDI_AUDIO, DUBBED_VIDEO)

    print("\n✅ Dubbing complete!")
    print(f"Final output saved as: {DUBBED_VIDEO}")


if __name__ == "__main__":
    main()