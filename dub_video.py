from utils.extract import extract_clip, extract_audio

INPUT_VIDEO = "input.mp4"
CLIP_VIDEO = "clip.mp4"
CLIP_AUDIO = "clip.wav"

def main():
    print("Extracting 15-second clip...")
    extract_clip(INPUT_VIDEO, CLIP_VIDEO)

    print("Extracting audio...")
    extract_audio(CLIP_VIDEO, CLIP_AUDIO)

    print("Done.")

if __name__ == "__main__":
    main()