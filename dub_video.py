import os
from utils.extract import extract_clip, extract_audio, merge_audio
from utils.transcribe import transcribe_audio
from utils.analyze import detect_gender
from utils.translate import translate_to_hindi
from utils.tts import generate_hindi_audio

def run_dubbing_pipeline():
    print("\n" + "="*60)
    print("🎬 STARTING PRODUCTION-GRADE DUBBING PIPELINE 🎬")
    print("="*60)

    # Intelligent path detection:
    # 1. First check if input.mp4 exists in the current Colab root directory
    # 2. If not, fallback to the Google Drive folder
    if os.path.exists("input.mp4"):
        INPUT_VIDEO = "input.mp4"
        print("📁 Found 'input.mp4' in root directory.")
    elif os.path.exists("/content/drive/MyDrive/Supernan-Dubber/input.mp4"):
        INPUT_VIDEO = "/content/drive/MyDrive/Supernan-Dubber/input.mp4"
        print("📁 Found 'input.mp4' in Google Drive.")
    else:
        print(f"❌ Error: Source video not found in root or Google Drive!")
        print("Please upload your video as 'input.mp4' in Google Colab or Google Drive.")
        return

    CLIP_VIDEO = "clip.mp4"
    CLIP_AUDIO = "clip.wav"
    HINDI_AUDIO = "hindi.wav"
    OUTPUT_VIDEO = "dubbed_output.mp4"

    print("\n[PHASE 1] Video & Audio Extraction")
    extract_clip(INPUT_VIDEO, CLIP_VIDEO, start_time="00:00:15", duration="15")
    extract_audio(CLIP_VIDEO, CLIP_AUDIO)

    print("\n[PHASE 1.5] Speaker Analysis (Gender Detection)")
    speaker_gender = detect_gender(CLIP_AUDIO)

    print("\n[PHASE 2] High-Accuracy Transcription (Whisper large-v3-turbo)")
    transcript, detected_lang = transcribe_audio(CLIP_AUDIO)
    print(f"Transcript ({detected_lang}): {transcript}")

    print("\n[PHASE 3] Multilingual-to-Hindi Translation (NLLB-1.3B)")
    hindi_text = translate_to_hindi(transcript, detected_lang)
    print(f"Hindi: {hindi_text}")

    print("\n[PHASE 4] Text-to-Speech Generation (Edge-TTS)")
    generate_hindi_audio(hindi_text, HINDI_AUDIO, gender=speaker_gender)

    print("\n[PHASE 5] Video & Audio Merging")
    merge_audio(CLIP_VIDEO, HINDI_AUDIO, OUTPUT_VIDEO)

    print("\n" + "="*60)
    print(f"🎉 PIPELINE COMPLETE! Download your mapped video: {OUTPUT_VIDEO}")
    print("="*60)

if __name__ == "__main__":
    run_dubbing_pipeline()