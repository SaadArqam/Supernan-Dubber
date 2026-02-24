import os
from utils.extract import extract_clip, extract_audio, merge_audio
from utils.transcribe import transcribe_audio
from utils.translate import translate_to_hindi
from utils.tts import generate_hindi_audio

# Attempt to load lipsync, fallback to ffmpeg merge if Wav2Lip is not configured in Colab
try:
    from utils.lipsync import apply_wav2lip
    LIPSYNC_ENABLED = True
except ImportError:
    LIPSYNC_ENABLED = False


# Constants for the 15s extraction
INPUT_VIDEO = "/content/drive/MyDrive/Supernan-Dubber/input.mp4"
CLIP_VIDEO = "clip.mp4"
CLIP_AUDIO = "clip.wav"
HINDI_AUDIO = "hindi.wav"  # Switch to WAV for TTS voice cloning
DUBBED_VIDEO = "dubbed_output.mp4"


def process_15s_segment():
    """
    Core pipeline architecture running a specific segment.
    Designed modularly to extend to batch-chunked logic for full videos.
    """
    print("\n--- PHASE 1: EXTRACTION ---")
    if not os.path.exists(CLIP_VIDEO):
        extract_clip(INPUT_VIDEO, CLIP_VIDEO, start_time="00:00:15", duration="15")
    
    if not os.path.exists(CLIP_AUDIO):
        extract_audio(CLIP_VIDEO, CLIP_AUDIO)


    print("\n--- PHASE 2: TRANSCRIPTION ---")
    # Using faster-whisper (ensure it's installed via pip)
    transcript, detected_lang = transcribe_audio(CLIP_AUDIO)
    print(f"[{detected_lang}] Transcript: {transcript}")
    
    # Save for user debugging
    with open("transcript.txt", "w", encoding="utf-8") as f:
        f.write(transcript)


    print("\n--- PHASE 3: CONTEXT-AWARE TRANSLATION ---")
    # Using IndicTrans2 (1B model for high accuracy)
    hindi_text = translate_to_hindi(transcript, detected_lang)
    print(f"[HI] Translation: {hindi_text}")

    # Save for user debugging
    with open("hindi.txt", "w", encoding="utf-8") as f:
        f.write(hindi_text)


    print("\n--- PHASE 4: ZERO-SHOT VOICE CLONING (XTTS) ---")
    # Generates cloned audio matching the original speaker's tone
    generate_hindi_audio(hindi_text, HINDI_AUDIO, reference_audio_path=CLIP_AUDIO)


    print("\n--- PHASE 5: HIGH-FIDELITY LIP-SYNCING ---")
    if LIPSYNC_ENABLED and os.path.exists("Wav2Lip"):
        print("Wav2Lip found! Synchronizing lip movements...")
        apply_wav2lip(CLIP_VIDEO, HINDI_AUDIO, DUBBED_VIDEO)
    else:
        print("Wav2Lip not detected. Falling back to simple FFMPEG audio replacement.")
        print("NOTE: For 'The Golden 15 Seconds' assignment, please install Wav2Lip in Colab.")
        merge_audio(CLIP_VIDEO, HINDI_AUDIO, DUBBED_VIDEO)

    print("\n✅ High-Fidelity Dubbing Complete!")
    print(f"Final output saved as: {DUBBED_VIDEO}")


if __name__ == "__main__":
    process_15s_segment()