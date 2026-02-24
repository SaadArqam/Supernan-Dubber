import os
from utils.extract import extract_clip, extract_audio, separate_vocals, mix_audio, merge_audio_to_video, match_audio_duration
from utils.transcribe import transcribe_audio
from utils.translate import translate_to_hindi
from utils.tts import generate_hindi_audio

# Attempt to load lipsync
try:
    from utils.lipsync import apply_wav2lip
    LIPSYNC_ENABLED = True
except ImportError:
    LIPSYNC_ENABLED = False


# Constants for the 15s High-Fidelity Pipeline
INPUT_VIDEO = "/content/drive/MyDrive/Supernan-Dubber/input.mp4"
CLIP_VIDEO = "clip.mp4"
CLIP_AUDIO = "clip.wav"

VOCALS_WAV = "vocals.wav"
BGM_WAV = "no_vocals.wav"

HINDI_VOCALS = "hindi_vocals.wav"
HINDI_VOCALS_STRETCHED = "hindi_vocals_stretched.wav"
FINAL_MIXED_AUDIO = "final_hindi_mixed.wav"
DUBBED_VIDEO = "dubbed_output.mp4"

def process_15s_segment():
    """
    "The Golden 15 Seconds" Pipeline Orchestrator
    """
    print("\n" + "="*50)
    print("🎬 INITIATING HIGH-FIDELITY DUBBING PIPELINE 🎬")
    print("="*50)

    print("\n--- PHASE 1: EXTRACTION & VOCAL SEPARATION (Demucs) ---")
    if not os.path.exists(CLIP_VIDEO):
        extract_clip(INPUT_VIDEO, CLIP_VIDEO, start_time="00:00:15", duration="15")
    
    if not os.path.exists(CLIP_AUDIO):
        extract_audio(CLIP_VIDEO, CLIP_AUDIO)

    if not os.path.exists(VOCALS_WAV) or not os.path.exists(BGM_WAV):
        separate_vocals(CLIP_AUDIO, vocals_out=VOCALS_WAV, bgm_out=BGM_WAV)


    print("\n--- PHASE 2: CLEAN TRANSCRIPTION ---")
    # Using faster-whisper exclusively on the isolated vocals so BGM doesn't confuse it
    transcript, detected_lang = transcribe_audio(VOCALS_WAV)
    print(f"[{detected_lang}] Transcript: {transcript}")
    
    with open("transcript.txt", "w", encoding="utf-8") as f:
        f.write(transcript)


    print("\n--- PHASE 3: CONTEXT-AWARE LLM TRANSLATION ---")
    # Using Qwen2.5-1.5B-Instruct for conversational, natural Hindi dubbing
    hindi_text = translate_to_hindi(transcript, detected_lang)
    print(f"[HI] Translation: {hindi_text}")

    with open("hindi.txt", "w", encoding="utf-8") as f:
        f.write(hindi_text)


    print("\n--- PHASE 4: ZERO-SHOT VOICE CLONING (XTTS) ---")
    # Clone voice using a strict 5-6 second sample of pristine isolated vocals.
    # Passing the full 15s CLIP_AUDIO overloads XTTS and causes horrible noisy/robotic artifacts.
    CLONE_REF = "clone_ref.wav"
    import subprocess
    subprocess.run(["ffmpeg", "-y", "-i", VOCALS_WAV, "-t", "6", "-loglevel", "error", CLONE_REF])
    
    generate_hindi_audio(hindi_text, HINDI_VOCALS, reference_audio_path=CLONE_REF)

    print("\n--- PHASE 4.5: TIME-STRETCHING AUDIO TO FIT 15 SECONDS ---")
    # Shrink or stretch the Hindi speech to fit perfectly inside the 15s window
    match_audio_duration(HINDI_VOCALS, target_duration=15.0, output_audio=HINDI_VOCALS_STRETCHED)

    print("\n--- PHASE 5: CINEMATIC MIXING ---")
    # Mix the stretched Hindi voice clone back into the original Background Music
    if os.path.exists(BGM_WAV):
        mix_audio(HINDI_VOCALS_STRETCHED, BGM_WAV, FINAL_MIXED_AUDIO)
    else:
        # Fallback if bgm extraction failed
        import shutil
        shutil.copy(HINDI_VOCALS_STRETCHED, FINAL_MIXED_AUDIO)


    print("\n--- PHASE 6: HIGH-FIDELITY LIP-SYNCING ---")
    if LIPSYNC_ENABLED and os.path.exists("Wav2Lip"):
        print("Wav2Lip found! Synchronizing lip movements with Hindi audio...")
        apply_wav2lip(CLIP_VIDEO, FINAL_MIXED_AUDIO, DUBBED_VIDEO)
    else:
        print("Wav2Lip not detected. Falling back to simple FFMPEG audio replacement.")
        print("NOTE: For 'The Golden 15 Seconds' assignment, please install Wav2Lip in Colab.")
        merge_audio_to_video(CLIP_VIDEO, FINAL_MIXED_AUDIO, DUBBED_VIDEO)

    print("\n✅ GOLDEN DUBBING COMPLETE!")
    print(f"Final output saved as: {DUBBED_VIDEO}")


if __name__ == "__main__":
    process_15s_segment()