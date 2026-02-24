import os
import subprocess
import shutil

def extract_clip(input_video, output_video, start_time="00:00:15", duration="15"):
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", start_time,
        "-i", input_video,
        "-t", duration,
        "-map", "0:v:0",
        "-map", "0:a:0",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-b:a", "192k",
        output_video
    ]
    print(f"Extracting {duration}s clip from {input_video}...")
    subprocess.run(cmd, check=True)

def extract_audio(input_video, output_audio):
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_video,
        "-ac", "1",        # mono
        "-ar", "16000",    # 16kHz
        output_audio
    ]
    print(f"Extracting raw audio to {output_audio}...")
    subprocess.run(cmd, check=True)

def separate_vocals(input_audio, vocals_out="vocals.wav", bgm_out="no_vocals.wav"):
    """
    Uses Demucs to separate background music/noise from the speaker's voice.
    This guarantees XTTS has a crystal-clear reference for voice cloning,
    and Whisper has clean audio for highly accurate transcription.
    """
    print(f"Isolating vocals from background noise using Demucs...")
    out_dir = "separated_audio"
    cmd = [
        "python", "-m", "demucs.separate",
        "-n", "htdemucs",
        "--two-stems=vocals",
        input_audio,
        "-o", out_dir
    ]
    subprocess.run(cmd, check=True)
    
    # Demucs outputs to separated_audio/htdemucs/{filename}/vocals.wav
    filename = os.path.splitext(os.path.basename(input_audio))[0]
    generated_vocals = os.path.join(out_dir, "htdemucs", filename, "vocals.wav")
    generated_bgm = os.path.join(out_dir, "htdemucs", filename, "no_vocals.wav")
    
    shutil.copy(generated_vocals, vocals_out)
    shutil.copy(generated_bgm, bgm_out)
    print(f"Vocals isolated to {vocals_out}. Background music saved to {bgm_out}.")

def mix_audio(vocals_path, bgm_path, output_path):
    """
    Mixes the generated Hindi vocals back with the original background music
    to preserve the cinematic feel of the video.
    """
    print("Mixing new Hindi vocals with original background music...")
    cmd = [
        "ffmpeg",
        "-y",
        "-i", vocals_path,
        "-i", bgm_path,
        "-filter_complex", "amix=inputs=2:duration=longest",
        output_path
    ]
    subprocess.run(cmd, check=True)

def merge_audio_to_video(video_path, new_audio_path, output_path):
    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-i", new_audio_path,
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        output_path
    ]
    subprocess.run(cmd, check=True)