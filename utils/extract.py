import subprocess
import os

def extract_clip(input_video, output_video, start_time="00:00:15", duration="15"):
    """Extracts a 15-second visual clip from the main video."""
    print(f"Extracting {duration}s clip from {input_video}...")
    cmd = [
        "ffmpeg", "-y",
        "-ss", start_time,
        "-i", input_video,
        "-t", duration,
        "-c:v", "libx264",
        "-c:a", "aac",
        output_video
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Clip saved to {output_video}")

def extract_audio(input_video, output_audio):
    print(f"Extracting audio to {output_audio}...")
    cmd = [
        "ffmpeg", "-y",
        "-i", input_video,
        "-ac", "1",
        "-ar", "16000",
        output_audio
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Audio extracted to {output_audio}")

def merge_audio(video_path, audio_path, output_path):
    print(f"Merging {audio_path} into {video_path}...")
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest", # PREVENTS the video from stretching if Hindi audio is slightly longer!
        output_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Merged video saved to {output_path}")