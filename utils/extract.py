import subprocess

import subprocess

def extract_clip(input_video, output_video, start_time="00:00:30", duration="15"):
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
    subprocess.run(cmd, check=True)