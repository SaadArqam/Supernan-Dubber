import subprocess

def extract_clip(input_video, output_video, start_time="00:00:15", end_time="00:00:30"):
    cmd = [
        "ffmpeg",
        "-i", input_video,
        "-ss", start_time,
        "-to", end_time,
        "-c", "copy",
        output_video
    ]
    subprocess.run(cmd, check=True)


def extract_audio(input_video, output_audio):
    cmd = [
        "ffmpeg",
        "-i", input_video,
        "-q:a", "0",
        "-map", "a",
        output_audio
    ]
    subprocess.run(cmd, check=True)