import os
import subprocess

def apply_wav2lip(video_path: str, audio_path: str, output_path: str):
    wav2lip_dir = "Wav2Lip"
    checkpoint_path = "Wav2Lip/checkpoints/wav2lip_gan.pth"
    
    if not os.path.exists(wav2lip_dir):
        raise FileNotFoundError(
            "Wav2Lip repository not found. Please run: "
            "!git clone https://github.com/Rudrabha/Wav2Lip.git in Colab."
        )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            "Wav2Lip checkpoint not found. Please download "
            "wav2lip_gan.pth and place it in Wav2Lip/checkpoints/."
        )

    print("Running Wav2Lip high-fidelity inference. This may take a few minutes...")

    command = [
        "python", f"{wav2lip_dir}/inference.py",
        "--checkpoint_path", checkpoint_path,
        "--face", video_path,
        "--audio", audio_path,
        "--outfile", output_path,
        "--pads", "0", "10", "0", "0"
    ]

    try:
        subprocess.run(command, check=True)
        print(f"Lip-syncing complete. Final video saved at: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Wav2Lip Failed: {e}")
        raise
