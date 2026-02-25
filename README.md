# Supernan-Dubber: High-Fidelity Multilingual Video Dubbing Pipeline

This repository contains a modular, production-ready video dubbing pipeline. It extracts audio from a target video, automatically detects the source language, provides near-perfect transcription, strictly translates the script to Devanagari Hindi, detects the speaker's gender dynamically, and synthesizes natural Hindi voice audio mapped perfectly to the source video.

Built specifically for stability, high accuracy, and zero-conflict Google Colab execution.

## 🚀 Architecture overview

1. **Phase 1 (Extraction):** `ffmpeg` strictly targets 15 seconds and isolates 16kHz mono audio.
2. **Phase 1.5 (Analysis):** `librosa.pyin` extracts the fundamental harmonic frequencies to dynamically classify the speaker's vocal pitch (gender).
3. **Phase 2 (ASR):** `faster-whisper` (`large-v3-turbo`) transcribes the speech to text with sub-second accuracy and automatically detects the input language.
4. **Phase 3 (Translation):** `facebook/nllb-200-1.3B` forces the tokenizer's output to strictly `hin_Deva` tokens, guaranteeing zero English hallucinations.
5. **Phase 4 (TTS):** Microsoft's Serverless `edge-tts` engine natively routes the text to either a hyper-realistic female (`Swara`) or male (`Madhur`) neural voice based on the Phase 1.5 pitch analysis.
6. **Phase 5 (Merge):** `ffmpeg` `-shortest` flag cleanly merges the output back into the video, strictly capping the duration to avoid frame stretching.

## 🛠 Setup & Installation

This pipeline is optimized for **Google Colab (T4 GPU)**. 

### 1. Clone the Repository
```bash
git clone https://github.com/SaadArqam/Supernan-Dubber.git
cd Supernan-Dubber
```

### 2. Install Dependencies
```bash
pip install faster-whisper transformers sentencepiece accelerate scipy edge-tts librosa
```
*Note: We strictly avoid unstable libraries like `coqui-tts` to prevent dependency conflicts (e.g., numpy/ffmpeg overriding).*

### 3. Run the Pipeline
Ensure your source video is named `input.mp4` and placed in the root folder, then execute:
```bash
python dub_video.py
```
The final rendered file will be saved as `dubbed_output.mp4`.

## 💸 Cost Analysis (At Scale)

If deployed onto AWS / GCP cloud architecture using the exact same hardware profile (NVIDIA T4 16GB VRAM instance - e.g., AWS `g4dn.xlarge` @ ~$0.52/hour):

*   **Extraction & VITS Generation:** Uses negligible CPU time.
*   **Whisper large-v3-turbo & NLLB-1.3B Inference Processing Speed:** ~4x Real-Time (Processes 1 minute of video in ~15 seconds of GPU compute).
*   **Estimated Cloud Cost per minute of video:** ~$0.0021
*   **Cost for 1 Hour of video dubbing:** **~$0.13** 

This is incredibly cost-effective because the translation and TTS modules run entirely via local/free-tier LLM endpoints rather than relying on paid APIs like OpenAI or ElevenLabs.

## 🚨 Known Limitations

1. **Lip Syncing Disconnect:** Because Hindi translations require typically 20-30% more syllables to phrase conversational sentences than English, the spoken audio no longer matches the visual lip movements frame-by-frame.
2. **Singular Voice Engine:** Pitch analysis works wonderfully for classifying a single prominent speaker. However, if multiple people talk over each other (diarization), `librosa` pitch tracing will average the frequencies, and the pipeline will synthesize all voices using a single avatar.
3. **Background Music Loss:** Because `extract_audio` currently isolates down to 1 channel (16kHz mono), background music and environmental Foley sounds are stripped from the final dubbed video. 

## 🚀 Future Improvements (With more time)

1. **Vocal Isolation & Cinematic Mixing (Demucs):** Implement `demucs` to separate the speaker's vocal stem from the background music stem. The new Hindi dialogue could be overlaid back onto the original background music track, preserving the emotional ambiance of the video.
2. **Speaker Diarization (PyAnnote):** Implement `pyannote.audio` to identify speaker turnover ("Speaker A", "Speaker B"). The system could then dynamically prompt English-to-Hindi multi-character scripts and assign independent unique Neural TTS avatars to each speaker.
3. **High-Fidelity Lip-Sync Re-mapping (Wav2Lip):** Instead of using `-shortest` to truncate audio, the pipeline should aggressively time-stretch the audio using `ffmpeg atempo`, then feed the entire video and new audio array into `Wav2Lip` to physically warp the speaker's mouth geometry in the video to match the new Hindi phonemes.
