from TTS.api import TTS

def generate_hindi_audio(text, output_path):
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
    tts.tts_to_file(text=text, file_path=output_path)