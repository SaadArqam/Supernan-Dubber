from gtts import gTTS

def generate_hindi_audio(text, output_path):
    print("Generating Hindi audio with gTTS...")
    tts = gTTS(text=text, lang="hi")
    tts.save(output_path)
    print("Hindi audio saved.")