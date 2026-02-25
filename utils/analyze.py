import librosa
import numpy as np
import warnings

def detect_gender(audio_path):
    print("Analyzing audio frequencies to detect speaker gender...")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = librosa.load(audio_path, sr=None)
        

        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=300)
        

        valid_f0 = f0[~np.isnan(f0)]
        
        if len(valid_f0) == 0:
            print("Could not clearly detect pitch. Defaulting to 'male'.")
            return "male"
            
        median_pitch = np.median(valid_f0)
        print(f"Detected Median Pitch: {median_pitch:.1f} Hz")
        

        if median_pitch > 165.0:
            print("Detected Female Speaker.")
            return "female"
        else:
            print("Detected Male Speaker.")
            return "male"
            
    except Exception as e:
        print(f"Warning: Gender detection failed ({str(e)}). Defaulting to 'male'.")
        return "male"
