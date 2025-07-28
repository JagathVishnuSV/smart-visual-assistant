import whisper
import sounddevice as sd
import numpy as np
import tempfile
import scipy.io.wavfile

model = whisper.load_model("small")

def record_audio(duration=5,samplerate=16000):
    print("Listening...")
    audio = sd.rec(int(duration * samplerate),samplerate=samplerate,channels=1)
    sd.wait()  # Wait until recording is finished
    return np.squeeze(audio)  # Remove extra dimensions

def transcribe_audio():
    audio = record_audio()
    with tempfile.NamedTemporaryFile(suffix=".wav",delete=False) as f:
        scipy.io.wavfile.write(f.name,16000,audio)
        result = model.transcribe(f.name)
    return result["text"]