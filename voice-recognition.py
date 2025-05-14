import whisper
import sounddevice as sd
import soundfile as sf
import tempfile
import os
import time
import torch

# Parameters
SAMPLE_RATE = 16000
CHANNELS = 1
DURATION = 5  # seconds

# Print the CUDA and PyTorch versions
print(f"CUDA version: {torch.version.cuda}")
print(f"PyTorch version: {torch.__version__}")

# Check if CUDA is available and set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Print CUDA device information
if torch.cuda.is_available():
    print(f"CUDA is available: {torch.cuda.is_available()}")
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")


model = whisper.load_model("base").to(device)

try:
    while True:
        print("Recording for", DURATION, "seconds...")
        audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32')
        sd.wait()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            sf.write(tmpfile.name, audio, SAMPLE_RATE)
            temp_wav_path = tmpfile.name

        result = model.transcribe(temp_wav_path, language="en")
        print("Transcription:", result["text"])

        os.remove(temp_wav_path)
        time.sleep(1)  # Optional: pause before next recording
except KeyboardInterrupt:
    print("Stopped by user.")