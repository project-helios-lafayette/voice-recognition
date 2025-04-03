import sounddevice as sd
import numpy as np
import whisper
import wave
import io
import torch

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

model = whisper.load_model("small").to(device)

# Audio settings
CHUNK = 1024
FORMAT = 'int16'  # 16 bit audio
CHANNELS = 1
RATE = 44100
DURATION = 4  # seconds

# List available audio devices
print("Available audio devices:")
print(sd.query_devices())

# Select the desired device by index
device_index = int(input("Select device index: "))

def record_and_transcribe():
    print("Recording...")
    # Record audio
    audio_data = sd.rec(int(RATE * DURATION), samplerate=RATE, channels=CHANNELS, dtype=FORMAT, device=device_index)
    sd.wait()  # Wait until recording is finished

    with io.BytesIO() as wav_buffer:
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(RATE)
            wf.writeframes(audio_data.tobytes())

        wav_buffer.seek(0)
        # Read the audio data from the buffer and convert it to a NumPy array
        wav_buffer.seek(0)
        audio_data = np.frombuffer(wav_buffer.read(), dtype=np.int16)

    # Convert audio data to float32
    audio_data = audio_data.astype(np.float32) / 32768.0

    # Transcribe the audio with Whisper
    print("Transcribing...")
    result = model.transcribe(audio=audio_data)

    print("Transcription:")
    print(result['text'])

# Call the function to record and transcribe
while True:
    record_and_transcribe()