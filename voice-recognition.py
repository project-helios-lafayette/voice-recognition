import sounddevice as sd
import numpy as np
import whisper
import torch
import queue

# Check device availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load a faster Whisper model
model = whisper.load_model("base").to(device)  # Try "tiny" for even faster results

# Audio settings
CHUNK = 1024  # Small chunk size for low latency
FORMAT = np.int16
CHANNELS = 1
RATE = 44100

# Queue to store incoming audio chunks
audio_queue = queue.Queue()


# Callback function to continuously stream audio
def audio_callback(indata, frames, time, status):
    if status:
        print(status, flush=True)
    audio_queue.put(indata.copy())


# Function to process audio from queue and transcribe in real time
def transcribe_stream():
    print("Listening...")
    buffer = np.array([], dtype=np.float32)

    with sd.InputStream(samplerate=RATE, channels=CHANNELS, dtype="int16", callback=audio_callback):
        while True:
            try:
                # Retrieve and normalize new audio chunk
                chunk = audio_queue.get()
                chunk = chunk.astype(np.float32) / 32768.0  # Convert to float32

                # Append to buffer
                buffer = np.concatenate((buffer, chunk.flatten()))

                # Transcribe every 3 seconds of audio
                if len(buffer) >= RATE * 3:
                    print("Transcribing...")
                    result = model.transcribe(buffer, fp16=False)  # Disable fp16 if using CPU
                    print("Transcription:", result['text'])

                    # Clear buffer after processing
                    buffer = np.array([], dtype=np.float32)

            except KeyboardInterrupt:
                print("Stopping...")
                break


# Start real-time transcription
transcribe_stream()