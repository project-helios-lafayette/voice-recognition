import whisper
import torch
import numpy as np
import sounddevice as sd
import queue

# Load the model
model = whisper.load_model("base")  # Use a smaller model for real-time performance

# Audio capture parameters
SAMPLE_RATE = 16000  # Whisper works best with 16kHz audio
BLOCK_SIZE = 1024  # Adjust for latency/performance balance
CHANNELS = 1

# Queue to hold incoming audio data
audio_queue = queue.Queue()


def audio_callback(indata, frames, time, status):
    """Callback function to receive live audio and put it in the queue."""
    if status:
        print(status)
    audio_queue.put(indata.copy())


# Start streaming audio from the microphone
with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32',
                    blocksize=BLOCK_SIZE, callback=audio_callback):
    print("Listening... (Press Ctrl+C to stop)")

    try:
        while True:
            # Get audio from the queue
            audio_data = audio_queue.get()
            audio_data = audio_data.flatten()  # Convert from 2D array to 1D

            # Pad/trim to ensure Whisper gets a fixed-length input
            audio_data = whisper.pad_or_trim(audio_data)

            # Convert to log-Mel spectrogram
            mel = whisper.log_mel_spectrogram(audio_data).to(model.device)

            # Transcribe the audio with the specified language
            options = whisper.DecodingOptions(language="en")  # Set the language here
            result = whisper.decode(model, mel, options)

            # Print the transcription
            print("Transcription:", result.text)

    except KeyboardInterrupt:
        print("\nStopped listening.")