from flask import Flask, request, render_template, url_for
import torch
from models import build_model
from kokoro import generate
import os
import soundfile as sf
import numpy as np
import time

app = Flask(__name__)

# Set device to CPU
device = 'cpu'

# Load the model
def load_model(path):
    try:
        return build_model(path, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

# Load the voicepack
def load_voicepack(path):
    if os.path.exists(path):
        return torch.load(path, map_location='cpu')
    else:
        raise FileNotFoundError(f"Voicepack {path} not found.")

# Initialize the model and default voicepack
model_path = 'kokoro-v0_19.pth'
default_voicepack_path = 'voices/af.pt'
model = load_model(model_path)
voicepack = load_voicepack(default_voicepack_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    # Initialize variables to update the UI dynamically
    generated_text = None
    error_message = None
    audio_file_url = None
    selected_voice = None

    if request.method == 'POST':
        text = request.form['text_input']
        selected_voice = request.form['voice_select']

        # Load the selected voicepack
        voicepack_path = f'voices/{selected_voice}.pt'
        try:
            voicepack = load_voicepack(voicepack_path)
        except FileNotFoundError:
            voicepack = load_voicepack(default_voicepack_path)

        # Generate audio
        try:
            audio, phonemes = generate(model, text, voicepack)

            # Debugging the generated audio
            print(f"Audio Type: {type(audio)}")
            print(f"Audio Min: {np.min(audio) if isinstance(audio, np.ndarray) else 'Unknown'}")
            print(f"Audio Max: {np.max(audio) if isinstance(audio, np.ndarray) else 'Unknown'}")
            print(f"Audio Length: {len(audio) if hasattr(audio, '__len__') else 'Unknown'}")

            # Ensure the audio is valid and convert to NumPy array
            if isinstance(audio, torch.Tensor):
                audio_np = audio.cpu().numpy()
            elif isinstance(audio, np.ndarray):
                audio_np = audio
            else:
                raise ValueError("Invalid audio format returned by generate function")

            # Normalize audio to 16-bit range (-32768 to 32767)
            audio_np = np.clip(audio_np, -1.0, 1.0)  # Ensure values are between -1 and 1
            audio_np = (audio_np * 32767).astype(np.int16)

            # Save as a .wav file using soundfile
            wav_path = "static/generated_audio.wav"
            sf.write(wav_path, audio_np, samplerate=22050, subtype='PCM_16')

            # Add a timestamp to force the browser to fetch the latest file
            audio_file_url = url_for('static', filename='generated_audio.wav', t=time.time())

            # Update the generated text to match the latest submission
            generated_text = text

        except Exception as e:
            print(f"Error generating audio: {e}")
            error_message = "Failed to generate audio. Please try again."

    return render_template(
        'index.html',
        selected_voice=selected_voice,
        generated_text=generated_text,
        audio_file_url=audio_file_url,
        error_message=error_message
    )

if __name__ == '__main__':
    # Ensure the static directory exists for the .wav file
    if not os.path.exists("static"):
        os.makedirs("static")

    app.run(debug=True)
