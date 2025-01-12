from flask import Flask, request, render_template
import torch
from models import build_model
from kokoro import generate
import os
import base64

app = Flask(__name__)

# Set device to CPU
device = 'cpu'

# Load the model
def load_model(path):
    try:
        # Remove weights_only argument and load model normally
        return build_model(path, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

# Load the voicepack
def load_voicepack(path):
    if os.path.exists(path):
        # Ensure map_location='cpu' for loading on CPU
        return torch.load(path, map_location='cpu')
    else:
        raise FileNotFoundError(f"Voicepack {path} not found.")

# Custom Jinja2 filter to encode data as Base64
@app.template_filter('to_base64')
def to_base64(data):
    if isinstance(data, (bytes, bytearray)):
        return base64.b64encode(data).decode('utf-8')
    return data

# Initialize the model and default voicepack
model_path = 'kokoro-v0_19.pth'
default_voicepack_path = 'voices/af.pt'
model = load_model(model_path)
voicepack = load_voicepack(default_voicepack_path)

@app.route('/', methods=['GET', 'POST'])
def index():
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
            # Convert audio to base64 for embedding
            audio_base64 = base64.b64encode(audio.tobytes()).decode('utf-8')
            return render_template(
                'index.html',
                generated_text=text,
                audio_data=audio_base64
            )
        except Exception as e:
            print(f"Error generating audio: {e}")
            return render_template('index.html', error_message="Failed to generate audio. Please try again.")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

