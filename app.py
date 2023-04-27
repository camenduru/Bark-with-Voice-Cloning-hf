import os

os.system("pip install git+https://github.com/suno-ai/bark.git")

from bark.generation import SUPPORTED_LANGS
from bark import SAMPLE_RATE, generate_audio
from scipy.io.wavfile import write as write_wav
from datetime import datetime

import shutil
import gradio as gr

import sys

import string
import time
import argparse
import json

import numpy as np
# import IPython
# from IPython.display import Audio

import torch

from TTS.tts.utils.synthesis import synthesis
from TTS.tts.utils.text.symbols import make_symbols, phonemes, symbols
try:
  from TTS.utils.audio import AudioProcessor
except:
  from TTS.utils.audio import AudioProcessor


from TTS.tts.models import setup_model
from TTS.config import load_config
from TTS.tts.models.vits import *

from TTS.tts.utils.speakers import SpeakerManager
from pydub import AudioSegment

# from google.colab import files
import librosa

from scipy.io.wavfile import write, read

import subprocess

'''
from google.colab import drive
drive.mount('/content/drive')
src_path = os.path.join(os.path.join(os.path.join(os.path.join(os.getcwd(), 'drive'), 'MyDrive'), 'Colab Notebooks'), 'best_model_latest.pth.tar')
dst_path = os.path.join(os.getcwd(), 'best_model.pth.tar')
shutil.copy(src_path, dst_path)
'''

TTS_PATH = "TTS/"

# add libraries into environment
sys.path.append(TTS_PATH) # set this if TTS is not installed globally

# Paths definition

OUT_PATH = 'out/'

# create output path
os.makedirs(OUT_PATH, exist_ok=True)

# model vars 
MODEL_PATH = 'best_model.pth.tar'
CONFIG_PATH = 'config.json'
TTS_LANGUAGES = "language_ids.json"
TTS_SPEAKERS = "speakers.json"
USE_CUDA = torch.cuda.is_available()

# load the config
C = load_config(CONFIG_PATH)

# load the audio processor
ap = AudioProcessor(**C.audio)

speaker_embedding = None

C.model_args['d_vector_file'] = TTS_SPEAKERS
C.model_args['use_speaker_encoder_as_loss'] = False

model = setup_model(C)
model.language_manager.set_language_ids_from_file(TTS_LANGUAGES)
# print(model.language_manager.num_languages, model.embedded_language_dim)
# print(model.emb_l)
cp = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
# remove speaker encoder
model_weights = cp['model'].copy()
for key in list(model_weights.keys()):
  if "speaker_encoder" in key:
    del model_weights[key]

model.load_state_dict(model_weights)

model.eval()

if USE_CUDA:
    model = model.cuda()

# synthesize voice
use_griffin_lim = False

# Paths definition

CONFIG_SE_PATH = "config_se.json"
CHECKPOINT_SE_PATH = "SE_checkpoint.pth.tar"

# Load the Speaker encoder

SE_speaker_manager = SpeakerManager(encoder_model_path=CHECKPOINT_SE_PATH, encoder_config_path=CONFIG_SE_PATH, use_cuda=USE_CUDA)

# Define helper function

def compute_spec(ref_file):
  y, sr = librosa.load(ref_file, sr=ap.sample_rate)
  spec = ap.spectrogram(y)
  spec = torch.FloatTensor(spec).unsqueeze(0)
  return spec


def voice_conversion(ta, ra, da):

  target_audio = 'target.wav'
  reference_audio = 'reference.wav'
  driving_audio = 'driving.wav'

  write(target_audio, ta[0], ta[1])
  write(reference_audio, ra[0], ra[1])
  write(driving_audio, da[0], da[1])          

  # !ffmpeg-normalize $target_audio -nt rms -t=-27 -o $target_audio -ar 16000 -f
  # !ffmpeg-normalize $reference_audio -nt rms -t=-27 -o $reference_audio -ar 16000 -f
  # !ffmpeg-normalize $driving_audio -nt rms -t=-27 -o $driving_audio -ar 16000 -f

  files = [target_audio, reference_audio, driving_audio]

  for file in files:
      subprocess.run(["ffmpeg-normalize", file, "-nt", "rms", "-t=-27", "-o", file, "-ar", "16000", "-f"])

  # ta_ = read(target_audio)

  target_emb = SE_speaker_manager.compute_d_vector_from_clip([target_audio])
  target_emb = torch.FloatTensor(target_emb).unsqueeze(0)

  driving_emb = SE_speaker_manager.compute_d_vector_from_clip([reference_audio])
  driving_emb = torch.FloatTensor(driving_emb).unsqueeze(0)

  # Convert the voice

  driving_spec = compute_spec(driving_audio)
  y_lengths = torch.tensor([driving_spec.size(-1)])
  if USE_CUDA:
      ref_wav_voc, _, _ = model.voice_conversion(driving_spec.cuda(), y_lengths.cuda(), driving_emb.cuda(), target_emb.cuda())
      ref_wav_voc = ref_wav_voc.squeeze().cpu().detach().numpy()
  else:
      ref_wav_voc, _, _ = model.voice_conversion(driving_spec, y_lengths, driving_emb, target_emb)
      ref_wav_voc = ref_wav_voc.squeeze().detach().numpy()

  # print("Reference Audio after decoder:")
  # IPython.display.display(Audio(ref_wav_voc, rate=ap.sample_rate))

  return (ap.sample_rate, ref_wav_voc)


def generate_text_to_speech(text_prompt, selected_speaker, text_temp, waveform_temp):
    audio_array = generate_audio(text_prompt, selected_speaker, text_temp, waveform_temp)

    now = datetime.now()
    date_str = now.strftime("%m-%d-%Y")
    time_str = now.strftime("%H-%M-%S")

    outputs_folder = os.path.join(os.getcwd(), "outputs")
    if not os.path.exists(outputs_folder):
        os.makedirs(outputs_folder)

    sub_folder = os.path.join(outputs_folder, date_str)
    if not os.path.exists(sub_folder):
        os.makedirs(sub_folder)

    file_name = f"audio_{time_str}.wav"
    file_path = os.path.join(sub_folder, file_name)
    write_wav(file_path, SAMPLE_RATE, audio_array)

    return file_path


speakers_list = []

for lang, code in SUPPORTED_LANGS:
    for n in range(10):
        speakers_list.append(f"{code}_speaker_{n}")

with gr.Blocks() as demo:
    gr.Markdown(
            f""" # <center>🐶🎶🥳 - Bark with Voice Cloning</center>
            
            ### <center>🤗 - Powered by [Bark](https://huggingface.co/spaces/suno/bark) and [YourTTS](https://github.com/Edresson/YourTTS). Inspired by [bark-webui](https://github.com/makawy7/bark-webui).</center>
            1. You can duplicate and use it with a GPU: <a href="https://huggingface.co/spaces/{os.getenv('SPACE_ID')}?duplicate=true"><img style="display: inline; margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate Space" /></a>
            2. First use Bark to generate audio from text and then use YourTTS to get new audio in a custom voice you like. Easy to use!
            
        """
    )
    
    with gr.Row().style(equal_height=True):
        inp1 = gr.Textbox(label="Input Text", lines=4, placeholder="Enter text here...")

        inp3 = gr.Slider(
            0.1,
            1.0,
            value=0.7,
            label="Generation Temperature",
            info="1.0 more diverse, 0.1 more conservative",
        )

        inp4 = gr.Slider(
            0.1, 1.0, value=0.7, label="Waveform Temperature", info="1.0 more diverse, 0.1 more conservative"
        )
    with gr.Row().style(equal_height=True):

        inp2 = gr.Dropdown(speakers_list, value=speakers_list[0], label="Acoustic Prompt")

        button = gr.Button("Generate using Bark")
        
        out1 = gr.Audio(label="Generated Audio")
    
    button.click(generate_text_to_speech, [inp1, inp2, inp3, inp4], [out1])
    
   
    with gr.Row().style(equal_height=True):
        inp5 = gr.Audio(label="Reference Audio for Voice Cloning")
        inp6 = out1
        inp7 = out1

        btn = gr.Button("Generate using YourTTS")
        out2 = gr.Audio(label="Generated Audio in a Custom Voice")

    btn.click(voice_conversion, [inp5, inp6, inp7], [out2])
  
    gr.Markdown(
            """ ### <center>NOTE: Please do not generate any audio that is potentially harmful to any person or organization.</center>
                        
        """
    )
    gr.Markdown(
            """ 
## 🌎 Foreign Language
Bark supports various languages out-of-the-box and automatically determines language from input text. \
When prompted with code-switched text, Bark will even attempt to employ the native accent for the respective languages in the same voice.
Try the prompt:
```
Buenos días Miguel. Tu colega piensa que tu alemán es extremadamente malo. But I suppose your english isn't terrible.
```
## 🤭 Non-Speech Sounds
Below is a list of some known non-speech sounds, but we are finding more every day. \
Please let us know if you find patterns that work particularly well on Discord!
* [laughter]
* [laughs]
* [sighs]
* [music]
* [gasps]
* [clears throat]
* — or ... for hesitations
* ♪ for song lyrics
* capitalization for emphasis of a word
* MAN/WOMAN: for bias towards speaker
Try the prompt:
```
" [clears throat] Hello, my name is Suno. And, uh — and I like pizza. [laughs] But I also have other interests such as... ♪ singing ♪."
```
## 🎶 Music
Bark can generate all types of audio, and, in principle, doesn't see a difference between speech and music. \
Sometimes Bark chooses to generate text as music, but you can help it out by adding music notes around your lyrics.
Try the prompt:
```
♪ In the jungle, the mighty jungle, the lion barks tonight ♪
```
## 🧬 Voice Cloning
Bark has the capability to fully clone voices - including tone, pitch, emotion and prosody. \
The model also attempts to preserve music, ambient noise, etc. from input audio. \
However, to mitigate misuse of this technology, we limit the audio history prompts to a limited set of Suno-provided, fully synthetic options to choose from.
## 👥 Speaker Prompts
You can provide certain speaker prompts such as NARRATOR, MAN, WOMAN, etc. \
Please note that these are not always respected, especially if a conflicting audio history prompt is given.
Try the prompt:
```
WOMAN: I would like an oatmilk latte please.
MAN: Wow, that's expensive!
```
## Details
Bark model by [Suno](https://suno.ai/), including official [code](https://github.com/suno-ai/bark) and model weights. \
Gradio demo supported by 🤗 Hugging Face. Bark is licensed under a non-commercial license: CC-BY 4.0 NC, see details on [GitHub](https://github.com/suno-ai/bark).
                        
        """
    )
    
        
    gr.HTML('''
        <div class="footer">
                    <p>🎶🖼️🎡 - It’s the intersection of technology and liberal arts that makes our hearts sing — Steve Jobs
                    </p>
        </div>
    ''')     

demo.queue().launch(show_error=True)