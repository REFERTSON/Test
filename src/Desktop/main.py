# V4
import os
from num2t4ru import num2text
import torch
import time

def get_audio():
    return model.save_wav(text=example_text,
                                 speaker=speaker,
                                 sample_rate=sample_rate)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(4)
local_file = 'Model/Silero/ru_v4.pt'


model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
model.to(device)

example_text = f'Я могу управлять умным домом!'
sample_rate = 48000
speaker = 'aidar'

start = time.time()

try:
    audio_paths = get_audio()

except:
    device = torch.device("cpu")
    model.to(device)

    audio_paths = get_audio()
print(time.time() - start)
