import torch
import pickle
import soundfile as sf


def check_picklable(obj):
    try:
        pickle.dumps(obj)
        return True
    except pickle.PicklingError:
        return False
    
# Read the audio file
audio_file = 'amused_1-15_0001.wav'
audio, sample_rate = sf.read(audio_file)

# Check if the audio object can be pickled
is_picklable = check_picklable(audio)
print(is_picklable)
