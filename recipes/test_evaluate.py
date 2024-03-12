import torchaudio
from speechbrain.inference import interfaces

from speechbrain.inference.interfaces import foreign_class

def speech_sentiment(file, normalise=True):
    """
    Return the valence and arousal of the audio file.

    Args:
        file (str): Path to the audio file. Make sure the sample rate is fs=16000 Hz.
        normalise (bool): Whether to normalize the valence and arousal values to [-1, 1].

    Returns:
        valence (float): Valence of the audio file.
        arousal (float): Arousal of the audio file.
    """
    classifier = foreign_class(source="Speechbrain_Regression/recipes/huggingface_repo", 
                               pymodule_file="custom.py", 
                               classname="CustomEncoderWav2vec2Classifier",
                               savedir="Speechbrain_Regression/recipes/huggingface_repo")

    res = classifier.classify_file(file)
    valence, arousal = res[0][0].item(), res[0][1].item()
    # res is still a tensor, extract valence and arousal

    if normalise:
        # Normalise values from [1, 5] to [-1, 1]
        valence = 2 * (valence - 1) / (5 - 1) - 1
        arousal = 2 * (arousal - 1) / (5 - 1) - 1

    return valence, arousal
