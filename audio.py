from first import pipe

from IPython.display import Audio, display

display(Audio("/content/drive/MyDrive/upacare/bark/en_speaker_0.wav"))

from scipy.io import wavfile
sr, audio_array = wavfile.read('/content/drive/MyDrive/upacare/bark/en_speaker_0.wav')

result = pipe({"raw":audio_array, "sampling_rate":sr},generate_kwargs={"language": "english"})
print(result["text"])