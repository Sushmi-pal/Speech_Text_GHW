from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
from scipy.io import wavfile
import numpy as np
import torch


class SpeechRecognitionPipeline:
    def __init__(self,
                 model_id="openai/whisper-large-v3"):

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_id)

        self.pipeline = None

        self.generate_kwargs = {"language": "english"}

    def create_pipeline(self):

        self.pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

    def generate_transcript(self, audio, sr=None, path=False):

        if path:
            transcribed_text = self.pipeline(audio, generate_kwargs=self.generate_kwargs)

        else:
            transcribed_text = self.pipeline({"raw": audio, "sampling_rate": sr}, generate_kwargs=self.generate_kwargs)

        return transcribed_text["text"]


if __name__ == "__main__":
    asr_pipe = SpeechRecognitionPipeline()
    asr_pipe.create_pipeline()

    sr, audio_array = wavfile.read('data/en_speaker_0.wav')

    audio_array = audio_array.astype(np.float32)
    audio_array /= np.max(np.abs(audio_array))

    result = asr_pipe.generate_transcript(
        audio=audio_array,
        sr=sr
    )

    print("Text : ", result)
