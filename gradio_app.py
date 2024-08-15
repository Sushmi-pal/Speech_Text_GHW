import numpy as np
import gradio_app as gr

from asrservice.model import SpeechRecognitionPipeline


asr_pipe = SpeechRecognitionPipeline()
asr_pipe.create_pipeline()


def transcribe(audio):

  print("*"*100)
  sr, y = audio

  y = y.astype(np.float32)
  y /= np.max(np.abs(y))

  result = asr_pipe.generate_transcript(
    audio = y,
    sr = sr
  )

  return result



if __name__ == "__main__":
  with gr.Blocks() as demo:

    with gr.Row():
      with gr.Column():
        audio = gr.Audio(sources = ["microphone"], format="wav", show_download_button=True)
        generated_transcript = gr.Textbox(label="Generated Transcript", interactive = True)


    audio.change(transcribe, inputs=audio, outputs=generated_transcript)

  demo.launch(share=True, debug=True)