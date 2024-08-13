import numpy as np
from first import pipe
import gradio as gr
import audio

def transcribe(audio):

  print("*"*100)
  sr, y = audio

  y = y.astype(np.float32)
  y /= np.max(np.abs(y))

  result = pipe({"raw":y, "sampling_rate":sr},generate_kwargs={"language": "english"})

  return result["text"]




with gr.Blocks() as demo:

  with gr.Row():
    with gr.Column():
      audio = gr.Audio(sources = ["microphone"], format="wav", show_download_button=True)
      generated_transcript = gr.Textbox(label="Generated Transcript", interactive = True)


  audio.change(transcribe, inputs=audio, outputs=generated_transcript)

demo.launch(share=True, debug=True)