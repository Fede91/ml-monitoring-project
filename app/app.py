import gradio as gr
from model import predict, labels
from log import log

def handle_submit(text, target):
  label, scores = predict(text, target)

  log([text], [target], [label], [scores])

  return label, scores


app = gr.Interface(
  fn=handle_submit,
  inputs=[
    "text",
    gr.Radio(labels),
  ],
  outputs=[
    gr.Textbox(label="prediction"),
    gr.Textbox(label="scores")
  ])

app.launch()