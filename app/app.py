import gradio as gr
from model import predict, labels
from log import log

def handle_submit(text, target):
  """
  This function is called when the user submits the form.
  It takes the text and target sentiment from the form and calls the predict function.
  It then logs the text, target sentiment, predicted sentiment and scores to the log file.
  It returns the predicted sentiment and scores to the user.

  Args:
    text: The text to analyze.
    target: The target sentiment.

  Returns:
    The predicted sentiment and scores.
  """
  label, scores = predict(text, target)

  log([text], [target], [label], [scores])

  return label, scores


app = gr.Interface(
  fn=handle_submit,
  title="Corporate Sentiment Monitoring",
  description="Analyze corporate sentiment",
  inputs=[
    gr.Textbox(label="Text to analyze", placeholder="This company is doing great!", info="Enter the text to analyze"),
    gr.Radio(labels, label="Target sentiment", info="Select the target sentiment", value="positive"),
  ],
  outputs=[
    gr.Textbox(label="prediction"),
    gr.Textbox(label="scores")
  ])

app.launch()