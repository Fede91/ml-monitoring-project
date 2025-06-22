import gradio as gr
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import os

LOG_PATH = "logs/inference_log.csv"

tokenizer = AutoTokenizer.from_pretrained("0xfedev/corporate-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("0xfedev/corporate-sentiment")

model.eval()

labels = ['negative', 'neutral', 'positive']

def log(texts, targets, predictions):
  df = pd.DataFrame({
      "text": texts,
      "target": targets,
      "prediction": predictions
  })
  os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
  if os.path.exists(LOG_PATH):
      df.to_csv(LOG_PATH, mode="a", header=False, index=False)
  else:
      df.to_csv(LOG_PATH, index=False)

def predict(text, target):
  encoded_input = tokenizer(text, return_tensors='pt')

  with torch.no_grad():
    output = model(**encoded_input)
    scores = F.softmax(output.logits,
                       dim=1)
    predicted_class = torch.argmax(scores,
                                   dim=1).item()
    
    predicted_label = labels[predicted_class]

    log([text], [target], [predicted_label])

    return labels[predicted_class], scores[0].tolist()

app = gr.Interface(
  fn=predict,
  inputs=[
    "text",
    gr.Radio(labels),
  ],
  outputs=[
    gr.Textbox(label="prediction"),
    gr.Textbox(label="scores")
  ])

app.launch()