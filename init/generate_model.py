import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn import functional as F
import json
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import argparse
import os

# Mapping delle label (Hugging Face specifica l'ordine)
labels = ['negative', 'neutral', 'positive']

tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

model.eval()

def load_dataset(csv_path,
                 text_column='text',
                 target_column='sentiment'):
  """
  Load the data from the csv file.
  """
  df = pd.read_csv(csv_path)
  X = df[text_column]
  y = df[target_column]

  return X, y

def predict_sentiment(text):
  """
  Predict the sentiment of the text.

  Args:
    text: The text to predict the sentiment of.

  Returns:
    The predicted sentiment and the scores.
    Scores is a list of 3 elements, one for each sentiment.
    The index of the sentiment is the same as the index of the sentiment in the labels list.
  """
  encoded_input = tokenizer(text, return_tensors='pt')

  with torch.no_grad():
    output = model(**encoded_input)
    scores = F.softmax(output.logits,
                       dim=1)
    predicted_class = torch.argmax(scores,
                                   dim=1).item()

    return labels[predicted_class], scores[0].tolist()

def run_predictions(X):
  """
  Run the predictions on the given csv file.

  Args:
    csv_path: The path to the csv file.
    text_column: The column name of the text column.
    target_column: The column name of the target column.
  """

  predictions = []

  for text in tqdm(X):
    prediction, _ = predict_sentiment(str(text))

    predictions.append(prediction)

  return predictions

def evaluate_predictions(y, predictions):
  """
  Evaluate the predictions.

  Args:
    y: The true labels.
    predictions: The predicted labels.

  Returns:
    The accuracy of the predictions.
  """
  acc = accuracy_score(y, predictions)
  return {
    'accuracy': acc
  }

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', default='./data/validation.csv', help='Path to the CSV dataset')
  parser.add_argument("--output", default="./model", help="Folder path to save the model")
  parser.add_argument("--metrics", default="./metrics/base_metrics.json", help="Path to save the metrics file")
  args = parser.parse_args()

  if not os.path.exists(args.dataset):
    print("Error: Dataset file does not exist")
    exit(1)

  X, y = load_dataset(args.dataset)

  predictions = run_predictions(X)

  metrics = evaluate_predictions(y, predictions)

  model.save_pretrained(args.model)
  tokenizer.save_pretrained(args.model)

  with open(args.metrics, 'w') as f:
    json.dump(metrics, f)

  print("Inference completed. Model saved in", args.output)
  print("Metrics saved in", args.metrics)
  print(json.dumps(metrics, indent=2))