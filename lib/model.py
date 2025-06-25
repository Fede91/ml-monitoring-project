from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
from torch.nn import functional as F
from sklearn.metrics import accuracy_score

tokenizer = AutoTokenizer.from_pretrained("0xfedev/corporate-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("0xfedev/corporate-sentiment")

model.eval()
device = 'cpu' # torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = model.to(device)

print(f"Model loaded on {device}")

# Mapping delle label (Hugging Face specifica l'ordine)
labels = ['negative', 'neutral', 'positive']

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
  encoded_input = tokenizer(text, return_tensors='pt').to(device)

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
    X: The texts to predict the sentiment of.
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
