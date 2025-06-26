
import pandas as pd
import os
from datetime import datetime
from datasets import Dataset

LOG_PATH = "logs/inference_log.csv"

def export_logs():
  """
  This function reads the log file and pushes it to the Hugging Face dataset.
  """
  df = pd.read_csv(LOG_PATH)

  ds = Dataset.from_pandas(df)
  ds.push_to_hub("0xfedev/corporate-sentiment-logs", token=os.getenv('HF_TOKEN'))

def log(texts, targets, prediction, scores):
  """
  This function logs the text, target sentiment, predicted sentiment and scores to the log file.
  It also call the export_logs function to push the log file to the Hugging Face dataset.

  Args:
    texts: The text to analyze.
    targets: The target sentiment.
    prediction: The predicted sentiment.
  """
  df = pd.DataFrame({
      "text": texts,
      "target": targets,
      "prediction": prediction,
      "scores": scores,
      "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  })
  
  os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

  if os.path.exists(LOG_PATH):
    df.to_csv(LOG_PATH, mode="a", header=False, index=False)
  else:
    df.to_csv(LOG_PATH, index=False)
  
  export_logs()
  
