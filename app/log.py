
import pandas as pd
import os
from datetime import datetime

LOG_PATH = "logs/inference_log.csv"

def log(texts, targets, prediction, scores):
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