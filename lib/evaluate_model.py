import argparse
import json
from model import run_predictions, evaluate_predictions
from dataset import load_dataset
from metrics import compare_metrics
from datasets import load_dataset, Dataset


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", default="0xfedev/corporate-sentiment-logs", help="Dataset to evaluate")
  parser.add_argument('--metrics', default='../metrics/base_metrics.json', help='Path to the base metrics file')
  args = parser.parse_args()

  ds = load_dataset(args.dataset, split="train")

  X, y = ds['text'], ds['target']

  predicions = run_predictions(X)

  evaluation_metrics = evaluate_predictions(y, predicions)

  print(f"Evaluation metrics: {evaluation_metrics}")

  with open(args.metrics, 'r') as f:
    base_metrics = json.load(f)

  success_code = 0 if compare_metrics(base_metrics, evaluation_metrics) else 1

  exit(success_code)
