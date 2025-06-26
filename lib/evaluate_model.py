import argparse
import json
from model import run_predictions, evaluate_predictions
from dataset import load_dataset, save_dataset
from metrics import compare_metrics
from datasets import load_dataset, Dataset


if __name__ == "__main__":
  # Parse the arguments
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", default="0xfedev/corporate-sentiment-logs", help="Dataset to evaluate")
  parser.add_argument("--train_dataset", default="../data/train.csv", help="Train dataset to override")
  parser.add_argument('--metrics', default='../metrics/base_metrics.json', help='Path to the base metrics file')
  args = parser.parse_args()

  # Load the dataset
  ds = load_dataset(args.dataset, split="train")

  # Get the text and target data
  X, y = ds['text'], ds['target']

  # Run the predictions
  predicions = run_predictions(X)

  # Evaluate the predictions
  evaluation_metrics = evaluate_predictions(y, predicions)

  print(f"Evaluation metrics: {evaluation_metrics}")

  with open(args.metrics, 'r') as f:
    base_metrics = json.load(f)
  
  # Compare the metrics
  if compare_metrics(base_metrics, evaluation_metrics):
    # If the metrics are better, save the dataset
    save_dataset(X, y, args.train_dataset)

    # Exit with success
    exit(0)
  else:
    # If the metrics are not better, exit with failure
    exit(1)
