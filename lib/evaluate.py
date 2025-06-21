import argparse
from inference import load_dataset, run_predictions, evaluate_predictions
from metrics import save_metrics
import json


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", default="../data/data.csv", help="Path to the CSV dataset")
  parser.add_argument("--output", default="../metrics/new_metrics.json", help="Path to the output metrics file")
  args = parser.parse_args()

  X, y = load_dataset(args.dataset)
  predicions = run_predictions(X)

  metrics = evaluate_predictions(y, predicions)

  save_metrics(metrics, args.output)

  print("Evaluation completed. Metrics saved in", args.output)
  print(json.dumps(metrics, indent=2))