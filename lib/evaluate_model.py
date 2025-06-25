import argparse
from model import model, tokenizer, run_predictions, evaluate_predictions
from dataset import load_dataset
from metrics import save_metrics, compare_metrics
import json


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", default="../data/validation.csv", help="Path to the CSV dataset")
  parser.add_argument('--base', default='../metrics/base_metrics.json', help='Path to the base metrics file')
  parser.add_argument("--new", default="../metrics/new_metrics.json", help="Path to the output metrics file")
  parser.add_argument("--output", default="../model", help="Folder path to save the model")
  args = parser.parse_args()

  X, y = load_dataset(args.dataset)
  predicions = run_predictions(X)

  new_metrics = evaluate_predictions(y, predicions)

  print("Evaluation completed. Metrics saved in", args.new)
  print(json.dumps(new_metrics, indent=2))

  with open(args.base, 'r') as f:
    base_metrics = json.load(f)

  if compare_metrics(base_metrics, new_metrics):
    save_metrics(new_metrics, args.new)

    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    exit(0)
  else:
    print("Model is not improved")
    exit(1)  
    