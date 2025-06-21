import argparse
from inference import model, tokenizer, load_dataset, run_predictions, evaluate_predictions
from metrics import save_metrics, compare_metrics
import json


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", default="../data/data.csv", help="Path to the CSV dataset")
  parser.add_argument('--base', default='../metrics/base_metrics.json', help='Path to the base metrics file')
  parser.add_argument("--new", default="../metrics/new_metrics.json", help="Path to the output metrics file")
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

    model.save_pretrained(args.model)
    tokenizer.save_pretrained(args.model)

    exit(0)
  else:
    print("Model is not improved")
    exit(1)  
    