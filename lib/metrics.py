import argparse
import json
import os

def compare_metrics(base_metrics, new_metrics):
  """
  Compare the accuracy of the base and new metrics.

  Args:
    base_metrics: The base metrics.
    new_metrics: The new metrics.

  Returns:
    True if the accuracy of the new metrics is more than 2% higher than the base metrics, False otherwise.
  """

  if not new_metrics or not base_metrics:
    return False

  return new_metrics['accuracy'] - base_metrics['accuracy'] >= 0.02

def save_metrics(metrics, output_path):
  """
  Save the metrics to a file.

  Args:
    metrics: The metrics to save.
    output_path: The path to the output file.
  """
  with open(output_path, 'w') as f:
    json.dump(metrics, f)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--base', default='../metrics/base_metrics.json', help='Path to the base metrics file')
  parser.add_argument('--new', default='../metrics/new_metrics.json', help='Path to the new metrics file')
  args = parser.parse_args()

  if not os.path.exists(args.base) or not os.path.exists(args.new):
    print("Error: Base or new metrics file does not exist")
    exit(1)

  with open(args.base, 'r') as f:
    base_metrics = json.load(f)

  with open(args.new, 'r') as f:
    new_metrics = json.load(f)

  exit(0) if compare_metrics(base_metrics, new_metrics) else exit(1)