# Rif https://github.com/cardiffnlp/timelms/blob/main/scripts/train_sentiment.py
import argparse
from datasets import load_dataset
from model import model, tokenizer,  MODEL_NAME
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
import numpy as np
import evaluate
import json
from metrics import compare_metrics, save_metrics
import os

# Load the accuracy metric
metric = evaluate.load('accuracy')

def preprocess(data):
  """
  Preprocess the data.

  Args:
    data: The data to preprocess.

  Returns:
    The preprocessed data.
  """
  label_dict = {'negative':0, 'neutral': 1, 'positive': 2}
  
  # Tokenize the text
  tokenized = tokenizer(data["text"], truncation=True) #, return_tensors="pt") #.to(device)
  tokenized["labels"] = [label_dict[label] for label in data['sentiment'] ]

  return tokenized

def compute_metrics(eval_pred):
  """
  Compute the metrics.

  Args:
    eval_pred: The evaluation predictions.

  Returns:
    The metrics.
  """
  logits, labels = eval_pred
  predictions = np.argmax(logits, axis=-1)

  return metric.compute(predictions=predictions, references=labels)

if __name__ == "__main__":
  # Parse the arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--trainset', default='./data/train.csv', help='Path to the CSV train set')
  parser.add_argument('--validationset', default='./data/validation.csv', help='Path to the CSV validation set')
  parser.add_argument("--model", default="./model", help="Folder path to save the model")
  parser.add_argument("--metrics", default="./metrics/base_metrics.json", help="Path to save the metrics file")
  args = parser.parse_args()
  
  # Load the dataset
  dataset = load_dataset('csv',
                         data_files={'train': args.trainset, 'validation': args.validationset})

  # Encode the dataset
  encoded_dataset = dataset.map(preprocess, batched=True)

  # Create the data collator
  data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

  # Create the training arguments using "epoch" as strategy
  training_args = TrainingArguments(
    do_eval=True,
    output_dir=args.model,
    num_train_epochs=1,
    logging_strategy='epoch',
    learning_rate=1e-05,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model='eval_accuracy',
    push_to_hub_token=os.getenv("HF_TOKEN"),
    push_to_hub=True,
    hub_model_id=MODEL_NAME,
  )

  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
  )

  # Train the model
  trainer.train()

  # Evaluate the model
  eval_results = trainer.evaluate()

  new_metrics = {
    'accuracy': eval_results['eval_accuracy']
  }

  print(f"Eval accuracy: {eval_results['eval_accuracy']:.4f}")

  with open(args.metrics, 'r') as f:
    base_metrics = json.load(f)

  # Compare the metrics
  if compare_metrics(base_metrics, new_metrics):
    # If the metrics are better, save the metrics
    save_metrics(new_metrics, args.metrics)

    # Push the model to the Hugging Face hub
    trainer.push_to_hub()

    # Exit with success
    exit(0)
  else:
    # If the metrics are not better, exit with failure
    print("Model is not improved")
    exit(1)  