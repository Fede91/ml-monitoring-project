# Rif https://github.com/cardiffnlp/timelms/blob/main/scripts/train_sentiment.py
import argparse
from datasets import load_dataset
from model import model, tokenizer, device, MODEL_NAME
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
import numpy as np
from sklearn.metrics import accuracy_score
import evaluate
import json
from metrics import compare_metrics, save_metrics
import os

metric = evaluate.load('accuracy')

def preprocess(data):
  label_dict = {'negative':0, 'neutral': 1, 'positive': 2}
  
  tokenized = tokenizer(data["text"], truncation=True) #, return_tensors="pt") #.to(device)
  tokenized["labels"] = [label_dict[label] for label in data['sentiment'] ]
  return tokenized

def compute_metrics(eval_pred):
  # labels = pred.label_ids
  # preds = np.argmax(pred.predictions, axis=1)
  logits, labels = eval_pred
  predictions = np.argmax(logits, axis=-1)

  return metric.compute(predictions=predictions, references=labels)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--trainset', default='./data/train.csv', help='Path to the CSV train set')
  parser.add_argument('--validationset', default='./data/validation.csv', help='Path to the CSV validation set')
  parser.add_argument("--model", default="./model", help="Folder path to save the model")
  parser.add_argument("--metrics", default="./metrics/base_metrics.json", help="Path to save the metrics file")
  args = parser.parse_args()
  
  dataset = load_dataset('csv',
                         data_files={'train': args.trainset, 'validation': args.validationset})

  encoded_dataset = dataset.map(preprocess, batched=True)

  data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

  training_args = TrainingArguments(
    do_eval=True,
    output_dir=args.model,
    # logging_dir='./logs',
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

  # 6. Addestra (incrementale se il modello era già fine-tunato)
  trainer.train()

  eval_results = trainer.evaluate()

  new_metrics = {
    'accuracy': eval_results['eval_accuracy']
  }

  print(f"Eval accuracy: {eval_results['eval_accuracy']:.4f}")

  with open(args.metrics, 'r') as f:
    base_metrics = json.load(f)

  if compare_metrics(base_metrics, new_metrics):
    save_metrics(new_metrics, args.metrics)

    trainer.push_to_hub()

    exit(0)
  else:
    print("Model is not improved")
    exit(1)  