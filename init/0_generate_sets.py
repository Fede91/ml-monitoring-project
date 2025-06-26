import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
import os

RANDOM_STATE = 1

if __name__ == "__main__":
  # Parse the arguments
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", default="../data/data.csv", help="Dataset path")
  parser.add_argument("--output", default="../data", help="Output path")
  args = parser.parse_args()

  # Read the dataset
  df = pd.read_csv(args.dataset)
  
  # Keep only text and sentiment columns
  df = df[['text', 'sentiment']]

  # Split the dataset into train and test sets
  train_df, test_df = train_test_split(df, test_size=0.3, random_state=RANDOM_STATE)

  # Split the train set into 10 parts
  chunk_size = len(train_df) // 10
  remainder = len(train_df) % 10

  for i in range(10):
    start_idx = i * chunk_size
    # Add extra rows to the last chunk if necessary
    end_idx = start_idx + chunk_size + (remainder if i == 9 else 0)
    
    chunk_df = train_df.iloc[start_idx:end_idx].copy()
    
    # Save with the same header
    output_file = os.path.join(args.output, f'train_chunk_{i+1}.csv')
    chunk_df.to_csv(output_file, index=False)
    
    print(f"Created {output_file} with {len(chunk_df)} rows")

  print(f"Split the train_data.csv file into 10 parts in the directory {args.output}")
  
  # Save the test set
  test_df.to_csv(f"{args.output}/validation.csv", index=False)

  print(f"Validation set saved in {args.output}/validation.csv")