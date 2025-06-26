import pandas as pd

def load_dataset(csv_path,
                 text_column='text',
                 target_column='sentiment'):
  """
  Load the data from the csv file.
  """
  df = pd.read_csv(csv_path)
  X = df[text_column]
  y = df[target_column]

  return X, y

def save_dataset(X, y, path):
  df = pd.DataFrame({
    "text": X,
    "sentiment": y,
  })

  df.to_csv(path,index=False)
