import pandas as pd

def load_dataset(csv_path,
                 text_column='text',
                 target_column='sentiment'):
  """
  Load the data from the csv file.

  Args:
    csv_path: The path to the csv file.
    text_column: The column name of the text column.
    target_column: The column name of the target column.

  Returns:
    X: The text data.
    y: The target data.
  """
  df = pd.read_csv(csv_path)
  X = df[text_column]
  y = df[target_column]

  return X, y

def save_dataset(X, y, path):
  """
  Save the data to a csv file.

  Args:
    X: The text data.
    y: The target data.
    path: The path to the csv file.
  """
  df = pd.DataFrame({
    "text": X,
    "sentiment": y,
  })

  df.to_csv(path,index=False)
