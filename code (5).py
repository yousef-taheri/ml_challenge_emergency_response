import pandas as pd
from config import X_TRAIN_PATH, X_TEST_PATH, Y_TRAIN_PATH
from typing import Tuple

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads the training and test data."""
    try:
        x_train = pd.read_csv(X_TRAIN_PATH)
        x_test = pd.read_csv(X_TEST_PATH)
        y_train = pd.read_csv(Y_TRAIN_PATH)
        return x_train, x_test, y_train
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        raise

def merge_x_y(x_train: pd.DataFrame, y_train: pd.DataFrame) -> pd.DataFrame:
  """Merges x_train and y_train based on 'emergency vehicle selection'."""
  try:
      merged_train = pd.merge(x_train, y_train, on="emergency vehicle selection", how="left")
      return merged_train
  except KeyError as e:
      print(f"Error merging data: Key {e} not found in DataFrames.")
      raise

if __name__ == '__main__':
    # Example Usage
    try:
      x_train, x_test, y_train = load_data()
      print("X_train head:")
      print(x_train.head())
      print("y_train head:")
      print(y_train.head())
      merged_train = merge_x_y(x_train, y_train)
      print("Merged head")
      print(merged_train.head())
    except Exception as e:
        print(f"An error occurred: {e}")