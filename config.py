import os

# Data paths
DATA_DIR = "data"
X_TRAIN_PATH = os.path.join(DATA_DIR, "x_train.csv")
X_TEST_PATH = os.path.join(DATA_DIR, "x_test.csv")
Y_TRAIN_PATH = os.path.join(DATA_DIR, "y_train_u9upqBE.csv")

# Feature configuration (example)
CATEGORICAL_FEATURES = ["alert reason category", "alert reason", "emergency vehicle type", "rescue center"]
NUMERICAL_FEATURES = ["longitude intervention", "latitude intervention", "longitude before departure", "latitude before departure"]
TEMPORAL_FEATURES = ["selection time", "date key sélection", "time key sélection"]

# Target variable
TARGET_VARIABLE = "delta departure-presentation"  # Or "delta selection-departure"

# Model training parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2  # For validation split
