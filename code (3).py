from sklearn.model_selection import train_test_split
from config import RANDOM_STATE, TEST_SIZE

def split_data(X, y):
    """Splits data into training and validation sets."""
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    return X_train, X_val, y_train, y_val