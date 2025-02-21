import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from geopy.distance import geodesic
from config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TEMPORAL_FEATURES, TARGET_VARIABLE

def calculate_distance(row):
    """Calculates Haversine distance between two GPS coordinates."""
    try:
        coords_1 = (row['latitude before departure'], row['longitude before departure'])
        coords_2 = (row['latitude intervention'], row['longitude intervention'])
        return geodesic(coords_1, coords_2).km
    except (ValueError, TypeError):
        return np.nan  # Handle missing or invalid coordinates


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates temporal features from datetime columns."""
    for col in TEMPORAL_FEATURES:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')  # Coerce invalid dates to NaT
                df[f'{col}_hour'] = df[col].dt.hour
                df[f'{col}_dayofweek'] = df[col].dt.dayofweek
                df[f'{col}_month'] = df[col].dt.month
            except Exception as e:
                print(f"Error processing temporal feature {col}: {e}")
                # Handle the error (e.g., drop the column, fill with defaults)
    return df


def preprocess_data(df: pd.DataFrame, is_train: bool = True, encoder=None, scaler=None, imputer=None):
    """Preprocesses the data, handling missing values, scaling, and encoding."""

    # 1. Feature Creation
    if ('latitude before departure' in df.columns and 'longitude before departure' in df.columns and
            'latitude intervention' in df.columns and 'longitude intervention' in df.columns):
        df['distance_km'] = df.apply(calculate_distance, axis=1)

    df = create_temporal_features(df)  # Create temporal features


    # 2. Separate Numerical and Categorical Features
    numerical_cols = [col for col in NUMERICAL_FEATURES + ['distance_km'] + [f'{col}_hour' for col in TEMPORAL_FEATURES] + [f'{col}_dayofweek' for col in TEMPORAL_FEATURES] + [f'{col}_month' for col in TEMPORAL_FEATURES] if col in df.columns]
    categorical_cols = [col for col in CATEGORICAL_FEATURES if col in df.columns]

    # 3. Imputation
    if imputer is None:
        imputer = SimpleImputer(strategy='mean')  # Or 'median', 'most_frequent'
        df[numerical_cols] = imputer.fit_transform(df[numerical_cols]) if numerical_cols else df
    else:
        df[numerical_cols] = imputer.transform(df[numerical_cols]) if numerical_cols else df

    # 4. Encoding
    if encoder is None and categorical_cols:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoded_data = encoder.fit_transform(df[categorical_cols])
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols), index=df.index)

    elif categorical_cols:
        encoded_data = encoder.transform(df[categorical_cols])
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols), index=df.index)
    else:
        encoded_df = pd.DataFrame(index=df.index)  # Empty DataFrame if no categorical features

    # 5. Scaling
    if scaler is None and numerical_cols:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[numerical_cols])
        scaled_df = pd.DataFrame(scaled_data, columns=numerical_cols, index=df.index)
    elif numerical_cols:
        scaled_data = scaler.transform(df[numerical_cols])
        scaled_df = pd.DataFrame(scaled_data, columns=numerical_cols, index=df.index)
    else:
        scaled_df = pd.DataFrame(index=df.index)  # Empty DataFrame if no numerical features

    # 6. Concatenate
    processed_df = pd.concat([scaled_df, encoded_df], axis=1)

    if is_train and TARGET_VARIABLE in df.columns:
      y = df[TARGET_VARIABLE]
      return processed_df, y, encoder, scaler, imputer

    return processed_df, encoder, scaler, imputer



if __name__ == '__main__':
    # Example Usage
    try:
        x_train, x_test, y_train = load_data()
        merged_train = merge_x_y(x_train, y_train)
        X_train_processed, y_train_processed, encoder, scaler, imputer = preprocess_data(merged_train.copy(), is_train=True)
        X_test_processed, encoder, scaler, imputer = preprocess_data(x_test.copy(), is_train=False, encoder=encoder, scaler=scaler, imputer=imputer)
        print("Processed X_train head:")
        print(X_train_processed.head())
        print("Processed X_test head:")
        print(X_test_processed.head())
        print("y_train head:")
        print(y_train_processed.head())

    except Exception as e:
        print(f"An error occurred: {e}")
