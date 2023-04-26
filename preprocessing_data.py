import pandas as pd
import numpy as np

def fill_missing_values(df):
    for column in df.columns:
        if df[column].dtype == np.object:
            df[column].fillna(df[column].mode()[0], inplace=True)
        else:
            df[column].fillna(df[column].median(), inplace=True)
    return df

def encode_categorical_variables(df):
    categorical_columns = df.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        df[column] = pd.Categorical(df[column]).codes
    return df

def remove_outliers(df, z_threshold=3):
    from scipy import stats
    z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
    df = df[(z_scores < z_threshold).all(axis=1)]
    return df

def normalize_data(df):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    for column in df.columns:
        if df[column].dtype in [np.float64, np.int64]:
            df[column] = scaler.fit_transform(df[[column]])
    return df

def clean_and_organize_data(df):
    # Fill missing values
    df = fill_missing_values(df)

    # Encode categorical variables
    df = encode_categorical_variables(df)

    # Remove outliers
    df = remove_outliers(df)

    # Normalize numerical data
    df = normalize_data(df)

    return df

# Load the dataset
data = pd.read_csv('path/to/your/dataset.csv')

# Clean and organize the data
cleaned_data = clean_and_organize_data(data)

# Save the cleaned dataset
cleaned_data.to_csv('path/to/your/cleaned_dataset.csv', index=False)
