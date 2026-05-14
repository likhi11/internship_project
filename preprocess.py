import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

def preprocess_data(df):

    encoders = {}

    # Find all text columns
    categorical_columns = df.select_dtypes(include=['object']).columns

    for col in categorical_columns:

        le = LabelEncoder()

        df[col] = le.fit_transform(df[col].astype(str))

        encoders[col] = le

    # Save encoders
    joblib.dump(encoders, 'models/encoders.pkl')

    return df