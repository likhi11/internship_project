import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from preprocess import preprocess_data

# Load dataset
df = pd.read_csv(
    'dataset/accident_data.csv',
    encoding='latin1'
)
df.columns = (
    df.columns
    .str.strip()
    .str.replace(' ', '_')
    .str.replace('-', '_')
)
print(df.columns.tolist())
print(df['Accident_Severity'].value_counts())
# Select features
features = [
    'Road_Type',
    'Speed_limit',
    'Number_of_Vehicles',
    'Number_of_Casualties',
    'Day_of_Week'
]

# Keep only needed columns
df = df[features + ['Accident_Severity']]

# Remove missing values
df.dropna(inplace=True)

# Preprocess
df = preprocess_data(df)

# Features and target
X = df[features]
y = df['Accident_Severity']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = RandomForestClassifier(
    class_weight='balanced',
    n_estimators=200,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print(f'Accuracy: {accuracy * 100:.2f}%')

# Save model
joblib.dump(model, 'models/accident_model.pkl')

print('Model saved successfully')