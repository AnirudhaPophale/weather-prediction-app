import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import joblib

# Load dataset
data = pd.read_csv('seattle-weather.csv')
numeric_data = data.select_dtypes(include=['number'])  # Select only numeric columns
data[numeric_data.columns] = numeric_data.fillna(numeric_data.mean())  # Fill NaNs with mean values

# Check the first few rows of the dataset to understand its structure
print(data.head())
print(data.info())
# Handle missing values (if any)
imputer = SimpleImputer(strategy='mean')
data.fillna(data.mean(), inplace=True)

# Encode categorical features
label_encoder = LabelEncoder()
if 'weather' in data.columns:
    data['weather'] = label_encoder.fit_transform(data['weather'])

# Select features and target
X = data.drop(['date', 'weather'], axis=1)  # Drop 'date' and other irrelevant columns
y = data['weather']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize the classifiers
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Voting Classifier as an ensemble of Random Forest and Gradient Boosting
ensemble_model = VotingClassifier(estimators=[
    ('rf', rf_model), ('gb', gb_model)],
    voting='soft'
)

# Train and evaluate each model
models = {'Random Forest': rf_model, 'Gradient Boosting': gb_model, 'Ensemble Voting': ensemble_model}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy * 100:.2f}%")


# Save the best-performing model
best_model = ensemble_model  # Replace with the chosen model
joblib.dump(best_model, 'best_weather_prediction_model.pkl')
print("Best model saved as 'best_weather_prediction_model.pkl'")
