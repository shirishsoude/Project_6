import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import subprocess


# Step 1: Load the dataset
data = pd.read_csv('movie.csv')  # Assuming the dataset is in CSV format
print("Dataset shape:", data.shape)

# Step 2: Preprocess the data
# Assuming the dataset has two columns: 'review' and 'sentiment'
reviews = data['review']
sentiments = data['sentiment']

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(reviews, sentiments, test_size=0.2, random_state=42)

# Step 4: Feature extraction using Bag-of-Words model
vectorizer = CountVectorizer()
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

# Print intermediate values to check the code execution
print("X_train shape:", X_train_features.shape)
print("y_train shape:", y_train.shape)

# Step 5: Train the classifier (Support Vector Machine)
classifier = SVC()
classifier.fit(X_train_features, y_train)

# Step 6: Make predictions on the test set
y_pred = classifier.predict(X_test_features)

# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Execute PowerShell command to re-enable PSReadLine if screen reader warning is present
result = subprocess.run(['powershell', 'Import-Module', 'PSReadLine'], capture_output=True, text=True)
if 'screen reader' in result.stdout:
    print("Screen reader warning detected. Re-enabling PSReadLine.")
    subprocess.run(['powershell', '-Command', 'Import-Module PSReadLine'])

