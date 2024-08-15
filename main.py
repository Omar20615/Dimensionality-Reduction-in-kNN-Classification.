import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Load the data from the CSV file
data = pd.read_csv('heart_failure_clinical_records_dataset.csv')

# Define the features for each iteration
features = [
    ['ejection_fraction', 'time'],
    ['ejection_fraction', 'time', 'serum_creatinine'],
    ['ejection_fraction', 'time', 'serum_creatinine', 'age', 'smoking']
]

# Define the different values of k
k_values = [3, 5, 7,9,11]

# Perform KNN classification for each set of features and each value of k
for i, feature_set in enumerate(features):
    for k in k_values:
        accuracies = []
        for _ in range(5):  # repeat 5 times
            X = data[feature_set]
            y = data['DEATH_EVENT']

            # Split the data into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            # Create and fit the KNN classifier
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train_scaled, y_train)

            # Make predictions
            predictions = knn.predict(X_test_scaled)

            # Calculate the accuracy
            accuracy = (predictions == y_test).mean()
            accuracies.append(accuracy)

        # Calculate the average accuracy and standard deviation
        avg_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)

        print(f'Feature set {i+1}, k = {k}: Avg Accuracy = {avg_accuracy:.4f}, Std = {std_accuracy:.4f}')