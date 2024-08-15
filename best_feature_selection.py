from sklearn.feature_selection import RFE, SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from scipy.stats import pearsonr
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.linear_model import LinearRegression

heart_attack_data = pd.read_csv("heart_failure_clinical_records_dataset.csv")
X = heart_attack_data.iloc[:, :-1].values
y = heart_attack_data.iloc[:, -1].values


def apply_knn(X_train, X_test, y_train, y_test, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=None)
    return X_train, X_test, y_train, y_test


def compare_results(results):
    for method, acc in results.items():
        max_k, max_accuracy = max(acc, key=lambda x: x[1])
        print(f"{method}: k = {max_k}, Max Accuracy = {max_accuracy:.4f}")



def feature_selection(X_train, y_train, X_test, method, feature_names):
    selected_features = None

    if method == 'EFS':
        lr = LinearRegression()
        efs = EFS(lr, 
                  min_features=1,
                  max_features=X_train.shape[1],  # set max_features to the total number of features
                  scoring='neg_mean_squared_error',
                  print_progress=True,
                  cv=5)
        efs = efs.fit(X_train, y_train)
        X_train = efs.transform(X_train)
        X_test = efs.transform(X_test)
        selected_features = feature_names[list(efs.best_idx_)]
    elif method == 'Pearson':
        correlations = []
        for i in range(X_train.shape[1]):
            corr, _ = pearsonr(X_train[:, i], y_train)
            if abs(corr) > 0.3:  # lower the threshold
                correlations.append(i)
        X_train = X_train[:, correlations]
        X_test = X_test[:, correlations]
        selected_features = feature_names[correlations]
    return X_train, X_test, selected_features


def main():
    k_values = [1,3, 5, 7,9,11]
    num_repeats = 5
    methods = ['EFS', 'Pearson']
    results = {}
    feature_names = heart_attack_data.columns[:-1]

    for method in methods:
        results[method] = []
        for _ in range(num_repeats):
            X_train, X_test, y_train, y_test = split_data(X, y)
            X_train, X_test, selected_features = feature_selection(X_train, y_train, X_test, method, feature_names)
            for k in k_values:
                accuracy = apply_knn(X_train, X_test, y_train, y_test, k)
                results[method].append((k, accuracy, selected_features))

    # Find the best combination
    best_method, best_k, best_accuracy, best_features = max(
        ((method, k, accuracy, features) for method, result in results.items() for k, accuracy, features in result),
        key=lambda x: x[2]
    )
    print(f"Best method: {best_method}, Best k: {best_k}, Best accuracy: {best_accuracy}, Best features: {best_features}")

if __name__ == "__main__":
    main()