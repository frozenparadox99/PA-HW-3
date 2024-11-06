import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from data_preprocessing import DataPreprocessing


class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)  # Ensure y_train is a NumPy array for indexing consistency

    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)

    def _predict(self, x):
        # Compute Euclidean distances
        distances = np.linalg.norm(self.X_train - x, axis=1)
        # Get indices of the k nearest neighbors
        k_indices = distances.argsort()[:self.k]
        # Get the labels of the k nearest neighbors
        k_nearest_labels = self.y_train[k_indices]
        # Return the most common class label
        return np.bincount(k_nearest_labels).argmax()
    

if __name__ == "__main__":
    # Train and test KNN
    data_preprocessor = DataPreprocessing(data_path="Breast_Cancer_dataset.csv")
    X_train, X_test, y_train, y_test = data_preprocessor.preprocess_data()
    knn = KNNClassifier(k=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred_knn)
    precision = precision_score(y_test, y_pred_knn, average='weighted')  # Use 'weighted' for multiclass
    recall = recall_score(y_test, y_pred_knn, average='weighted')
    f1 = f1_score(y_test, y_pred_knn, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred_knn)
    class_report = classification_report(y_test, y_pred_knn)

    # Display the metrics
    print(f"KNN Accuracy: {accuracy:.4f}")
    print(f"KNN Precision: {precision:.4f}")
    print(f"KNN Recall: {recall:.4f}")
    print(f"KNN F1 Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)