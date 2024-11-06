import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


from data_preprocessing import DataPreprocessing

data_preprocessor = DataPreprocessing(data_path="Breast_Cancer_dataset.csv")
X_train, X_test, y_train, y_test = data_preprocessor.preprocess_data()

# Helper function to train and evaluate models
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    # KNN Classifier (from scratch)
    class KNNClassifier:
        def __init__(self, k=3):
            self.k = k

        def fit(self, X_train, y_train):
            self.X_train = np.array(X_train)
            self.y_train = np.array(y_train)

        def predict(self, X_test):
            predictions = [self._predict(x) for x in X_test]
            return np.array(predictions)

        def _predict(self, x):
            distances = np.linalg.norm(self.X_train - x, axis=1)
            k_indices = distances.argsort()[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            return np.bincount(k_nearest_labels).argmax()

    knn = KNNClassifier(k=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)

    # Naive Bayes
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    y_pred_nb = nb_model.predict(X_test)

    # Decision Tree
    dt_model = DecisionTreeClassifier(criterion='entropy', random_state=42)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    # Gradient Boosting
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)
    y_pred_gb = gb_model.predict(X_test)

    # Neural Network
    nn_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
    nn_model.fit(X_train, y_train)
    y_pred_nn = nn_model.predict(X_test)

    # Collect model predictions
    model_predictions = {
        'KNN': y_pred_knn,
        'Naive Bayes': y_pred_nb,
        'Decision Tree': y_pred_dt,
        'Random Forest': y_pred_rf,
        'Gradient Boosting': y_pred_gb,
        'Neural Network': y_pred_nn
    }

    # Print accuracy and other metrics for all models
    for name, y_pred in model_predictions.items():
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        print(f"\n{name} Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

# Apply each feature selection technique and train models

from sklearn.preprocessing import MinMaxScaler

# Apply MinMaxScaler to ensure non-negative input for chi2 test
scaler_minmax = MinMaxScaler()
X_train_non_negative = scaler_minmax.fit_transform(X_train)
X_test_non_negative = scaler_minmax.transform(X_test)

# Apply SelectKBest (Chi-square) on non-negative data
print("\nApplying SelectKBest (Chi-square)")
selector_kbest = SelectKBest(score_func=chi2, k=10)
X_train_kbest = selector_kbest.fit_transform(X_train_non_negative, y_train)
X_test_kbest = selector_kbest.transform(X_test_non_negative)
train_and_evaluate_models(X_train_kbest, X_test_kbest, y_train, y_test)


# 2. L1 Regularization (Lasso)
print("\nApplying L1 Regularization (Lasso)")
lasso = LogisticRegression(penalty='l1', solver='liblinear', C=0.1)
lasso.fit(X_train, y_train)
selected_features_l1 = np.where(lasso.coef_[0] != 0)[0]
X_train_l1 = X_train[:, selected_features_l1]
X_test_l1 = X_test[:, selected_features_l1]
train_and_evaluate_models(X_train_l1, X_test_l1, y_train, y_test)

# 3. Tree-based Feature Selection
print("\nApplying Tree-based Feature Selection")
tree_model = ExtraTreesClassifier(n_estimators=100, random_state=42)
tree_model.fit(X_train, y_train)
importances = tree_model.feature_importances_
indices_tree = np.argsort(importances)[::-1][:10]  # Select top 10 features
X_train_tree = X_train[:, indices_tree]
X_test_tree = X_test[:, indices_tree]
train_and_evaluate_models(X_train_tree, X_test_tree, y_train, y_test)

# 4. Recursive Feature Elimination with Cross-Validation (RFECV)
print("\nApplying RFECV")
rfecv = RFECV(estimator=RandomForestClassifier(n_estimators=100, random_state=42), step=1, cv=5)
rfecv.fit(X_train, y_train)
X_train_rfecv = rfecv.transform(X_train)
X_test_rfecv = rfecv.transform(X_test)
train_and_evaluate_models(X_train_rfecv, X_test_rfecv, y_train, y_test)

# 5. Mutual Information
print("\nApplying Mutual Information")
mi_scores = mutual_info_classif(X_train, y_train)
indices_mi = np.argsort(mi_scores)[::-1][:10]  # Select top 10 features
X_train_mi = X_train[:, indices_mi]
X_test_mi = X_test[:, indices_mi]
train_and_evaluate_models(X_train_mi, X_test_mi, y_train, y_test)
