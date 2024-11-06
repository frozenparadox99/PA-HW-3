from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from data_preprocessing import DataPreprocessing

data_preprocessor = DataPreprocessing(data_path="Breast_Cancer_dataset.csv")
X_train, X_test, y_train, y_test = data_preprocessor.preprocess_data()
# Hyperparameter search for Random Forest
print("\nHyperparameter Search for Random Forest")
rf_param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rf_grid_search = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=3, scoring='accuracy')
rf_grid_search.fit(X_train, y_train)
print(f"Best parameters for Random Forest: {rf_grid_search.best_params_}")
rf_best_model = rf_grid_search.best_estimator_
y_pred_rf_best = rf_best_model.predict(X_test)

print("\nRandom Forest Metrics with Best Parameters:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf_best):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_rf_best, average='weighted'):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_rf_best, average='weighted'):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_rf_best, average='weighted'):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf_best))


print("\nHyperparameter Search for Neural Network")
nn_param_grid = {
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate_init': [0.001, 0.01, 0.1]  # Using learning_rate_init as the learning rate parameter
}

nn_grid_search = GridSearchCV(MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42), nn_param_grid, cv=3, scoring='accuracy')
nn_grid_search.fit(X_train, y_train)
print(f"Best parameters for Neural Network: {nn_grid_search.best_params_}")
nn_best_model = nn_grid_search.best_estimator_
y_pred_nn_best = nn_best_model.predict(X_test)

print("\nNeural Network Metrics with Best Parameters:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_nn_best):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_nn_best, average='weighted'):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_nn_best, average='weighted'):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_nn_best, average='weighted'):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_nn_best))