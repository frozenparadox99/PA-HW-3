from lazypredict.Supervised import LazyClassifier

from data_preprocessing import DataPreprocessing

data_preprocessor = DataPreprocessing(data_path="Breast_Cancer_dataset.csv")
X_train, X_test, y_train, y_test = data_preprocessor.preprocess_data()

clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)

print(models)