# Predicitve Analytics: HW 3

## Link to Colab Notebook 
This contains all diagrams, experiments and visualizations
[Colab](https://colab.research.google.com/drive/1o7VcV28JWraWLrTLmxbxoV8I8B8bI_oq?usp=sharing)

## Preprocessing
- Missing value imputation using mean for numerical and most_frequent for categorical
- Normalization
- Encoding
- PCA with features capturing 95% variance

## Modeling
- KNN Accuracy: 88.66%
- Naive BayesAccuracy: 87.25%
- Decision Tree Accuracy: 85.18%
- Random Forest Accuracy: 90.65%
- Gradient Boosting Accuracy: 89.74%
- Neural Network Accuracy: 90.31%

## Modeling with Feature Selection and Reranking
| Method       | KNN Accuracy | KNN F1  | Naive Bayes Accuracy | Naive Bayes F1 | Decision Tree Accuracy | Decision Tree F1 | Random Forest Accuracy | Random Forest F1 | Gradient Boost Accuracy | Gradient Boost F1 | Neural Network Accuracy | Neural Network F1 |
|--------------|--------------|---------|-----------------------|----------------|------------------------|------------------|------------------------|------------------|-------------------------|-------------------|--------------------------|--------------------|
| Chi Square   | 0.88         | 0.8614  | 0.8725               | 0.8663         | 0.8518                | 0.8537          | 0.9065                | 0.8961           | 0.8974                  | 0.887            | 0.9015                  | 0.891             |
| Lasso Reg.   | 0.8924       | 0.8789  | 0.8916               | 0.8813         | 0.8452                | 0.8459          | 0.9073                | 0.8976           | 0.8998                  | 0.8903           | 0.9023                  | 0.8936            |
| Tree Based   | 0.8866       | 0.8726  | 0.8725               | 0.8663         | 0.856                 | 0.8563          | 0.9073                | 0.8965           | 0.8974                  | 0.887            | 0.8982                  | 0.8885            |
| RFECV        | 0.8957       | 0.8831  | 0.8962               | 0.8848         | 0.8386                | 0.8454          | 0.9048                | 0.8961           | 0.899                   | 0.8892           | 0.9031                  | 0.892             |
| Mutual Info  | 0.8866       | 0.8726  | 0.8725               | 0.8663         | 0.851                 | 0.8513          | 0.9048                | 0.8939           | 0.8982                  | 0.8877           | 0.9015                  | 0.8921            |

# Metrics for more models
| Model                        | Accuracy | Balanced Accuracy | ROC AUC | F1 Score |
|------------------------------|----------|-------------------|---------|----------|
| NearestCentroid              | 0.82     | 0.81             | 0.81    | 0.84     |
| Perceptron                   | 0.82     | 0.75             | 0.75    | 0.84     |
| SGDClassifier                | 0.91     | 0.75             | 0.75    | 0.90     |
| QuadraticDiscriminantAnalysis| 0.88     | 0.74             | 0.74    | 0.88     |
| LinearDiscriminantAnalysis   | 0.91     | 0.74             | 0.74    | 0.90     |
| LGBMClassifier               | 0.90     | 0.73             | 0.73    | 0.89     |
| LogisticRegression           | 0.90     | 0.73             | 0.73    | 0.90     |
| RandomForestClassifier       | 0.91     | 0.73             | 0.73    | 0.90     |
| XGBClassifier                | 0.90     | 0.73             | 0.73    | 0.89     |
| CalibratedClassifierCV       | 0.90     | 0.72             | 0.72    | 0.89     |
| AdaBoostClassifier           | 0.89     | 0.72             | 0.72    | 0.88     |
| LinearSVC                    | 0.90     | 0.71             | 0.71    | 0.89     |
| GaussianNB                   | 0.87     | 0.70             | 0.70    | 0.87     |
| ExtraTreesClassifier         | 0.90     | 0.70             | 0.70    | 0.88     |
| DecisionTreeClassifier       | 0.85     | 0.69             | 0.69    | 0.85     |
| BaggingClassifier            | 0.89     | 0.68             | 0.68    | 0.87     |
| LabelPropagation             | 0.85     | 0.68             | 0.68    | 0.85     |
| LabelSpreading               | 0.85     | 0.68             | 0.68    | 0.85     |
| SVC                          | 0.89     | 0.66             | 0.66    | 0.87     |
| RidgeClassifierCV            | 0.89     | 0.66             | 0.66    | 0.87     |
| RidgeClassifier              | 0.89     | 0.66             | 0.66    | 0.87     |
| KNeighborsClassifier         | 0.88     | 0.65             | 0.65    | 0.86     |
| ExtraTreeClassifier          | 0.80     | 0.63             | 0.63    | 0.81     |
| BernoulliNB                  | 0.88     | 0.61             | 0.61    | 0.85     |
| PassiveAggressiveClassifier  | 0.80     | 0.55             | 0.55    | 0.79     |
| DummyClassifier              | 0.86     | 0.50             | 0.50    | 0.79     |
