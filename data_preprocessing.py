import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

class DataPreprocessing:
    def __init__(self, data_path, target_column='Status', test_size=0.3, random_state=42, pca_components=10):
        self.data_path = data_path
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.pca_components = pca_components
        self.data = pd.read_csv(data_path)
    
    def preprocess_data(self):
        # Separate numerical and categorical columns
        num_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        cat_cols = self.data.select_dtypes(include=['object']).columns
        
        # Impute missing values
        imputer_num = SimpleImputer(strategy='mean')
        imputer_cat = SimpleImputer(strategy='most_frequent')
        self.data[num_cols] = imputer_num.fit_transform(self.data[num_cols])
        self.data[cat_cols] = imputer_cat.fit_transform(self.data[cat_cols])

        # Scale numerical columns
        scaler = StandardScaler()
        self.data[num_cols] = scaler.fit_transform(self.data[num_cols])

        # Encode categorical columns
        encoder = LabelEncoder()
        for col in cat_cols:
            self.data[col] = encoder.fit_transform(self.data[col])

        # Split data into features (X) and target (y)
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]

        # Apply PCA if feature count exceeds threshold
        if X.shape[1] > self.pca_components:
            pca = PCA(n_components=self.pca_components)
            X_reduced = pca.fit_transform(X)
        else:
            X_reduced = X  # No dimensionality reduction needed

        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_reduced, y, test_size=self.test_size, random_state=self.random_state
        )

        return X_train, X_test, y_train, y_test
