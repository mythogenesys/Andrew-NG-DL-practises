import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tqdm import tqdm

class LogisticRegression:
    def __init__(self, num_iterations=5000, learning_rate=0.001, lambda_param=0.01, 
                 beta1=0.9, beta2=0.999, epsilon=1e-8, is_adam=True, is_earlystopping=True):
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.is_adam = is_adam
        self.is_earlystopping = is_earlystopping
        self.w = None
        self.b = None
        self.m_w = None
        self.v_w = None
        self.m_b = None
        self.v_b = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def initialize_parameters(self, dim):
        self.w = np.zeros((dim, 1))
        self.b = 0
        self.m_w = np.zeros((dim, 1))
        self.v_w = np.zeros((dim, 1))
        self.m_b = 0
        self.v_b = 0

    def propagate(self, X, Y):
        m = X.shape[1]
        A = self.sigmoid(np.dot(self.w.T, X) + self.b)
        cost = -1/m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) + (self.lambda_param / (2 * m)) * np.sum(self.w**2)
        dw = 1/m * np.dot(X, (A - Y).T) + (self.lambda_param / m) * self.w
        db = 1/m * np.sum(A - Y)
        return dw, db, cost

    def optimize(self, X, Y, X_val, Y_val):
        costs = []
        train_accuracies = []
        val_accuracies = []
        best_val_cost = float('inf')
        patience = 5
        patience_counter = 0
        t = 0
        
        for i in tqdm(range(self.num_iterations), desc="Training"):
            t += 1
            dw, db, cost = self.propagate(X, Y)
            
            if self.is_adam:
                # Adam optimization
                self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * dw
                self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (dw**2)
                m_w_corrected = self.m_w / (1 - self.beta1**t)
                v_w_corrected = self.v_w / (1 - self.beta2**t)
                self.w -= self.learning_rate * m_w_corrected / (np.sqrt(v_w_corrected) + self.epsilon)

                self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * db
                self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (db**2)
                m_b_corrected = self.m_b / (1 - self.beta1**t)
                v_b_corrected = self.v_b / (1 - self.beta2**t)
                self.b -= self.learning_rate * m_b_corrected / (np.sqrt(v_b_corrected) + self.epsilon)
            else:
                # Standard gradient descent
                self.w -= self.learning_rate * dw
                self.b -= self.learning_rate * db
            
            if i % 100 == 0:
                costs.append(cost)
                train_acc = self.score(X, Y)
                val_acc = self.score(X_val, Y_val)
                train_accuracies.append(train_acc)
                val_accuracies.append(val_acc)
                
                if self.is_earlystopping:
                    _, _, val_cost = self.propagate(X_val, Y_val)
                    if val_cost < best_val_cost:
                        best_val_cost = val_cost
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= patience:
                        print(f"Early stopping at iteration {i}")
                        break
        
        return costs, train_accuracies, val_accuracies
    
    def predict(self, X):
        A = self.sigmoid(np.dot(self.w.T, X) + self.b)
        return (A > 0.5).astype(int)

    def fit(self, X_train, Y_train, X_val, Y_val):
        self.initialize_parameters(X_train.shape[0])
        return self.optimize(X_train, Y_train, X_val, Y_val)

    def score(self, X, Y):
        Y_prediction = self.predict(X)
        return 1 - np.mean(np.abs(Y_prediction - Y))

def preprocess_data(df):
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    df['Title'] = df['Title'].map(title_mapping)
    
    df['AgeBand'] = pd.cut(df['Age'], 5)
    
    df['Deck'] = df['Cabin'].str.slice(0, 1)
    df['Deck'] = df['Deck'].fillna('U')
    
    df['FareBand'] = pd.qcut(df['Fare'], 4)
    
    return df

def prepare_data(df):
    X = df.drop(['Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
    y = df['Survived']
    
    numeric_features = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']
    categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'IsAlone', 'Deck']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    X_processed = preprocessor.fit_transform(X)
    
    return X_processed, y

def load_and_split_data(file_path):
    df = pd.read_csv(file_path)
    df = preprocess_data(df)
    X, y = prepare_data(df)
    return train_test_split(X, y, test_size=0.2, random_state=42)
