import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from data_loader import get_data

# Soft Voting: Ensuring all models output probabilities
def soft_voting(models, X):
    preds = [model.predict_proba(X) for model in models]
    
    # Stack the predicted probabilities for each model (along axis 0)
    avg_preds = np.mean(preds, axis=0)
    
    # Choose the class with the highest average probability
    return np.argmax(avg_preds, axis=1)

# Improved training functions
def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    return rf

def train_knn(X_train, y_train):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    return knn

def train_svm(X_train, y_train):
    svm = SVC(kernel="linear", probability=True, random_state=42)  # Ensure probability=True for predict_proba
    svm.fit(X_train, y_train)
    return svm

# Evaluate the ensemble model
def evaluate_ensemble(models, X_val, y_val):
    ensemble_preds = soft_voting(models, X_val)
    accuracy = accuracy_score(y_val, ensemble_preds)
    print(f"Ensemble Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    # Load and prepare data
    X, y = get_data("data/validation")
    
    # Standardize the features before training
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split into training and validation
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train individual models
    rf_model = train_random_forest(X_train, y_train)
    knn_model = train_knn(X_train, y_train)
    svm_model = train_svm(X_train, y_train)

    # Evaluate ensemble
    print("Ensemble Model:")
    evaluate_ensemble([rf_model, knn_model, svm_model], X_val, y_val)

#import numpy as np
#from sklearn.metrics import accuracy_score
#from train import train_random_forest, train_knn, train_svm
#from data_loader import get_data

# def soft_voting(models, X):
#     preds = [model.predict_proba(X) for model in models]
#     avg_preds = np.mean(preds, axis=0)
#     return np.argmax(avg_preds, axis=1)

# def evaluate_ensemble(models, X_val, y_val):
#     ensemble_preds = soft_voting(models, X_val)
#     accuracy = accuracy_score(y_val, ensemble_preds)
#     print(f"Ensemble Accuracy: {accuracy * 100:.2f}%")

# if __name__ == "__main__":
#     X, y = get_data("data/validation")

#     rf_model = train_random_forest(X, y)
#     knn_model = train_knn(X, y)
#     svm_model = train_svm(X, y)

#     print("Ensemble Model:")
#     evaluate_ensemble([rf_model, knn_model, svm_model], X, y)

