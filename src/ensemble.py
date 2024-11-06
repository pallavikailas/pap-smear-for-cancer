import numpy as np
from sklearn.metrics import accuracy_score
from train import train_random_forest, train_knn, train_svm
from data_loader import get_data

# Weighted Voting: Use model accuracy as weights for predictions
def weighted_voting(models, X, accuracies):
    # Get predicted probabilities for each model
    preds = [model.predict_proba(X) for model in models]
    
    # Normalize accuracies so that they sum to 1
    weights = np.array(accuracies) / np.sum(accuracies)
    
    # Stack the predicted probabilities for each model
    weighted_preds = np.average(np.array(preds), axis=0, weights=weights)
    
    # Choose the class with the highest weighted average probability
    return np.argmax(weighted_preds, axis=1)

# Evaluate the ensemble model with weighted voting
def evaluate_ensemble(models, X_val, y_val, accuracies):
    # Get ensemble predictions using weighted voting
    ensemble_preds = weighted_voting(models, X_val, accuracies)
    accuracy = accuracy_score(y_val, ensemble_preds)
    print(f"Ensemble Accuracy: {accuracy * 100:.2f}%")
    

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

