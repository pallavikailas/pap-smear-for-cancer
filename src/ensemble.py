import numpy as np
from sklearn.metrics import accuracy_score
from train import train_random_forest, train_knn, train_svm
from data_loader import get_data
from sklearn.metrics import f1_score

# Weighted Voting: Use f1-score as weights for predictions
def weighted_voting(models, X, score):
    preds = [model.predict_proba(X) for model in models]
    
    # Ensure that predictions are in the expected shape
    print("Predictions shape:", np.array(preds).shape)
    
    # Normalize accuracies so that they sum to 1
    weights = np.array(score) / np.sum(score)
    print("Normalized Weights:", weights)
    
    # Stack the predicted probabilities for each model
    weighted_preds = np.average(np.array(preds), axis=0, weights=weights)
    print("Weighted Predictions:", weighted_preds[0])
    
    # Choose the class with the highest weighted average probability
    return np.argmax(weighted_preds, axis=1)


# Evaluate the ensemble model with weighted voting
def evaluate_ensemble(models, X_val, y_val, score):
    # Get ensemble predictions using weighted voting
    ensemble_preds = weighted_voting(models, X_val, score)
    
    # Calculate the F1-score for the ensemble model
    f1 = f1_score(y_val, ensemble_preds, average='weighted')  # Use 'weighted' for multi-class problems
    print(f"Ensemble F1-Score: {f1 * 100:.2f}%")
