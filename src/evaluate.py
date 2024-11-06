from sklearn.metrics import classification_report, f1_score
from train import train_random_forest, train_knn, train_svm
from data_loader import get_data

def evaluate_model(model, X_val, y_val):
    preds = model.predict(X_val)
    f1 = f1_score(y_val, preds, average='weighted')
    report = classification_report(y_val, preds)
    print(f"F1-Score: {f1 * 100:.2f}%")
    print("Classification Report:\n", classification_report(y_val, preds))

if __name__ == "__main__":
    X, y = get_data("data/validation")

    rf_model = train_random_forest(X, y)
    knn_model = train_knn(X, y)
    svm_model = train_svm(X, y)

    print("Random Forest:")
    evaluate_model(rf_model, X, y)
    print("KNN:")
    evaluate_model(knn_model, X, y)
    print("SVM:")
    evaluate_model(svm_model, X, y)
    
