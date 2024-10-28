from sklearn.metrics import accuracy_score, classification_report
from train import train_random_forest, train_knn, train_svm
from data_loader import get_data

def evaluate_model(model, X_val, y_val):
    preds = model.predict(X_val)
    accuracy = accuracy_score(y_val, preds)
    report = classification_report(y_val, preds)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:\n", report)

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
