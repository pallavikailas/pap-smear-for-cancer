from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from data_loader import get_data

def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    return rf

def train_knn(X_train, y_train):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    return knn

def train_svm(X_train, y_train):
    svm = SVC(kernel="linear", probability=True)
    svm.fit(X_train, y_train)
    return svm

if __name__ == "__main__":
    X, y = get_data("data/train")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = train_random_forest(X_train, y_train)
    knn_model = train_knn(X_train, y_train)
    svm_model = train_svm(X_train, y_train)
