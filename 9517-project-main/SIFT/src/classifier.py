from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

def train_and_evaluate_svm(X_train, X_test, y_train, y_test, random_state=42):
    """
    Train and evaluate an SVM classifier.
    """
    print("🔧 Training SVM classifier...")
    clf = SVC(kernel="linear", random_state=random_state)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("✅ SVM classifier training complete!")

    return clf, y_pred


def train_and_evaluate_knn(X_train, X_test, y_train, y_test, k=5):
    """
    Train and evaluate a KNN classifier.
    """
    print(f"🔧 Training KNN classifier (k={k})...")
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("✅ KNN classifier training complete!")

    return clf, y_pred