import joblib
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from .utils import plot_metrics

def train_and_evaluate(X_train, X_test, y_train, y_test, classifier='knn', save_model=False, model_filename="model.pkl"):
    model = KNeighborsClassifier(n_neighbors=3) if classifier == 'knn' else SVC(kernel='linear')
    model.fit(X_train, y_train)

    if save_model:
        joblib.dump(model, model_filename)
        print(f"Model saved as {model_filename}")

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"\nResults for {classifier.upper()}:\nAccuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1-score: {f1:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    plot_metrics(acc, prec, rec, f1, classifier.upper())
