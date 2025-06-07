from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.dataset import load_lbp_dataset
from src.train import train_and_evaluate
from src.inference import load_model

def main():
    dataset_path = 'data/Aerial_Landscapes'
    classifier = 'knn'
    model_path = 'models/knn_model.pkl'
    save_model = True

    X, y, _ = load_lbp_dataset(dataset_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = load_model(model_path)
    if model is None:
        train_and_evaluate(X_train, X_test, y_train, y_test, classifier, save_model, model_path)
    else:
        print("Model already exists. Run evaluation manually if needed.")

if __name__ == "__main__":
    main()
