from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd

try:
    from classification.classificator import load_model, labels, feature_extraction, get_dataset
except ModuleNotFoundError:
    from classificator import load_model, labels, feature_extraction, get_dataset


def plot_confusion_matrix(y_model, y_true, labels):
    cm = confusion_matrix(y_true,y_model,normalize='true')
    _, ax = plt.subplots(figsize=(7,7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=False)
    plt.title("Confusion matrix")
    plt.grid(False)
    plt.show()


def evaluate():
    model, tfidf_vectorizer = load_model()
    # Getting dfs
    test_data, test_labels = get_dataset(train=False)
    # Extracting features
    test_features = feature_extraction(tfidf_vectorizer, test_data, fit=False)
    y_preds = model.predict(test_features)
    plot_confusion_matrix(y_preds, test_labels, labels.values())


def classify_phrase(phrase: str):
    model, tfidf_vectorizer = load_model("classification")
    
    # Converting string to compatible format
    phrase = pd.Series(phrase)
    
    prediction = model.predict(feature_extraction(tfidf_vectorizer, phrase, False))
    return labels[prediction[0]]

if __name__ == "__main__":
    evaluate()
