import os
import nltk
import pickle
import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer


nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('portuguese')
tfidf_vectorizer = TfidfVectorizer(max_features=10_000)
labels = {
        0: "tristeza", 
        1: "felicidade", 
        2: "raiva"
}


def feature_extraction(vectorizer: TfidfVectorizer, data: pd.Series, train=True):
    """Extract features from data

    :param data: Data in text format
    :return: Extracted features
    """
    if train:
        return vectorizer.fit_transform(data)
    return vectorizer.transform(data)


def get_dataset(train=True):
    """Load training and testing datasets

    :return: training and testing datasets, divided by text and label
    """
    # Getting file path and remove classification folder, if necessary
    files_path = os.getcwd().replace("\\classification", "")
    files_path = os.path.join(files_path, "data")
    
    # Loading DataFrames
    if train:
        data = pd.read_csv(os.path.join(files_path, "pt_training.csv"))
    else:
        data = pd.read_csv(os.path.join(files_path, "pt_test.csv"))
    
    # Balancing amount of rows per label
    row_amount = min(data["label"].value_counts())
    data = data.groupby("label").sample(row_amount)
    
    return data["text"], data["label"]


def save_model(model, file_path="model.sav"):
    """Save model in pickle format

    :param model: NB model
    :param file_path: Path to model, defaults to "model.sav"
    """
    pickle.dump(model, open(file_path, "wb"))


def load_model(file_path="model.sav"):
    """Load model object from pickle

    :param file_path: Path to model, defaults to "model.sav"
    :return: Model object
    """
    loaded_model = pickle.load(open(file_path, 'rb'))
    return loaded_model


def main():
    """Create, train and save model for emotion classification
    """
    # Getting dfs
    train_data, train_labels = get_dataset()
    # Extracting features
    train_features = feature_extraction(tfidf_vectorizer, train_data)
    
    # Creating model and training
    model = SVC()
    print("Training Model")
    model.fit(train_features, train_labels)
    print("Training Complete")
    
    save_model(model)


if __name__ == "__main__":
    main()
