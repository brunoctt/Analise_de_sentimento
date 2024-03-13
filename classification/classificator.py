import os
import nltk
import pickle
import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('portuguese')
labels = {
        0: "tristeza", 
        1: "felicidade", 
        2: "raiva"
}


def feature_extraction(vectorizer: TfidfVectorizer, data: pd.Series, fit=True):
    """Extract features from data

    :param vectorizer: Vectorizer object
    :param data: Data in text format
    :param train: If vectorizer is to be fitted or not, defaults to True
    :return: _description_
    """
    if fit:
        return vectorizer.fit_transform(data)
    return vectorizer.transform(data)


def get_dataset(train: bool = True):
    """Load training and testing datasets

    :param train: If training dataset or test dataset is to be loaded, defaults to True
    :return: training and testing datasets, divided by text and label
    """
    # Getting file path and remove classification folder, if necessary
    files_path = os.getcwd().replace("\\classification", "")
    files_path = os.path.join(files_path, "data")
    
    # Loading DataFrame
    data = pd.read_csv(os.path.join(files_path, "prompt_generated.csv"))
    
    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(data["text"], data["label"], test_size=0.1)
    
    if train:
        return X_train, y_train
    else:
        return X_test, y_test


def save_model(model: SVC, feat_extractor: TfidfVectorizer, file_path=""):
    """Save model in pickle format

    :param model: 
    :param file_path: Path to model, defaults to "model.sav"
    """
    with open(os.path.join(file_path, 'model.pkl'),'wb') as f:
        pickle.dump(model,f)
    with open(os.path.join(file_path, 'feat_ext.pkl'),'wb') as f:
        pickle.dump(feat_extractor,f)


def load_model(file_path=""):
    """Load model and  object from pickle

    :param file_path: Path to objects, defaults to ""
    :return: SVC object and TfidfVectorizer object
    """
    with open(os.path.join(file_path, 'model.pkl'), 'rb') as f:
        loaded_model = pickle.load(f)
    with open(os.path.join(file_path, 'feat_ext.pkl'), 'rb') as f:
        loaded_feat_ext = pickle.load(f)
    return loaded_model, loaded_feat_ext


def extract_train_features(tfidf_vectorizer):
    """Load and extract train features

    :return: Training features and respective labels
    """
    # Getting dfs
    train_data, train_labels = get_dataset()
    # Extracting features
    train_features = feature_extraction(tfidf_vectorizer, train_data)
    return train_features, train_labels


def main():
    """Create, train and save model for emotion classification
    """
    tfidf_vectorizer = TfidfVectorizer(max_features=10_000)
    train_features, train_labels = extract_train_features(tfidf_vectorizer)
    # Creating model and training
    model = SVC()
    print("Training Model")
    model.fit(train_features, train_labels)
    print("Training Complete")
    
    save_model(model, tfidf_vectorizer)


if __name__ == "__main__":
    main()
