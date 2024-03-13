from tqdm import tqdm
tqdm.pandas(desc="Processando dados")
import pandas as pd
pd.options.mode.copy_on_write = True
from deep_translator import GoogleTranslator


class Translator:
    """Class to preprocess data to translate twitter dataset to portuguese
    """
    def __init__(self) -> None:
        # Creating Google Translator object
        self.translator = GoogleTranslator(source='auto', target='portuguese', service_url="https://translate.googleapis.com")

    def translate(self, phrase: str):
        """Translate a given phrase to portuguese

        :param phrase: Phrase to be translated
        :return: Translation result
        """
        return self.translator.translate(phrase)


def pre_process(dataframe: pd.DataFrame):
    """Preprocess dataframe data, removing NaN value and sentiments different from (sadness, joy, raiva).

    :param dataframe: Dataframe with tweets
    :return: Preprocessed dataframe
    """
    # Only keeping data matching desired sentiments
    dataframe = dataframe[dataframe["label"].isin([0, 1, 3])]
    # Changing label 3 to be 2 so labels are in [0, 1, 2] interval
    dataframe["label"] = dataframe["label"].replace(3, 2)
    # Removing Na values
    dataframe.dropna(inplace=True)
    return dataframe


def main():
    """Preprocess test and training datasets and save
    """
    translator = Translator()
    for file_name in ["original_test", "original_training"]:
        try:
            df = pd.read_csv(f"{file_name}.csv")
        except FileNotFoundError:
            df = pd.read_csv(f"data/{file_name}.csv")
        df = pre_process(df)
        df["text"] = df["text"].progress_apply(translator.translate)
        df.to_csv(f"pt_{file_name.replace('original_', '')}.csv", sep=",", encoding="utf-8", index=False)


if __name__ == "__main__":
    main()
