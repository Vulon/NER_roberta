import nltk
import os
import dvc.api

params = dvc.api.params_show()

def download_nltk(nltk_folder: str):
    nltk.download('punkt', download_dir=nltk_folder)
    nltk.download('averaged_perceptron_tagger', download_dir=nltk_folder)


if __name__ == "__main__":

    nltk_folder = os.path.join(params["SCORE"]["PACKAGE_FOLDER"], "NLTK")
    download_nltk(nltk_folder)