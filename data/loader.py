from huggingface_hub import hf_hub_download
import pandas as pd

class Loader:
    def __init__(self, repo_id, subfolder, prefix, repo_type="dataset"):
        self.repo_id = repo_id
        self.subfolder = subfolder
        self.prefix = prefix
        self.repo_type = repo_type

    def _file_id(self, split, kind):
        return f"{self.subfolder}/{self.prefix}_{kind}_{split}.csv"

    def load_split(self, split):
        X_id = self._file_id(split, "X")
        y_id = self._file_id(split, "y")
        X = pd.read_csv(
            hf_hub_download(
                repo_id=self.repo_id,
                filename=X_id,
                repo_type=self.repo_type
            )
        )
        y = pd.read_csv(
            hf_hub_download(
                repo_id=self.repo_id,
                filename=y_id,
                repo_type=self.repo_type
            )
        ).values.ravel()

        return X, y

    def load_all(self):
        return {
            "train": self.load_split("train"),
            "val": self.load_split("val"),
            "test": self.load_split("test")
        }
