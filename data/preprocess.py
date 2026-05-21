import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split


class Processor:
    def __init__(self, root_dir="./", output_dir="aggregated_csvs", random_seed=4221, n_env_cols=424, n_rxn_cols=1715):
        self.root_dir = root_dir
        self.output_dir = os.path.join(root_dir, output_dir)
        self.random_seed = random_seed
        self.n_env_cols = n_env_cols
        self.n_rxn_cols = n_rxn_cols
        os.makedirs(self.output_dir, exist_ok=True)

        self.categories = [
            "comp_envs", "fcoop_envs",
            "comp_INTrxnfluxes", "fcoop_INTrxnfluxes",
            "comp_rhsfluxes", "fcoop_rhsfluxes",
            "comp_TRANrxnfluxes", "fcoop_TRANrxnfluxes"]
        self.all_data = {cat: [] for cat in self.categories}

    def load_array(self, mat_path):
        try:
            data = loadmat(mat_path)
            key = os.path.splitext(os.path.basename(mat_path))[0]
            arr = data[key]
            if hasattr(arr, "toarray"):
                arr = arr.toarray()
            return np.array(arr)
        except Exception as e:
            print(f"Could not read {mat_path}: {e}")
            return None

    def pad_arrays(self, arr_list):
        if not arr_list:
            return np.empty((0, 0))
        max_rows = max(a.shape[0] for a in arr_list)
        max_cols = max(a.shape[1] for a in arr_list)
        padded = []
        for a in arr_list:
            pad_r = max_rows - a.shape[0]
            pad_c = max_cols - a.shape[1]
            a_padded = np.pad(a, ((0, pad_r), (0, pad_c)), mode='constant', constant_values=0)
            padded.append(a_padded)
        return np.concatenate(padded, axis=0)

    def aggregate_mat_files(self):
        for folder in sorted(os.listdir(self.root_dir)):
            subdir = os.path.join(self.root_dir, folder)
            if not os.path.isdir(subdir):
                continue
            if not folder.startswith("AG_"):
                continue
            print(f"Processing {folder}...")
            for cat in self.categories:
                matches = [f for f in os.listdir(subdir) if f.startswith(cat) and f.endswith(".mat")]
                for fname in matches:
                    mat_path = os.path.join(subdir, fname)
                    arr = self.load_array(mat_path)
                    if arr is not None:
                        self.all_data[cat].append(arr)
        for cat, arr_list in self.all_data.items():
            if not arr_list:
                continue
            combined = self.pad_arrays(arr_list)
            save_path = os.path.join(self.output_dir, f"{cat}.csv")
            pd.DataFrame(combined).to_csv(save_path, index=False, header=False)
            print(f"Saved {save_path}")

    def clean_csvs(self):
        for fname in sorted(os.listdir(self.output_dir)):
            if not fname.endswith(".csv"):
                continue
            csv_path = os.path.join(self.output_dir, fname)
            print(f"Cleaning {fname}...")
            df = pd.read_csv(csv_path, header=None)
            if df.shape[0] > 2:
                df = df.iloc[2:].reset_index(drop=True)
            # Remove all-zero rows
            arr = df.to_numpy()
            nonzero_mask = np.any(arr != 0, axis=1)
            df = df[nonzero_mask].reset_index(drop=True)
            # Assign columns
            if "envs" in fname:
                num_cols = min(df.shape[1], self.n_env_cols)
                cols = [f"c{i+1}" for i in range(num_cols)]
            else:
                num_cols = min(df.shape[1], self.n_rxn_cols)
                cols = [f"r{i+1}" for i in range(num_cols)]
            df = df.iloc[:, :num_cols]
            df.columns = cols
            # Labels
            if fname.startswith("fcoop_"):
                df["label"] = 1
            elif fname.startswith("comp_"):
                df["label"] = 0
            else:
                df["label"] = np.nan
            df.to_csv(csv_path, index=False)
            print(f"Saved cleaned {fname}")

    def split_datasets(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        split_dir = os.path.join(self.output_dir, "splits")
        os.makedirs(split_dir, exist_ok=True)
        data_groups = {
            "envs": [],
            "rhsfluxes": [],
            "INTrxnfluxes": [],
            "TRANrxnfluxes": []}
        for fname in sorted(os.listdir(self.output_dir)):
            if not fname.endswith(".csv"):
                continue
            fpath = os.path.join(self.output_dir,fname)
            if "envs" in fname:
                data_groups["envs"].append(fpath)
            elif "rhsfluxes" in fname:
                data_groups["rhsfluxes"].append(fpath)
            elif "INTrxnfluxes" in fname:
                data_groups["INTrxnfluxes"].append(fpath)
            elif "TRANrxnfluxes" in fname:
                data_groups["TRANrxnfluxes"].append(fpath)
        all_flux_df = []
        for group_name, file_list in data_groups.items():
            if not file_list:
                continue
            print(f"\nSplitting {group_name}")
            dfs = [pd.read_csv(f) for f in file_list]
            df = pd.concat(dfs, ignore_index=True)
            X = df.drop(columns=["label"])
            y = df["label"]
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(1 - train_ratio), stratify=y, random_state=self.random_seed)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_ratio / (test_ratio + val_ratio), stratify=y_temp, random_state=self.random_seed)
            for name, X_part, y_part in [
                ("train", X_train, y_train),
                ("val", X_val, y_val),
                ("test", X_test, y_test)]:
                X_part.to_csv(os.path.join(split_dir, f"{group_name}_X_{name}.csv"), index=False)
                y_part.to_csv(os.path.join(split_dir, f"{group_name}_y_{name}.csv"), index=False)
                print(f"{group_name} {name}: "f"{X_part.shape}")
            if group_name != "envs":
                all_flux_df.append(df)
        # Combined flux dataset
        if all_flux_df:
            all_flux_df = pd.concat(all_flux_df,ignore_index=True)
            X = all_flux_df.drop(columns=["label"])
            y = all_flux_df["label"]
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(1 - train_ratio), stratify=y, random_state=self.random_seed)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_ratio / (test_ratio + val_ratio), stratify=y_temp, random_state=self.random_seed)
            for name, X_part, y_part in [
                ("train", X_train, y_train),
                ("val", X_val, y_val),
                ("test", X_test, y_test)]:
                  
                X_part.to_csv(os.path.join(split_dir, f"all_fluxes_X_{name}.csv"),index=False)
                y_part.to_csv(os.path.join(split_dir, f"all_fluxes_y_{name}.csv"), index=False)

    def run(self):
        self.aggregate_mat_files()
        self.clean_csvs()
        self.split_datasets()


if __name__ == "__main__":
    processor = Processor(root_dir="./")
    processor.run()
  
