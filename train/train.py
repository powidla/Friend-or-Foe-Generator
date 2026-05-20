import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics 

class Trainer:
    def __init__(self, max_depth=2, class_weight="balanced", random_state=4221):
        self.model = DecisionTreeClassifier(max_depth=max_depth, class_weight=class_weight, random_state=random_state)
        self.max_depth = max_depth
        self.is_trained = False

    def fit(self, X_train, y_train):
        print(f"\n Training ...")
        self.model.fit(X_train, y_train)
        self.feature_names = X_train.columns
        self.is_trained = True

    def evaluate(self, X, y, split_name="Test"):
        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)[:, 1]

        metrics = {
            "Accuracy": sklearn.metrics.accuracy_score(y, y_pred),
            "Precision": sklearn.metrics.precision_score(y, y_pred),
            "Recall": sklearn.metrics.recall_score(y, y_pred),
            "F1": sklearn.metrics.f1_score(y, y_pred),
            "AUC": sklearn.metrics.roc_auc_score(y, y_prob),
        }
        print(f"\n {split_name} performance:")
        for k, v in metrics.items():
            print(f"{k:10s}: {v:.3f}")
        return metrics

    def evaluate_all(self, X_val, y_val, X_test, y_test):
        self.evaluate(X_val, y_val, "Validation")
        self.evaluate(X_test, y_test, "Test")

        print("\n Classification report (Test):")
        print(classification_report(y_test, self.model.predict(X_test)))

    def feature_importance(self, top_k=5):
        assert self.is_trained, "Model must be trained first."

        importances = self.model.feature_importances_
        df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances
        })

        df = df[df["importance"] > 0]
        df = df.sort_values("importance", ascending=False)
        print(f"\n Top {top_k} feature importances:")
        print(df.head(top_k))
        return df

    def plot_feature_importance(self, top_k=5):
        df = self.feature_importance(top_k)

        plt.figure(figsize=(7, 4))
        plt.barh(df["feature"].head(top_k)[::-1], df["importance"].head(top_k)[::-1])
        plt.xlabel("Gini")
        plt.title(f"Top {top_k} Feature Importances")
        plt.tight_layout()
        plt.show()

    def plot_tree(self):
        assert self.is_trained, "Model must be trained first."

        plt.figure(figsize=(20, 10))
        plot_tree(
            self.model,
            feature_names=self.feature_names,
            class_names=["comp", "coop"],
            filled=True,
            rounded=True,
            fontsize=20)
        plt.title(f"Decision Tree (max_depth={self.max_depth})")
        plt.show()
      
