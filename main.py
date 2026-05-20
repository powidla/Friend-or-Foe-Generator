from data.loader import Loader
from train.train_dt import Trainer
import argparse


def main():
    parser = argparse.ArgumentParser(description="Streamline loading and decision tree training")
    parser.add_argument("--repo_id", type=str, default="powidla/DMP", help="HF repository to load from")
    parser.add_argument("--dataset", type=str, default="AG_M582andM599/splits", help="Dataset with microbes identified; splits is where train/val/test live")
    parser.add_argument("--prefix", type=str, default="envs", help="Dataset prefix (envs, all_fluxes, INTrxnfluxes, TRANrxnfluxes, rhsfluxes)")
    parser.add_argument("--max_depth", type=int, default=2, help="depth of DT")
    parser.add_argument("--top_k", type=int, default=5, help="topK features to pop")
    args = parser.parse_args()

    # loading data
    loader = Loader(repo_id=args.repo_id, subfolder=args.dataset, prefix=args.prefix)
    X_train, y_train = loader.load_split("train")
    X_val, y_val = loader.load_split("val")
    X_test, y_test = loader.load_split("test")

    # train a tree
    trainer = Trainer(max_depth=args.max_depth)
    trainer.fit(X_train, y_train)
    trainer.evaluate_all(X_val, y_val, X_test, y_test)
    trainer.plot_tree()
    trainer.plot_feature_importance(top_k=args.top_k)


if __name__ == "__main__":
    main()
  
