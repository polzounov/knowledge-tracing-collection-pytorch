import os
import json
import pickle
import argparse

import torch
import optuna
from torch.utils.data import DataLoader, random_split
from torch.optim import SGD, Adam


# from data_loaders.assist2015 import ASSIST2015
# from data_loaders.algebra2005 import Algebra2005
# from data_loaders.statics2011 import Statics2011
from data_loaders.assist2009 import ASSIST2009
from data_loaders.simulated import Simulated

from models.dkt import DKT
from models.utils import collate_fn


DATASETS = {
    "ASSIST2009": ASSIST2009,
    "simulated": Simulated,
    # "ASSIST2015": ASSIST2015,
    # "Algebra2005": Algebra2005,
    # "Statics2011": Statics2011,
}


def objective(trial):
    model_name = "dkt"  # You can also suggest models if you wish
    dataset_name = "simulated"  # Similarly, you can suggest datasets

    train_config = {
        "batch_size": 256,
        "num_epochs": 20,
        "seq_len": 50,
        "train_ratio": 0.8,
    }
    model_config = {"emb_size": 50, "hidden_size": 400}
    return train(model_name, dataset_name, trial, train_config, model_config)


def main(model_name, dataset_name):
    train_config = {
        "batch_size": 256,
        "num_epochs": 20,
        "seq_len": 50,
        "train_ratio": 0.8,
    }
    model_config = {"emb_size": 50, "hidden_size": 400}

    # Initialize Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("Value: ", trial.value)
    print("Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    train(model_name, dataset_name, trial, train_config, model_config)


def train(model_name, dataset_name, trial, train_config, model_config):
    num_epochs = train_config["num_epochs"]
    train_ratio = train_config["train_ratio"]
    seq_len = train_config["seq_len"]

    # batch_size = train_config["batch_size"]
    # optimizer = train_config["optimizer"]  # can be [sgd, adam]
    # learning_rate = train_config["learning_rate"]

    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    optimizer = trial.suggest_categorical("optimizer", ["sgd", "adam"])

    ds_loader = DATASETS[dataset_name]
    dataset = ds_loader(seq_len)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model = DKT(dataset.num_q, **model_config).to(device)

    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    if os.path.exists(os.path.join(dataset.dataset_dir, "train_indices.pkl")):
        with open(os.path.join(dataset.dataset_dir, "train_indices.pkl"), "rb") as f:
            train_dataset.indices = pickle.load(f)
        with open(os.path.join(dataset.dataset_dir, "test_indices.pkl"), "rb") as f:
            test_dataset.indices = pickle.load(f)
    else:
        with open(os.path.join(dataset.dataset_dir, "train_indices.pkl"), "wb") as f:
            pickle.dump(train_dataset.indices, f)
        with open(os.path.join(dataset.dataset_dir, "test_indices.pkl"), "wb") as f:
            pickle.dump(test_dataset.indices, f)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=test_size, shuffle=True, collate_fn=collate_fn
    )

    if optimizer == "sgd":
        opt = SGD(model.parameters(), learning_rate, momentum=0.9)
    elif optimizer == "adam":
        opt = Adam(model.parameters(), learning_rate)

    aucs, loss_means = model.train_model(
        train_loader,
        test_loader,
        num_epochs,
        opt,
        ckpt_path=f"ckpts/dkt/{dataset_name}",
    )

    # Return the maximum AUC (objective)
    return max(aucs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        type=str,
        default="dkt",
        help="The name of the model to train. \
            The possible models are in [dkt, dkt+, dkvmn, sakt, gkt]. \
            The default model is dkt.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="ASSIST2009",
        help="The name of the dataset to use in training.",
        choices=DATASETS.keys(),
    )
    args = parser.parse_args()

    main(args.model_name, args.dataset_name)
