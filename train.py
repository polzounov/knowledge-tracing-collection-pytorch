import os
import argparse
import json
import pickle
import hashlib
from hashlib import sha256

import torch

from torch.utils.data import DataLoader, random_split
from torch.optim import SGD, Adam


# from data_loaders.assist2015 import ASSIST2015
# from data_loaders.algebra2005 import Algebra2005
# from data_loaders.statics2011 import Statics2011
from data_loaders.assist2009 import ASSIST2009
from data_loaders.simulated import Simulated
from data_loaders.dkt_synthetic import DKTSynthetic

from models.dkt import DKT
from models.utils import collate_fn


DATASETS = {
    "ASSIST2009": ASSIST2009,
    "simulated": Simulated,
    "dkt-synthetic": DKTSynthetic,
    # "ASSIST2015": ASSIST2015,
    # "Algebra2005": Algebra2005,
    # "Statics2011": Statics2011,
}


def check_if_stale(ckpt_path):
    return True
    # if not os.path.exists(os.path.join(ckpt_path, "shasum.txt")):
    #     return True

    # with open(os.path.join(ckpt_path, "shasum.txt"), "rb") as f:
    #     shasum = f.read().decode()

    # with open(os.path.join(dataset.dataset_dir, "data.csv"), "rb") as f:
    #     data = f.read()
    #     new_shasum = sha256(data).hexdigest()

    # stale = shasum != new_shasum
    # if stale:
    #     print("Dataset is stale. Rebuilding...")
    #     return True
    # else:
    #     return False


def save_data(ckpt_path, dataset_dir, train_dataset_indices, test_dataset_indices):
    with open(os.path.join(dataset_dir, "train_indices.pkl"), "wb") as f:
        pickle.dump(train_dataset_indices, f)
    with open(os.path.join(dataset.dataset_dir, "test_indices.pkl"), "wb") as f:
        pickle.dump(test_dataset_indices, f)

    # Save SHA256 hash of the dataset
    with open(os.path.join(dataset_dir, "data.csv"), "rb") as f:
        data = f.read()
        shasum = sha256(data).hexdigest()
    with open(os.path.join(dataset.dataset_dir, "shasum.txt"), "wb") as f:
        f.write(shasum.encode())


def load_data(ckpt_path, dataset_dir):
    with open(os.path.join(dataset_dir, "train_indices.pkl"), "rb") as f:
        train_dataset_indices = pickle.load(f)
    with open(os.path.join(dataset_dir, "test_indices.pkl"), "rb") as f:
        test_dataset_indices = pickle.load(f)

    return train_dataset_indices, test_dataset_indices


def main(model_name, dataset_name, use_skill_id, dataset_path=None):
    if not os.path.isdir("ckpts"):
        os.mkdir("ckpts")

    ckpt_path = os.path.join("ckpts", model_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    ckpt_path = os.path.join(ckpt_path, dataset_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    with open("config.json") as f:
        config = json.load(f)
        model_config = config[model_name]
        train_config = config["train_config"]

    batch_size = train_config["batch_size"]
    num_epochs = train_config["num_epochs"]
    train_ratio = train_config["train_ratio"]
    learning_rate = train_config["learning_rate"]
    optimizer = train_config["optimizer"]  # can be [sgd, adam]
    seq_len = train_config["seq_len"]

    ds_loader = DATASETS[dataset_name]
    dataset = ds_loader(seq_len, dataset_path=dataset_path)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    with open(os.path.join(ckpt_path, "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=4)
    with open(os.path.join(ckpt_path, "train_config.json"), "w") as f:
        json.dump(train_config, f, indent=4)

    model = DKT(dataset.num_q, **model_config).to(device)

    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # if os.path.exists(os.path.join(dataset.dataset_dir, "train_indices.pkl")):
    #     if check_if_stale(ckpt_path):
    #         save_data(
    #             ckpt_path,
    #             dataset.dataset_dir,
    #             train_dataset.indices,
    #             test_dataset.indices,
    #         )
    #     else:
    #         loaded_data = load_data(ckpt_path, dataset.dataset_dir)
    #         train_dataset.indices, test_dataset.indices = loaded_data

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=test_size, shuffle=True, collate_fn=collate_fn)

    if optimizer == "sgd":
        opt = SGD(model.parameters(), learning_rate, momentum=0.9)
    elif optimizer == "adam":
        opt = Adam(model.parameters(), learning_rate)

    aucs, loss_means = model.train_model(train_loader, test_loader, num_epochs, opt, ckpt_path)

    with open(os.path.join(ckpt_path, "aucs.pkl"), "wb") as f:
        pickle.dump(aucs, f)
    with open(os.path.join(ckpt_path, "loss_means.pkl"), "wb") as f:
        pickle.dump(loss_means, f)

    return aucs, loss_means


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
    parser.add_argument(
        "--skill-id",
        action="store_true",
        help=(
            "If true, use skill_name instead of problem_id "
            "\nNot sure why were note using skill_name instead of skill_id but: \n"
            "https://github.com/hcnoh/knowledge-tracing-collection-pytorch/commit/d47f9900335f9ef4c78a0c6ef98de14704f7a039"
        ),
    )
    # optional dataset path
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="The path to the CSV that you want to use.",
    )
    args = parser.parse_args()

    main(args.model_name, args.dataset_name, args.skill_id, args.dataset_path)
