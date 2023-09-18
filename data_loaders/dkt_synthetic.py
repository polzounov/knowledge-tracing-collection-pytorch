import os
import pickle
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from models.utils import match_seq_len


DATASET_DIR = "datasets/dkt-synthetic/"


class DKTSynthetic(Dataset):
    def __init__(self, seq_len=None, dataset_dir=DATASET_DIR, **kwargs) -> None:
        super().__init__()

        self.dataset_dir = dataset_dir
        self.dataset_path = os.path.join(self.dataset_dir, "data.csv")

        self.q_seqs, self.r_seqs = self.preprocess()

        self.num_q = self.q_seqs[0].shape[0]
        self.num_u = self.len = len(self.q_seqs)

    def __getitem__(self, index):
        return self.q_seqs[index], self.r_seqs[index]

    def __len__(self):
        return self.len

    def preprocess(self):
        q_seqs = []
        r_seqs = []
        with open(self.dataset_path, "r") as f:
            for user_index, line in enumerate(f.readlines()):
                corrects = line.strip().split(",")
                corrects = np.array([int(i) for i in corrects])
                questions = np.array([i for i in range(len(corrects))])

                q_seqs.append(questions)
                r_seqs.append(corrects)

        return q_seqs, r_seqs
