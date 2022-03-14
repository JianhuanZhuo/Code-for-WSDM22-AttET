import random
import logging
import torch
from torch.utils.data import Dataset
from collections import defaultdict


class PnDataset(Dataset):
    def __init__(self,
                 dataset,
                 entity_ids,
                 rng=None,
                 ):
        self.dataset = dataset
        self.entity_ids = entity_ids
        self.dataset_set = set(self.dataset)
        self.rng = rng if rng else random.Random()
        self.total = len(self.dataset)
        self.dataset_h_r_negative_tree = defaultdict(set)
        self.dataset_r_t_negative_tree = defaultdict(set)
        print("------ building ere dataset negative search tree ------")
        for h, r, t in self.dataset_set:
            self.dataset_h_r_negative_tree[(h, r)].add(t)
            self.dataset_r_t_negative_tree[(r, t)].add(h)
        print("------ build over ------")
        pass

    def __len__(self):
        return self.total

    def __getitem__(self, index):
        if index > self.total:
            logging.warning(f"index: {index} > {self.total}")
            index = index % self.total
        positive_triplet = self.dataset[index]

        # if self.negative_verify:
        if self.rng.randint(0, 1):
            r_t = self.rng.choice(self.entity_ids)
            while r_t in self.dataset_h_r_negative_tree[(positive_triplet[0], positive_triplet[1])]:
                r_t = self.rng.choice(self.entity_ids)
            negative_triplet = (positive_triplet[0], positive_triplet[1], r_t)
        else:
            r_h = self.rng.choice(self.entity_ids)
            while r_h in self.dataset_r_t_negative_tree[(positive_triplet[1], positive_triplet[2])]:
                r_h = self.rng.choice(self.entity_ids)
            negative_triplet = (r_h, positive_triplet[1], positive_triplet[2])

        return torch.tensor(positive_triplet), torch.tensor(negative_triplet)
