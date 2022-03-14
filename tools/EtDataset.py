import random
import logging
import torch
from torch.utils.data import Dataset
from collections import defaultdict


class EtDataset(Dataset):
    def __init__(self,
                 dataset,
                 type_list,
                 config,
                 triplets,
                 entity_padding,
                 relation_padding,
                 rng=None,
                 ):
        self.config = config
        self.dataset = dataset
        self.triplets = triplets
        self.type_list = type_list
        self.entity_padding = entity_padding
        self.relation_padding = relation_padding
        self.rng = rng if rng else random.Random()
        self.dataset_e2ts = defaultdict(set)
        print("------ building et dataset negative search tree ------")
        for e, t in self.dataset:
            self.dataset_e2ts[e].add(t)
        self.dataset_entities = list(self.dataset_e2ts.keys())
        self.total = len(self.dataset)
        self.dataset_e2rts = defaultdict(list)
        for h, r, t in self.triplets:
            self.dataset_e2rts[t].append((h, r, 1))
            self.dataset_e2rts[h].append((t, r, -1))
        print("------ build over ------")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if index > self.total:
            logging.warning(f"index: {index} > {self.total}")
            index = index % self.total
        e, t = self.dataset[index]

        # neighbors
        if self.config['relation_padding']:
            nbs = self.rng.choices(self.dataset_e2rts[e], k=self.config['S_triplet_max'] - 1)
            nbs += [(self.entity_padding, self.relation_padding, 1)]
        else:
            nbs = self.rng.choices(self.dataset_e2rts[e], k=self.config['S_triplet_max'])
        nbes = torch.tensor([e for e, r, f in nbs])
        nbrs = torch.tensor([r for e, r, f in nbs])
        nbfs = torch.tensor([f for e, r, f in nbs])

        # positive types
        pts = torch.tensor(self.rng.choices(list(self.dataset_e2ts[e]), k=self.config['L_type_max']))

        # negative types
        nts = []
        for _ in range(self.config['L_type_max']):
            t = self.rng.choice(self.type_list)
            while t in self.dataset_e2ts[e]:
                t = self.rng.choice(self.type_list)
            nts.append(t)
        nts = torch.tensor(nts)
        return torch.tensor(e), nbes, nbrs, nbfs, pts, nts
