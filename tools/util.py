import os
import time

import numpy as np
import scipy.sparse as sp
from collections import OrderedDict
import random
import json
import yaml

import torch


def grad_sum_matrix(idx):
    """
    This function helps us to vectorize the training process.
    So that for each batch-size training samples, we could 
    calculate in parallel.
    """
    # unique_idx: unique entity_id; idx_inverse: index for each entity in idx in the unique_idx
    unique_idx, idx_inverse = np.unique(idx, return_inverse=True)
    # Calculate the number of entities that are nedded for updating.(including duplicates)
    sz = len(idx_inverse)
    # generate a coefficient matrix. M.shape = (num_of_unique_entities, num_of_samples)
    # M = [[1,1,0,1,0],
    #      [0,0,1,0,1],
    #      [1,1,1,1,1]]
    # This means the 1-st sample is used to update the 0-th and 2-ed entity; 
    M = sp.coo_matrix((np.ones(sz), (idx_inverse, np.arange(sz)))).tocsr()  # M.shape = (num_of_unique_idx, tot_sample)
    # normalize summation matrix so that each row sums to one
    tot_update_time = np.array(M.sum(axis=1))  # shape = (num_of_unique_entities, ) 
    return unique_idx, M, tot_update_time


def normalize(M, idx=None):
    """
    Used as a tool function to normalize the matrix M by using column-wise L-2 norm.
    If idx is not None, then only the row specified in idx would be normalized.
    """
    if idx is None:
        M /= np.sqrt(np.sum(M ** 2, axis=1))[:, np.newaxis]
    else:
        nrm = np.sqrt(np.sum(M[idx, :] ** 2, axis=1))[:, np.newaxis]
        M[idx, :] /= nrm
    return M


def loadVectors(path="output/entityVector.txt"):
    """
    Used to load the vector specified in the path.
    """
    vectorDict = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            k, v = line.strip().split("\t")
            v = np.array(eval(v))
            vectorDict[k] = v
    return vectorDict


def encode2id(triplet, *ids):
    """
    Traslate the triplet from string format to unique integer format. 
    (Unkonwn entities/relations/types are ignored.)
    """
    data = []
    for pairs in triplet:
        new_pairs = []
        try:
            for i in range(len(pairs)):
                new_pairs.append(ids[i][pairs[i]])
            data.append(tuple(new_pairs))
        except KeyError:
            pass

    return data


def init_nunif(sz):
    """
    Normalized uniform initialization

    See Glorot X., Bengio Y.: "Understanding the difficulty of training
    deep feedforward neural networks". AISTATS, 2010
    """
    bnd = np.sqrt(6) / np.sqrt(sz[0] + sz[1])
    p = np.random.uniform(low=-bnd, high=bnd, size=sz)
    return np.squeeze(p)


def loadEntityId(path):
    """
    Used to load the entity unique integer ID specified in the path.
    """
    d = OrderedDict()
    with open(path, 'r') as f:
        for line in f.readlines():
            e, id = line.strip().split("\t")
            d[e] = int(id)
    return d


def loadRelationId(path):
    """
    Used to load the relation unique integer ID specified in the path.
    """
    d = OrderedDict()
    with open(path, 'r') as f:
        for line in f.readlines():
            r, id = line.strip().split("\t")
            d[r] = int(id)
    return d


def loadTypeId(path):
    """
    Used to load the type unique integer ID specified in the path.
    """
    d = OrderedDict()
    with open(path, 'r') as f:
        for line in f.readlines():
            r, id = line.strip().split("\t")
            d[r] = int(id)
    return d


def loadTriplet(path="data/freebase_mtr100_mte100-train.txt"):
    """
    Used to load the triplets specified in the path.
    """
    triplet = []
    with open(path, 'r') as f:
        for line in f.readlines():
            h, l, t = line.strip().split("\t")
            triplet.append((h, l, t))
    return triplet


def loadEntity2Type(path="data/FB15k_Entity_Type_train.txt"):
    """
    Used to load the entity-type mapping data specified in the path.
    """
    triplet = []
    with open(path, 'r') as f:
        for line in f.readlines():
            h, t = line.strip().split("\t")
            triplet.append((h, t))
    return triplet


def getRandomObj(entity_, sourse, exclude_list=None):
    """
    Used to generate a ramdom negative sample(filtered).
    """
    random_entity_ = entity_
    if not exclude_list:
        exclude_list = {entity_}

    while (random_entity_ in exclude_list):
        random_entity_ = random.sample(sourse, 1)[0]
    return random_entity_


class Config:
    def __init__(self, dicts, keys, maps):
        self.dicts = dicts
        self.maps = maps
        self.keys = sorted(keys)

        if 'seed' not in dicts:
            dicts['origin_seed'] = "undefined"
            dicts['seed'] = random.randint(0, 100)
        else:
            dicts['origin_seed'] = dicts['seed']
            if not isinstance(dicts['seed'], int):
                dicts['seed'] = random.randint(0, 100)

        # set timestamp mask
        self['timestamp_mask'] = time.strftime("%m-%d-%H-%M-%S", time.localtime())

    def __getitem__(self, item):
        if item in self.dicts:
            return self.dicts[item]
        elif item in self.maps:
            return self.dicts[self.maps[item]]
        else:
            return None

    def __contains__(self, item):
        if item in self.dicts:
            return True
        elif item in self.maps:
            return True
        return False

    def __setitem__(self, key, value):
        if key in self.dicts:
            self.dicts[key] = value
        elif key in self.maps:
            self.dicts[self.maps[key]] = value
        else:
            self.dicts[key] = value

    def postfix(self):
        return "-".join([
            k + str(self[k] if not isinstance(self[k], bool) else "Y" if self[k] else "N")
            for k in self.keys
        ])

    def __str__(self):
        return json.dumps({
            "dicts": self.dicts,
            "keys": self.keys,
            "maps": self.maps,
        }, sort_keys=True, indent=4)

    def clone(self):
        return Config(self.dicts.copy(), self.keys.copy(), self.maps.copy())


def load_config(path):
    """
    Get and parse the configeration file in `path`
    """
    if path.endswith("yaml") or path.endswith("yml"):
        with open(path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            res = {}
            res.update(config['other'])
            res.update(config['main'])
            return Config(res, set(config['main'].keys()), config['maps'])
    raise Exception(f"config path exception: {path}")


def load_specific_config(path):
    if path.endswith("yaml") or path.endswith("yml"):
        with open("config.yaml", "r") as cm, open(path, "r") as sf:
            sf_config = yaml.load(sf, Loader=yaml.FullLoader)
            cm_config = yaml.load(cm, Loader=yaml.FullLoader)
            res = {}
            res.update(cm_config['other'])
            res.update(cm_config['main'])
            res.update(sf_config.get('other', {}))
            res.update(sf_config.get('main', {}))
            return Config(res, set(sf_config['main'].keys()), cm_config['maps'])
    raise Exception(f"config path exception: {path}")


def normalize_emb(emb):
    emb.weight.data.copy_(torch.renorm(emb.weight.detach().cpu(),
                                       p=2,
                                       dim=0,
                                       maxnorm=1))
    # torch.nn.utils.weight_norm(emb, name='weight', dim=0)
    pass


def bbmm(x, y):
    """
    :param x: [a,b,c,d]
    :param y: [a,b,d,e]
    :return: [a,b,c,e]
    """
    assert len(x.shape) == 4
    assert len(y.shape) == 4
    assert x.shape[:2] == y.shape[:2]
    assert x.shape[3] == y.shape[2]
    xdim = x.shape
    ydim = y.shape
    x = x.reshape([xdim[0] * xdim[1], xdim[2], xdim[3]])
    y = y.reshape([ydim[0] * ydim[1], ydim[2], ydim[3]])
    res = torch.bmm(x, y)
    return res.reshape([xdim[0], xdim[1], res.shape[1], res.shape[2]])
