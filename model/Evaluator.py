import pickle
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm


class Evaluator(object):
    def __init__(self, test_triplet, logger=None):
        self.logger = logger
        self.xs = test_triplet
        self.tot = len(test_triplet)
        self.pos = None
        self.fpos = None

    def __call__(self, model, epoch):
        raise NotImplementedError

    def positions(self, mdl):
        raise NotImplementedError

    def p_ranking_scores(self, pos, fpos, epoch, txt):
        rpos = [p for k in pos.keys() for p in pos[k]]
        frpos = [p for k in fpos.keys() for p in fpos[k]]
        fmrr = self._print_pos(
            np.array(rpos),
            np.array(frpos),
            epoch, txt)
        return fmrr

    def _print_pos(self, pos, fpos, epoch, txt):
        mrr, mean_pos, hits = self.compute_scores(pos)
        fmrr, fmean_pos, fhits = self.compute_scores(fpos)
        if self.logger:
            self.logger.info(
                f"[{epoch: 3d}] {txt}: MRR = {mrr:.2f}/{fmrr:.2f}, "
                f"Mean Rank = {mean_pos:.2f}/{fmean_pos:.2f}, "
                f"Hits@1 = {hits[0]:.2f}/{fhits[0]:.2f}, "
                f"Hits@3 = {hits[1]:.2f}/{fhits[1]:.2f}, "
                f"Hits@10 = {hits[2]:.2f}/{fhits[2]:.2f}"
            )
        else:
            print(
                f"[{epoch: 3d}] {txt}: MRR = {mrr:.2f}/{fmrr:.2f}, "
                f"Mean Rank = {mean_pos:.2f}/{fmean_pos:.2f}, "
                f"Hits@1 = {hits[0]:.2f}/{fhits[0]:.2f}, "
                f"Hits@3 = {hits[1]:.2f}/{fhits[1]:.2f}, "
                f"Hits@10 = {hits[2]:.2f}/{fhits[2]:.2f}"
            )
        return fmrr, fmean_pos, fhits

    def compute_scores(self, pos, hits=None):
        if hits is None:
            hits = [1, 3, 10]
        mrr = torch.mean(1.0 / pos.float())
        mean_pos = torch.mean(pos.float())
        hits_results = []
        for h in range(0, len(hits)):
            k = torch.mean((pos <= hits[h]).float())
            k2 = k.sum()
            hits_results.append(k2 * 100)
        return mrr, mean_pos, hits_results

    def save_prediction(self, path="output/pos", fpath="output/fpos_e2t_trt"):
        with open(path, 'wb') as f:
            pickle.dump(self.pos, f)
        with open(fpath, 'wb') as f:
            pickle.dump(self.fpos, f)

    def load_prediction(self, path="output/pos", fpath="output/fpos_e2t_trt"):
        with open(path, 'rb') as f:
            self.pos = pickle.load(f)

        with open(fpath, 'rb') as f:
            self.fpos = pickle.load(f)


class Type_Evaluator(Evaluator):

    def __init__(self, xs, true_tuples, eres, logger=None):
        super(Type_Evaluator, self).__init__(xs, logger)
        self.idx = defaultdict(list)  # defaultdict
        self.tt = defaultdict(list)  # true tuples
        self.sz = len(xs)

        for e, t in xs:
            self.idx[e].append((t))

        for e, t in true_tuples:
            self.tt[e].append(t)

        self.idx = dict(self.idx)
        self.tt = dict(self.tt)

        self.dataset_e2rts = defaultdict(list)
        for h, r, t in eres:
            self.dataset_e2rts[t].append((h, r, 1))
            self.dataset_e2rts[h].append((t, r, -1))
        print("------ build over ------")

    def __call__(self, model, epoch, path=None, fpath=None):
        if path and fpath:
            self.load_prediction(path, fpath)
        else:
            pos_v, fpos_v = self.positions(model)
            self.pos = pos_v
            self.fpos = fpos_v

        return self.et_ranking_scores(self.pos, self.fpos, epoch, 'VALID')

    def et_ranking_scores(self, pos, fpos, epoch, txt):
        tpos = [p for k in pos.keys() for p in pos[k]['type']]
        tfpos = [p for k in fpos.keys() for p in fpos[k]['type']]
        fmrr, fmean_pos, fhits = self._print_pos(
            torch.stack(tpos).float(),
            torch.stack(tfpos),
            epoch, txt)
        return fmrr, fmean_pos, fhits

    def positions(self, mdl):
        pos = {}  # Raw Positions
        fpos = {}  # Filtered Positions

        for e, ts in tqdm(self.idx.items(),
                          desc="evaluation",
                          bar_format="{desc}{percentage:3.0f}%|{bar:10}{r_bar}", ):
            ppos = {'type': []}
            pfpos = {'type': []}

            scores_origin = mdl.scores_et(e)
            for t in ts:
                scores_t = scores_origin.clone().detach()
                sortidx_t = torch.argsort(torch.argsort(scores_t))
                ppos['type'].append(sortidx_t[t] + 1)

                rm_idx = [i for i in self.tt[e] if i != t]
                scores_t[rm_idx] = np.Inf
                sortidx_t = torch.argsort(torch.argsort(scores_t))
                pfpos['type'].append(sortidx_t[t] + 1)

            pos[e] = ppos
            fpos[e] = pfpos

        return pos, fpos
