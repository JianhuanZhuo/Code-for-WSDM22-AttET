from collections import defaultdict
import random
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import MultiheadAttention


class E2T(nn.Module):
    def __init__(self,
                 entities_emb,
                 types_emb,
                 entity_dim,
                 relations_emb,
                 type_dim,
                 config,
                 eres,
                 entity_padding,
                 relation_padding,
                 margin=1.0,
                 norm=1,
                 ):
        super(E2T, self).__init__()
        self.entities_emb = entities_emb
        self.relations_emb = relations_emb
        self.types_emb = types_emb
        self.norm = norm
        self.config = config
        self.type_dim = type_dim
        if 'mlp' in config and config['mlp']:
            self.e2t = nn.Sequential(
                nn.Dropout(config['dropout'] if 'dropout' in config else 0),
                nn.Linear(entity_dim if config['disable_attention'] or config['disable_local'] else entity_dim * 2,
                          entity_dim * 2),
                nn.ReLU(),
                nn.Linear(entity_dim * 2,
                          type_dim),
                nn.ReLU(),
                nn.Linear(type_dim,
                          type_dim),
            )
        else:
            self.e2t = nn.Linear(
                entity_dim if config['disable_attention'] or config['disable_local'] else entity_dim * 2,
                type_dim)
        self.ee = nn.Sequential(
            nn.Linear(entity_dim, entity_dim),
            nn.ReLU(),
            nn.Linear(entity_dim, entity_dim),
            nn.ReLU(),
            nn.Linear(entity_dim, type_dim),
        )
        self.att_module = MultiheadAttention(embed_dim=self.type_dim, num_heads=1)
        self.entity_padding = entity_padding
        self.relation_padding = relation_padding

        self.dataset_e2rts = defaultdict(list)
        for h, r, t in eres:
            self.dataset_e2rts[t].append((h, r, 1))
            self.dataset_e2rts[h].append((t, r, -1))
        print("------ build over ------")

    def forward(self, es, nbes, nbrs, nbfs, pts, nts):
        """
        :param es: target entities in B shape
        :param nbes: neighbor entities in BxS shape
        :param nbrs: neighbor relations in BxS shape
        :param nbfs: neighbor relation direction flags in BxS shape
        :param pts: positive types in BxL
        :param nts: negative types in BxL
        :return: tuple of the model loss
        """
        assert es.size() == torch.Size([self.config['nbatch']])
        assert nbes.size() == torch.Size([self.config['nbatch'], self.config['S_triplet_max']])
        assert nbrs.size() == torch.Size([self.config['nbatch'], self.config['S_triplet_max']])
        assert nbfs.size() == torch.Size([self.config['nbatch'], self.config['S_triplet_max']])
        assert pts.size() == torch.Size([self.config['nbatch'], self.config['L_type_max']])
        assert nts.size() == torch.Size([self.config['nbatch'], self.config['L_type_max']])

        positive_distances = self._dist(es, nbes, nbrs, nbfs, pts)
        negative_distances = self._dist(es, nbes, nbrs, nbfs, nts)

        assert positive_distances.shape == torch.Size([self.config['nbatch'], self.config['L_type_max']])
        assert negative_distances.shape == torch.Size([self.config['nbatch'], self.config['L_type_max']])

        if self.config['np_compare'] == "max_min":
            ################################################################################################
            ps_max = positive_distances.max(dim=1)[0]
            ns_min = negative_distances.min(dim=1)[0]

            assert ps_max.shape == torch.Size([self.config['nbatch']])
            assert ns_min.shape == torch.Size([self.config['nbatch']])

            loss = F.relu(input=ps_max - ns_min + self.config['margin'])
            assert loss.shape == torch.Size([self.config['nbatch']])
        elif self.config['np_compare'] == "all_min":
            ################################################################################################
            ps_all = positive_distances
            ns_min = negative_distances.min(dim=1)[0]

            ns_min = ns_min.reshape([self.config['nbatch'], 1])

            assert ps_all.shape == torch.Size([self.config['nbatch'], self.config['L_type_max']])
            assert ns_min.shape == torch.Size([self.config['nbatch'], 1])

            loss = F.relu(input=ps_all - ns_min + self.config['margin']).sum(dim=-1)
            assert loss.shape == torch.Size([self.config['nbatch']])

        elif self.config['np_compare'] == "all_all":
            ################################################################################################
            ps_all = positive_distances.unsqueeze(dim=1)
            ns_all = negative_distances.unsqueeze(dim=2)

            assert ps_all.shape == torch.Size([self.config['nbatch'], 1, self.config['L_type_max']])
            assert ns_all.shape == torch.Size([self.config['nbatch'], self.config['L_type_max'], 1])

            loss = F.relu(input=ps_all - ns_all + self.config['margin']).sum(dim=[1, 2])
            assert loss.shape == torch.Size([self.config['nbatch']])
            ################################################################################################
        else:
            raise self.config['np_compare']
        return loss

    def _dist(self, es, nbes, nbrs, nbfs, ts, t_is_all=False):
        nb = es.shape[0]
        ss = nbes.shape[1]
        ls = ts.shape[1]
        assert es.size() == torch.Size([nb])
        assert nbes.size() == torch.Size([nb, ss])
        assert nbrs.size() == torch.Size([nb, ss])
        assert nbfs.size() == torch.Size([nb, ss])
        assert ts.size() == torch.Size([nb, ls])

        es_emb = self.entities_emb(es).unsqueeze(dim=1)
        es_emb = es_emb.repeat([1, ls, 1])
        assert es_emb.shape == torch.Size([nb, ls, self.config['type_dim']])

        if 'mlp_for_e2t' in self.config and self.config['mlp_for_e2t']:
            es_emb = self.ee(es_emb)

        if 'disable_attention' in self.config and self.config['disable_attention']:
            e2t_res = self.e2t(es_emb)
        else:
            patt = self._att(nbes, nbrs, nbfs, ts, t_is_all=t_is_all)
            assert patt.shape == torch.Size([nb, ls, self.config['type_dim']])
            if 'disable_local' in self.config and self.config['disable_local']:
                e2t_res = self.e2t(patt)
            else:
                if "pt_mask_local" in self.config and self.config['pt_mask_local']:
                    es_emb = es_emb * 0
                elif "pt_mask_attention" in self.config and self.config['pt_mask_attention']:
                    patt = patt * 0
                e2t_res = self.e2t(torch.cat([patt, es_emb], dim=2))

        assert e2t_res.shape == torch.Size([nb,
                                            ls,
                                            self.config['type_dim']])

        if t_is_all:
            pt_emb = self.types_emb.weight.unsqueeze(dim=0)
        else:
            pt_emb = self.types_emb(ts)

        p_m_res = e2t_res - pt_emb
        assert p_m_res.size() == torch.Size([nb, ls, self.config['type_dim']])

        distances = p_m_res.norm(p=self.norm, dim=2)
        return distances

    def _att(self, nbes, nbrs, nbfs, ts, t_is_all=False):
        """
        :param nbes: neighbors entities, torch.Size([self.config['nbatch'], self.config['S_triplet_max']])
        :param nbrs: neighbors relations, torch.Size([self.config['nbatch'], self.config['S_triplet_max']])
        :param nbfs: neighbors direction flag, torch.Size([self.config['nbatch'], self.config['S_triplet_max']])
        :param ts: target types, and shape should be torch.Size([self.config['nbatch'], self.config['L_type_max']])
        :return: distances vector, and shape should be
        torch.Size([self.config['nbatch'], self.config['type_max'], self.type_dim])
        """
        nb = nbes.shape[0]
        ss = nbes.shape[1]
        ls = ts.shape[1]
        assert nbes.size() == torch.Size([nb, ss])
        assert nbrs.size() == torch.Size([nb, ss])
        assert nbfs.size() == torch.Size([nb, ss])
        assert ts.size() == torch.Size([nb, ls])

        nbes_emb = self.entities_emb(nbes)
        nbrs_emb = self.relations_emb(nbrs)
        assert nbes_emb.size() == torch.Size([nb, ss, self.type_dim])
        assert nbrs_emb.size() == torch.Size([nb, ss, self.type_dim])

        r_embs = nbrs_emb * nbfs.unsqueeze(dim=2)
        value = nbes_emb + r_embs
        assert value.size() == torch.Size([nb, ss, self.type_dim])
        value = value.permute([1, 0, 2])
        assert value.shape == torch.Size([ss, nb, self.type_dim])

        if 'mean_att' in self.config and self.config['mean_att']:
            output = value.mean(dim=0)
            assert output.shape == torch.Size([nb, self.type_dim])
            output = output.unsqueeze(dim=0).repeat([ls, 1, 1])
        else:
            key = r_embs.permute([1, 0, 2])
            assert key.shape == torch.Size([ss, nb, self.type_dim])
            if t_is_all:
                query = self.types_emb.weight.unsqueeze(dim=0)
            else:
                query = self.types_emb(ts)
            query = query.permute([1, 0, 2])
            assert query.shape == torch.Size([ls, nb, self.type_dim])
            if 'att_key' in self.config and self.config['att_key']:
                output, attn_weights = self.att_module(query=query, key=key, value=value)
            else:
                output, attn_weights = self.att_module(query=query, key=value, value=value)
            if 'bm' in self.config and self.config['bm']:
                bmr = self.config['bmr']
                output = bmr * value.mean(dim=0).unsqueeze(dim=0).repeat([ls, 1, 1]) + (1 - bmr) * output
        assert output.shape == torch.Size([ls, nb, self.type_dim])
        return output.permute([1, 0, 2])

    def scores_et(self, entity):
        with torch.no_grad():
            # neighbors
            if self.config['relation_padding']:
                nbs = random.choices(self.dataset_e2rts[entity], k=self.config['S_triplet_max'] - 1)
                nbs += [(self.entity_padding, self.relation_padding, 1)]
            else:
                nbs = random.choices(self.dataset_e2rts[entity], k=self.config['S_triplet_max'])
            es = torch.tensor([entity]).cuda()
            nbes = torch.tensor([[e for e, r, f in nbs]]).cuda()
            nbrs = torch.tensor([[r for e, r, f in nbs]]).cuda()
            nbfs = torch.tensor([[f for e, r, f in nbs]]).cuda()
            ts = torch.tensor([[i for i in range(self.types_emb.num_embeddings)]]).cuda()

            return self._dist(es, nbes, nbrs, nbfs, ts, t_is_all=True).squeeze(dim=0)
