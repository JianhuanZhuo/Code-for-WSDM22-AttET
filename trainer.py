import logging
import sys

from torch import nn
from torch.optim.adagrad import Adagrad
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from model.E2T import E2T
from model.Evaluator import Type_Evaluator
from model.TransE import TransE
from tools.EtDataset import EtDataset
from tools.PnDataset import PnDataset
from tools.util import *
from setproctitle import setproctitle

logging.basicConfig(level=logging.INFO)


def wrap(config):
    config['writer_path'] = os.path.join(config['log_folder'],
                                         '%s-%s-%s' % (config['timestamp_mask'],
                                                       config['log_tag'],
                                                       config.postfix()))
    if not os.path.exists(config['writer_path']):
        os.makedirs(config['writer_path'])

    logfile = os.path.join(config['writer_path'], "grid.log")
    with open(logfile, "w") as fp:
        sys.stdout.write = fp.write
        sys.stderr.write = fp.write
        main_run(config)
        exit()


def emb_init(count, dim):
    emb = nn.Embedding(num_embeddings=count + 1,
                       embedding_dim=dim,
                       padding_idx=count)
    uniform_range = 6 / np.sqrt(dim)
    emb.weight.data.uniform_(-uniform_range, uniform_range)
    # -1 to avoid nan for OOV vector
    # relations_emb.weight.data[:-1, :].div_(relations_emb.weight.data[:-1, :].norm(p=1, dim=1, keepdim=True))
    return emb


def main_run(config):
    # set random seed
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = config['cuda']
    torch.manual_seed(config['seed'])
    random.seed(config['seed'])
    data_loc = config['data_loc']

    if 'writer_path' not in config:
        config['writer_path'] = os.path.join(config['log_folder'],
                                             '%s-%s-%s' % (config['timestamp_mask'],
                                                           config['log_tag'],
                                                           config.postfix()))
    if not os.path.exists(config['writer_path']):
        os.makedirs(config['writer_path'])

    setproctitle(config['writer_path'])

    if config['FB']:
        triplet_train = loadTriplet(data_loc + "data/FB15K/origin/freebase_mtr100_mte100-train.txt")

        relation2Id = loadRelationId(data_loc + "data/FB15K/relation2id.txt")
        entity2Id = loadEntityId(data_loc + "data/FB15K/entity2id.txt")
        type2Id = loadTypeId(data_loc + "data/FB15K/type2id.txt")

        valid_e2t = loadEntity2Type(path=data_loc + "data/FB15K/FB15k_Entity_Type_valid_clean.txt")
        test_e2t = loadEntity2Type(path=data_loc + "data/FB15K/FB15k_Entity_Type_test_clean.txt")
        e2t = loadEntity2Type(path=data_loc + "data/FB15K/origin/FB15k_Entity_Type_train.txt")
    else:
        triplet_train = loadTriplet(data_loc + "data/YAGO/YAGO43k/YAGO43k_name_train.txt")

        relation2Id = loadRelationId(data_loc + "data/YAGO/relation2id.txt")
        entity2Id = loadEntityId(data_loc + "data/YAGO/entity2id.txt")
        type2Id = loadTypeId(data_loc + "data/YAGO/type2id.txt")

        valid_e2t = loadEntity2Type(path=data_loc +
                                         "data/YAGO/YAGO43k_Entity_Types/YAGO43k_Entity_Type_valid_clean_clean.txt")
        test_e2t = loadEntity2Type(path=data_loc +
                                        "data/YAGO/YAGO43k_Entity_Types/YAGO43k_Entity_Type_test_clean_clean.txt")
        e2t = loadEntity2Type(path=data_loc +
                                   "data/YAGO/YAGO43k_Entity_Types/YAGO43k_Entity_Type_train_clean.txt")

    train_ere = encode2id(triplet_train, entity2Id, relation2Id, entity2Id)
    train_e2t = encode2id(e2t, entity2Id, type2Id)

    entities_emb = emb_init(len(entity2Id), config['entity_dim'])
    relations_emb = emb_init(len(relation2Id), config['entity_dim'])
    types_emb = emb_init(len(type2Id), config['type_dim'])

    model_trans = TransE(
        entities_emb=entities_emb,
        relations_emb=relations_emb,
        margin=config['margin'],
        norm=config['norm'],
    )
    model_e2t = E2T(
        entities_emb=entities_emb,
        relations_emb=relations_emb,
        types_emb=types_emb,
        entity_dim=config['entity_dim'],
        type_dim=config['type_dim'],
        config=config,
        eres=train_ere,
        margin=config['margin'],
        norm=config['norm'],
        entity_padding=len(entity2Id),
        relation_padding=len(relation2Id),
    )

    evaluator = Type_Evaluator(encode2id(test_e2t, entity2Id, type2Id),
                               encode2id(valid_e2t + e2t + test_e2t, entity2Id, type2Id),
                               eres=train_ere,
                               logger=logging)
    trans_dataloader = DataLoader(
        dataset=PnDataset(train_ere, list(set(entity2Id.values()))),
        shuffle=True,
        batch_size=config['nbatch'],
        drop_last=True,
        pin_memory=False,
    )
    e2t_dataloader = DataLoader(
        dataset=EtDataset(train_e2t,
                          list(set(type2Id.values())),
                          triplets=train_ere,
                          config=config,
                          entity_padding=len(entity2Id),
                          relation_padding=len(relation2Id),
                          ),
        shuffle=True,
        batch_size=config['nbatch'],
        drop_last=True,
        pin_memory=False,
    )

    summary_writer = SummaryWriter(config['writer_path'])
    summary_writer.add_text('config', config.__str__())
    pid = os.getpid()
    config['pid'] = pid
    logging.info(config)
    logging.info(f"pid is {pid}")
    logging.info(f"output to {config['writer_path']}")

    if 'fixed_transE' in config:
        logging.info("loading checkpoint...")
        checkpoint = torch.load(config['fixed_transE'])
        logging.info("loading load_state_dict...")
        model_trans.load_state_dict(checkpoint['model_trans_static_dict'])

    logging.info("loading model and assign GPU memory...")
    model_e2t = model_e2t.cuda()
    model_trans = model_trans.cuda()
    logging.info("loaded over.")

    trans_op = Adagrad(model_trans.parameters(), lr=config['learning_rate'])
    if config['dr']:
        e2t_op = Adagrad([
            {
                "params": [
                              *(model_e2t.types_emb.parameters()),
                              *(model_e2t.e2t.parameters()),
                              *(model_e2t.ee.parameters()),
                          ] + (
                              [
                                  *(model_e2t.entities_emb.parameters()),
                                  *(model_e2t.relations_emb.parameters()),
                              ] if 'train_e_and_r' in config and config['train_e_and_r'] else []
                          ),
                "lr": config['rate_dual'] * config['learning_rate']
            },
            {
                "params": model_e2t.att_module.parameters(),
                "lr": config['learning_rate']
            },
        ],
            lr=config['learning_rate'])
    else:
        e2t_op = Adagrad(model_e2t.parameters(), lr=config['learning_rate'], weight_decay=config['wd'])

    for epoch in range(config['epochs']):
        if 'not_train' not in config or not config['not_train']:
            if 'fixed_transE' not in config:
                trans_epoch_loss = []
                for positive_triplets, negative_triplets in tqdm(trans_dataloader,
                                                                 desc=f'trans\tepoch: {epoch}/{config["epochs"]}'):
                    trans_op.zero_grad()
                    loss, pd, nd = model_trans(positive_triplets.cuda(), negative_triplets.cuda())
                    loss.mean().backward()

                    trans_op.step()

                    trans_epoch_loss.append(loss)
                summary_writer.add_scalar('Epoch/Margin/trans',
                                          (torch.stack(trans_epoch_loss) > 0).float().mean().item(),
                                          global_step=epoch)
                summary_writer.add_scalar('Epoch/Loss/trans', torch.stack(trans_epoch_loss).mean().item(),
                                          global_step=epoch)

            e2t_epoch_loss = []
            for es, nbes, nbrs, nbfs, pts, nts in tqdm(e2t_dataloader,
                                                       desc=f'e2t  \tepoch: {epoch}/{config["epochs"]}',
                                                       bar_format="{desc}{percentage:3.0f}%|{bar:10}{r_bar}",
                                                       ):
                e2t_op.zero_grad()
                model_e2t.train()
                loss = model_e2t(es.cuda(), nbes.cuda(), nbrs.cuda(), nbfs.cuda(), pts.cuda(), nts.cuda())

                loss.mean().backward()

                e2t_op.step()

                e2t_epoch_loss.append(loss)
            summary_writer.add_scalar('Epoch/Margin/e2t', (torch.stack(e2t_epoch_loss) > 0).float().mean().item(),
                                      global_step=epoch)
            summary_writer.add_scalar('Epoch/Loss/e2t', torch.stack(e2t_epoch_loss).mean().item(), global_step=epoch)

        if (epoch + 1) % config['evaluator_time'] == 0:
            torch.save(
                {
                    "model_trans_static_dict": model_trans.state_dict(),
                    "trans_optimizer_state_dict": trans_op.state_dict(),
                    "model_e2t_static_dict": model_e2t.state_dict(),
                    "e2t_optimizer_state_dict": e2t_op.state_dict(),
                    "epoch": epoch,
                },
                os.path.join('%s' % config['writer_path'], f"checkpoint-{epoch}.tar")
            )
            if 'eval' not in config or config['eval']:
                model_e2t.eval()
                with torch.no_grad():
                    fmrr, fmean_pos, fhits = evaluator(model_e2t, epoch)
                    summary_writer.add_scalar('Eval/fmrr', fmrr, global_step=epoch)
                    summary_writer.add_scalar('Eval/fmean_pos', fmean_pos, global_step=epoch)
                    summary_writer.add_scalar('Eval/hits@1', fhits[0], global_step=epoch)
                    summary_writer.add_scalar('Eval/hits@3', fhits[1], global_step=epoch)
                    summary_writer.add_scalar('Eval/hits@10', fhits[2], global_step=epoch)

                    if "pt" in config and config['pt']:
                        config['pt_mask_local'] = True
                        config['pt_mask_attention'] = False
                        fmrr, fmean_pos, fhits = evaluator(model_e2t, epoch)
                        summary_writer.add_scalar('OnlyLocal/fmrr', fmrr, global_step=epoch)
                        summary_writer.add_scalar('OnlyLocal/fmean_pos', fmean_pos, global_step=epoch)
                        summary_writer.add_scalar('OnlyLocal/hits@1', fhits[0], global_step=epoch)
                        summary_writer.add_scalar('OnlyLocal/hits@3', fhits[1], global_step=epoch)
                        summary_writer.add_scalar('OnlyLocal/hits@10', fhits[2], global_step=epoch)

                        config['pt_mask_local'] = False
                        config['pt_mask_attention'] = True
                        fmrr, fmean_pos, fhits = evaluator(model_e2t, epoch)
                        summary_writer.add_scalar('OnlyAtt/fmrr', fmrr, global_step=epoch)
                        summary_writer.add_scalar('OnlyAtt/fmean_pos', fmean_pos, global_step=epoch)
                        summary_writer.add_scalar('OnlyAtt/hits@1', fhits[0], global_step=epoch)
                        summary_writer.add_scalar('OnlyAtt/hits@3', fhits[1], global_step=epoch)
                        summary_writer.add_scalar('OnlyAtt/hits@10', fhits[2], global_step=epoch)

                        config['pt_mask_local'] = False
                        config['pt_mask_attention'] = False


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    cfg_file = "config.yaml"
    if len(sys.argv) == 2:
        cfg_file = sys.argv[1]
        if not cfg_file.endswith(".yaml"):
            raise Exception(f"arguments exception: {sys.argv}")
    else:
        logging.warning(f"using default configure: {cfg_file}")
    main_run(load_specific_config(cfg_file))
