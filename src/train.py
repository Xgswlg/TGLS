# coding: utf-8
import json
import os
from tqdm import tqdm
import re
from pprint import pprint
import unicodedata
from transformers import BertModel, BertTokenizerFast
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import glob
import time
import logging
from utils import Preprocessor, DefaultLogger
from tgls import (EssentialLabelsTagging,
                           WholeLabelsTagging,
                           DataMaker,
                           TGLS,
                           MetricsCalculator)
import numpy as np
import config

config = config.train_config
hyper_parameters = config["hyper_parameters"]

# device
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ["CUDA_VISIBLE_DEVICES"] = str(config["device_num"])
device = torch.device("cuda:{}".format(config["device_num"]) if torch.cuda.is_available() else "cpu")

# for reproductivity
torch.manual_seed(hyper_parameters["seed"])  # pytorch random seed
torch.backends.cudnn.deterministic = True

# log
experiment_name = config["exp_name"]
logger = DefaultLogger(config["log_path"], experiment_name, config["run_name"], config["run_id"], hyper_parameters)
model_state_dict_dir = config["path_to_save_model"]
if not os.path.exists(model_state_dict_dir):
    os.makedirs(model_state_dict_dir)

# Load Data
data_home = config["data_home"]
train_data_path = os.path.join(data_home, experiment_name, config["train_data"])
valid_data_path = os.path.join(data_home, experiment_name, config["valid_data"])
rel2id_path = os.path.join(data_home, experiment_name, config["rel2id"])
ent2id_path = os.path.join(data_home, experiment_name, config["ent2id"])
train_data = json.load(open(train_data_path, "r", encoding="utf-8"))
valid_data = json.load(open(valid_data_path, "r", encoding="utf-8"))



# tokenizer
tokenize = lambda text: text.split(" ")

def get_tok2char_span_map(text):
    tokens = text.split(" ")
    tok2char_span = []
    char_num = 0
    for tok in tokens:
        tok2char_span.append((char_num, char_num + len(tok)))
        char_num += len(tok) + 1  # +1: whitespace
    return tok2char_span

preprocessor = Preprocessor(tokenize_func=tokenize,
                            get_tok2char_span_map_func=get_tok2char_span_map)

# train and valid max token num
max_tok_num = 0
all_data = train_data + valid_data

for sample in all_data:
    tokens = tokenize(sample["text"])
    max_tok_num = max(max_tok_num, len(tokens))
max_tok_num

if max_tok_num > hyper_parameters["max_seq_len"]:
    train_data = preprocessor.split_into_short_samples(train_data,
                                                       hyper_parameters["max_seq_len"],
                                                       sliding_len=hyper_parameters["sliding_len"],
                                                       data_type="valid"
                                                       )
    valid_data = preprocessor.split_into_short_samples(valid_data,
                                                       hyper_parameters["max_seq_len"],
                                                       sliding_len=hyper_parameters["sliding_len"],
                                                       data_type="valid"
                                                       )

print("train: {}".format(len(train_data)), "valid: {}".format(len(valid_data)))

max_seq_len = min(max_tok_num, hyper_parameters["max_seq_len"])
print('max_len: ', max_seq_len)
rel2id = json.load(open(rel2id_path, "r", encoding="utf-8"))
ent2id = json.load(open(ent2id_path, "r", encoding="utf-8"))
essential_tagger = EssentialLabelsTagging(rel2id, max_seq_len, ent2id)
whole_tagger = WholeLabelsTagging(rel2id, max_seq_len, ent2id)
essential_tag_size = essential_tagger.get_tag_size()
whole_tag_size = whole_tagger.get_tag_size()

bert_tokenizer = BertTokenizerFast.from_pretrained(config["bert_path"], add_special_tokens=False,
                                                    do_lower_case=False)
token2idx_path = os.path.join(data_home, experiment_name, config["token2idx"])
char_dict_path = os.path.join(data_home, experiment_name, config["char_dict"])
pos2idx_dict_path = os.path.join(data_home, experiment_name, config["pos2idx"])
lemma2idx_dict_path = os.path.join(data_home, experiment_name, config["lemma2idx"])
token2idx = json.load(open(token2idx_path, "r", encoding="utf-8"))
pos2idx = json.load(open(pos2idx_dict_path, "r", encoding="utf-8"))
lemma2idx = json.load(open(lemma2idx_dict_path, "r", encoding="utf-8"))
idx2token = {idx: tok for tok, idx in token2idx.items()}

char2idx = {line.replace('\n', ''): idx
            for idx, line in enumerate(open(char_dict_path, "r", encoding="utf-8").readlines())}
idx2char = {idx: char for char, idx in char2idx.items()}
char_size = len(idx2char)
pos_size = len(pos2idx)
lemma_size = len(lemma2idx)
max_tok_len = 0
for tok in token2idx.keys():
    max_tok_len = max(len(tok), max_tok_len)
print('max_tok_len: ', max_tok_len)


def text2indices(text, pos_list, lemma_list, max_seq_len, max_tok_len):
    # bsz * max_seq_len
    input_ids = []
    # bsz * max_seq_len * max_tok_len
    char_ids = []
    pos_ids = []
    lemma_ids = []
    # bsz * max_seq_len
    rel_tok_lens = []
    tokens = text.split(" ")
    for tok in tokens:
        if tok not in token2idx:
            input_ids.append(token2idx['<UNK>'])
        else:
            input_ids.append(token2idx[tok])
        tok_char_list = []
        for character in tok:
            if character not in char2idx:
                tok_char_list.append(char2idx['<UNK>'])
            else:
                tok_char_list.append(char2idx[character])
        rel_tok_len = max_tok_len
        if len(tok_char_list) < max_tok_len:
            rel_tok_len = len(tok_char_list)
            tok_char_list.extend([char2idx['<PAD>']] * (max_tok_len - len(tok_char_list)))
        tok_char_list = tok_char_list[:max_tok_len]
        rel_tok_lens.append(rel_tok_len)
        char_ids.append(tok_char_list)
    rel_len = max_seq_len
    if len(input_ids) < max_seq_len:
        rel_len = len(input_ids)
        for _ in range((max_seq_len - len(input_ids))):
            char_ids.extend([[char2idx['<PAD>']] * max_tok_len])
        rel_tok_lens.extend([0] * (max_seq_len - len(input_ids)))
        input_ids.extend([token2idx['<PAD>']] * (max_seq_len - len(input_ids)))

    for pos in pos_list:
        if pos not in pos2idx:
            pos_ids.append(pos2idx['<UNK>'])
        else:
            pos_ids.append(pos2idx[pos])
    pos_ids.extend([pos2idx['<PAD>']] * (max_seq_len - len(pos_ids)))
    for lemma in lemma_list:
        if lemma not in lemma2idx:
            lemma_ids.append(lemma2idx['<UNK>'])
        else:
            lemma_ids.append(lemma2idx[lemma])
    pos_ids.extend([pos2idx['<PAD>']] * (max_seq_len - len(pos_ids)))
    lemma_ids.extend([lemma2idx['<PAD>']] * (max_seq_len - len(lemma_ids)))

    pos_ids = torch.tensor(pos_ids[:max_seq_len])
    lemma_ids = torch.tensor(lemma_ids[:max_seq_len])
    input_ids = torch.tensor(input_ids[:max_seq_len])
    char_ids = torch.tensor(char_ids[:max_seq_len])

    return input_ids, pos_ids, lemma_ids, rel_len, char_ids, rel_tok_lens

data_maker = DataMaker(bert_tokenizer, text2indices, get_tok2char_span_map, essential_tagger,
                                whole_tagger)

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

indexed_train_data = data_maker.get_indexed_data(train_data, max_seq_len, max_tok_len, data_type="valid")
indexed_valid_data = data_maker.get_indexed_data(valid_data, max_seq_len, max_tok_len, data_type="valid")


train_dataloader = DataLoader(MyDataset(indexed_train_data),
                              batch_size=hyper_parameters["batch_size"],
                              shuffle=True,
                              num_workers=6,
                              drop_last=False,
                              collate_fn=data_maker.generate_batch,
                              )
valid_dataloader = DataLoader(MyDataset(indexed_valid_data),
                              batch_size=hyper_parameters["batch_size"],
                              shuffle=True,
                              num_workers=6,
                              drop_last=False,
                              collate_fn=data_maker.generate_batch,
                              )

# Model
def loadGloveModel(File):
    print("Loading Glove Model")
    import pandas as pd
    encoding = 'ISO-8859-1' if 'mpqa' in config['exp_name'] or 'ds' in config['exp_name'] else 'ISO-8859-1'
    df = pd.read_csv(File, sep=" ", quoting=3, header=None, index_col=0, skiprows=1, encoding=encoding)
    glove = {key: val.values for key, val in df.T.items()}
    return glove

bert_encoder = BertModel.from_pretrained(config["bert_path"])
glove = loadGloveModel(config["pretrained_word_embedding_path"])
# prepare embedding matrix
word_embedding_init_matrix = np.random.normal(-1, 1, size=(len(token2idx), hyper_parameters["word_embedding_dim"]))
count_in = 0

# If it is in the pre-trained word vector, use the pre-training vector, otherwise use a random vector
for ind, tok in tqdm(idx2token.items(), desc="Embedding matrix initializing..."):
    if tok in glove.keys():
        count_in += 1
        word_embedding_init_matrix[ind] = glove[tok][:hyper_parameters["word_embedding_dim"]]

print("{:.4f} tokens are in the pretrain word embedding matrix".format(count_in / len(idx2token)))  # 命中预训练词向量的比例
word_embedding_init_matrix = torch.FloatTensor(word_embedding_init_matrix)

rel_extractor = TGLS(experiment_name,
                                    bert_encoder,
                                    bert_tokenizer,
                                    word_embedding_init_matrix,
                                    hyper_parameters["emb_dropout"],
                                    hyper_parameters["enc_hidden_size"],
                                    hyper_parameters["dec_hidden_size"],
                                    hyper_parameters["rnn_dropout"],
                                    essential_tag_size,
                                    whole_tag_size,
                                    char_size,
                                    hyper_parameters["char_embedding_dim"],
                                    pos_size,
                                    lemma_size,
                                    )

rel_extractor = rel_extractor.to(device)



# loss and metric
metrics = MetricsCalculator(essential_tagger)
loss_func = lambda y_pred, y_true, cr_ceil, cr_low, weights: metrics._multilabel_categorical_crossentropy(y_pred, y_true, cr_ceil, cr_low, weights)

# Train

# train step
def train_step(batch_train_data, optimizer, ep):

    sample_list, batch_input_ids, tok2char_span_list, gold_essential_label, gold_whole_label, batch_rel_lens, \
    batch_char_ids, batch_rel_tok_lens, bert_input_ids_list, bert_attention_mask_list, bert_token_type_ids_list, bert_text_list, batch_pos_ids, batch_lemma_ids, batch_sample_weight_list = batch_train_data

    batch_input_ids, gold_essential_label, gold_whole_label, batch_char_ids, bert_input_ids_list, bert_attention_mask_list, bert_token_type_ids_list, batch_pos_ids, batch_lemma_ids, batch_sample_weight_list = (
        batch_input_ids.to(device),
        gold_essential_label.to(device),
        gold_whole_label.to(device),
        batch_char_ids.to(device),
        bert_input_ids_list.to(device),
        bert_attention_mask_list.to(device),
        bert_token_type_ids_list.to(device),
        batch_pos_ids.to(device), batch_lemma_ids.to(device),
        batch_sample_weight_list.to(device)
    )

    # zero the parameter gradients
    optimizer.zero_grad()

    attention_scoring4pred, threshold4pred, total_attention_scoring4graph, total_threshold4graph = rel_extractor(batch_sample_weight_list,
        bert_text_list, batch_pos_ids, batch_lemma_ids, bert_input_ids_list, bert_attention_mask_list,
        bert_token_type_ids_list, batch_input_ids, batch_rel_lens, batch_char_ids, batch_rel_tok_lens)

    # set margin=2
    main_cr_ceil = threshold4pred
    main_cr_low = main_cr_ceil - 2

    loss4main = loss_func(attention_scoring4pred, gold_essential_label, main_cr_ceil, main_cr_low, batch_sample_weight_list)

    # set margin=6
    loss4graph = loss_func(total_attention_scoring4graph[0], gold_whole_label,
                           total_threshold4graph[0], total_threshold4graph[0] - 6, batch_sample_weight_list) 

    alpha = 0.25 # 0 0.1 0.25 0.5 0.75 1
    loss = loss4main + alpha * loss4graph 

    loss.backward()
    optimizer.step()

    pred_essential_label = ((attention_scoring4pred > (main_cr_low + main_cr_ceil) / 2) * (
            attention_scoring4pred != 0)).long()
    sample_acc = metrics.get_sample_accuracy(pred_essential_label,
                                             gold_essential_label)

    return loss.item(), sample_acc.item()


# valid step
def valid_step(batch_valid_data, ep):

    sample_list, batch_input_ids, tok2char_span_list, gold_essential_label, _, batch_rel_lens, batch_char_ids, \
    batch_rel_tok_lens, bert_input_ids_list, bert_attention_mask_list, bert_token_type_ids_list, bert_text_list, batch_pos_ids, batch_lemma_ids, batch_sample_weight_list = batch_valid_data

    batch_input_ids, gold_essential_label, batch_char_ids, bert_input_ids_list, bert_attention_mask_list, bert_token_type_ids_list, batch_pos_ids, batch_lemma_ids, batch_sample_weight_list = (
        batch_input_ids.to(device),
        gold_essential_label.to(device),
        batch_char_ids.to(device),
        bert_input_ids_list.to(device),
        bert_attention_mask_list.to(device),
        bert_token_type_ids_list.to(device),
        batch_pos_ids.to(device), batch_lemma_ids.to(device),
        batch_sample_weight_list.to(device)
    )
    with torch.no_grad():
        attention_scoring4pred, threshold4pred, total_attention_scoring4graph, total_threshold4graph = rel_extractor(batch_sample_weight_list,
            bert_text_list, batch_pos_ids, batch_lemma_ids, bert_input_ids_list, bert_attention_mask_list,
            bert_token_type_ids_list, batch_input_ids, batch_rel_lens, batch_char_ids, batch_rel_tok_lens)

    main_cr_ceil = threshold4pred
    main_cr_low = main_cr_ceil - 2

    pred_essential_label = ((attention_scoring4pred > (main_cr_low + main_cr_ceil) / 2) * (attention_scoring4pred != 0)).long()
    sample_acc = metrics.get_sample_accuracy(pred_essential_label,
                                             gold_essential_label)

    cpg_dict = metrics.get_cpg(sample_list,
                               tok2char_span_list,
                               pred_essential_label)
    return sample_acc.item(), cpg_dict


max_f1 = 0.

def train_n_valid(train_dataloader, dev_dataloader, optimizer, scheduler, num_epoch):
    def train(dataloader, ep):
        # train
        rel_extractor.train()

        t_ep = time.time()
        total_loss, total_sample_acc = 0., 0.
        for batch_ind, batch_train_data in enumerate(dataloader):
            t_batch = time.time()

            loss, sample_acc = train_step(batch_train_data, optimizer, ep)

            total_loss += loss
            total_sample_acc += sample_acc

            avg_loss = total_loss / (batch_ind + 1)

            # scheduler
            if hyper_parameters["scheduler"] == "ReduceLROnPlateau":
                scheduler.step(avg_loss)
            else:
                scheduler.step()

            avg_sample_acc = total_sample_acc / (batch_ind + 1)

            batch_print_format = "\rrun_name: {}, Epoch: {}/{}, batch: {}/{}, train_loss: {}, " + "t_sample_acc: {}," + "lr: {}, batch_time: {}, total_time: {} -------------"

            print(batch_print_format.format(config["run_name"],
                                            ep + 1, num_epoch,
                                            batch_ind + 1, len(dataloader),
                                            avg_loss,
                                            avg_sample_acc,
                                            optimizer.param_groups[0]['lr'],
                                            time.time() - t_batch,
                                            time.time() - t_ep,
                                            ), end="")

            if (batch_ind+1)%200 == 0 and (ep+1)%2 == 0:
                valid_f1 = valid(valid_dataloader, ep)
                global max_f1
                if valid_f1 >= max_f1:
                    max_f1 = valid_f1
                    if valid_f1 > config["f1_2_save"]:  # save the best model
                        modle_state_num = len(glob.glob(model_state_dict_dir + "/model_state_dict_*.pt"))
                        torch.save(rel_extractor.state_dict(), os.path.join(model_state_dict_dir,
                                                                            "model_state_dict_{}_{}_{}.pt".format(modle_state_num, ep+1,
                                                                                                            str(valid_f1)[
                                                                                                            :6])))
                    #                 scheduler_state_num = len(glob.glob(schedule_state_dict_dir + "/scheduler_state_dict_*.pt"))
                    #                 torch.save(scheduler.state_dict(), os.path.join(schedule_state_dict_dir, "scheduler_state_dict_{}.pt".format(scheduler_state_num)))
                print("Current avf_f1: {}, Best f1: {}".format(valid_f1, max_f1))
                rel_extractor.train()                                        

    def valid(dataloader, ep):
        # valid
        rel_extractor.eval()
        t_ep = time.time()
        total_sample_acc = 0.
        total_cpg_dict = {}
        for batch_ind, batch_valid_data in enumerate(tqdm(dataloader, desc="Validating")):
            sample_acc, cpg_dict = valid_step(batch_valid_data, ep)
            total_sample_acc += sample_acc

            '''
            cpg = {
            "expression": [0, 0, 0],
            "holder": [0, 0, 0],
            "target": [0, 0, 0],
            "targeted": [0, 0, 0],
            "no_sent_graph": [0, 0, 0, 0],
            "sent_graph": [0, 0, 0, 0]
            }
            '''

            # init total_cpg_dict
            for k in cpg_dict.keys():
                if k not in total_cpg_dict:
                    total_cpg_dict[k] = [0, 0, 0, 0]

            for k, cpg in cpg_dict.items():
                for idx, n in enumerate(cpg):
                    total_cpg_dict[k][idx] += cpg[idx]

        avg_sample_acc = total_sample_acc / len(dataloader)

        log_dict = metrics.get_prf_scores(total_cpg_dict)
        log_dict["val_shaking_tag_acc"] = avg_sample_acc
        log_dict["time"] = time.time() - t_ep

        logger.log(log_dict)
        pprint(log_dict)
        final_score = log_dict['sent_graph_f1']

        return final_score

    for ep in range(num_epoch):
        train(train_dataloader, ep)

        if (ep+1)%2 == 0:

            valid_f1 = valid(valid_dataloader, ep)

            global max_f1
            if valid_f1 >= max_f1:
                max_f1 = valid_f1
                if valid_f1 > config["f1_2_save"]:  # save the best model
                    modle_state_num = len(glob.glob(model_state_dict_dir + "/model_state_dict_*.pt"))
                    torch.save(rel_extractor.state_dict(), os.path.join(model_state_dict_dir,
                                                                        "model_state_dict_{}_{}_{}.pt".format(modle_state_num, ep+1,
                                                                                                           str(valid_f1)[
                                                                                                           :6])))

            print("Current avf_f1: {}, Best f1: {}".format(valid_f1, max_f1))




# optimizer
init_learning_rate = float(hyper_parameters["lr"])
optimizer = torch.optim.Adam(rel_extractor.parameters(), lr=init_learning_rate)
# ingnored_params = list(map(id, rel_extractor.bert_encoder.parameters()))
# base_params = filter(lambda p: id(p) not in ingnored_params, rel_extractor.parameters())

# optimizer = torch.optim.Adam([{'params': base_params, 'lr': 3e-5}, {'params': rel_extractor.bert_encoder.parameters(), 'lr': 3e-5}], lr = init_learning_rate)

if hyper_parameters["scheduler"] == "CAWR":
    T_mult = hyper_parameters["T_mult"]
    rewarm_epoch_num = hyper_parameters["rewarm_epoch_num"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                     len(train_dataloader) * rewarm_epoch_num, T_mult)

elif hyper_parameters["scheduler"] == "Step":
    decay_rate = hyper_parameters["decay_rate"]
    decay_steps = hyper_parameters["decay_steps"]
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_steps, gamma=decay_rate)

elif hyper_parameters["scheduler"] == "ReduceLROnPlateau":
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", verbose=True, patience=6)


if not config["fr_scratch"]:
    model_state_path = config["model_state_dict_path"]
    rel_extractor.load_state_dict(torch.load(model_state_path))
    print("------------model state {} loaded ----------------".format(model_state_path.split("/")[-1]))

train_n_valid(train_dataloader, valid_dataloader, optimizer, scheduler, hyper_parameters["epochs"])
