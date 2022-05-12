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

config = config.eval_config
hyper_parameters = config["hyper_parameters"]

# device
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ["CUDA_VISIBLE_DEVICES"] = str(config["device_num"])
device = torch.device("cuda:{}".format(config["device_num"]) if torch.cuda.is_available() else "cpu")

data_home = config["data_home"]
experiment_name = config["exp_name"]
test_data_path = os.path.join(data_home, experiment_name, config["test_data"])
batch_size = config["hyper_parameters"]["batch_size"]
rel2id_path = os.path.join(data_home, experiment_name, config["rel2id"])
ent2id_path = os.path.join(data_home, experiment_name, config["ent2id"])
max_seq_len = config["hyper_parameters"]["max_seq_len"]

# for reproductivity
torch.backends.cudnn.deterministic = True

# Load Data
rel2id = json.load(open(rel2id_path, "r", encoding = "utf-8"))
ent2id = json.load(open(ent2id_path, "r", encoding = "utf-8"))
test_data = json.load(open(test_data_path, "r", encoding = "utf-8"))

# tokenize
tokenize = lambda text: text.split(" ")
def get_tok2char_span_map(text):
    tokens = text.split(" ")
    tok2char_span = []
    char_num = 0
    for tok in tokens:
        tok2char_span.append((char_num, char_num + len(tok)))
        char_num += len(tok) + 1 # +1: whitespace
    return tok2char_span

preprocessor = Preprocessor(tokenize_func = tokenize,
                            get_tok2char_span_map_func = get_tok2char_span_map)

# test max token num
max_tok_num = 0
for sample in test_data:
    tokens = tokenize(sample["text"])
    max_tok_num = max(max_tok_num, len(tokens))
max_tok_num
if max_tok_num > hyper_parameters["max_seq_len"]:
    test_data = preprocessor.split_into_short_samples(test_data,
                                                          hyper_parameters["max_seq_len"],
                                                          sliding_len = hyper_parameters["sliding_len"],
                                                          encoder = config["encoder"]
                                                         )
max_seq_len = min(max_tok_num, hyper_parameters["max_seq_len"])
print(max_seq_len)

essential_tagger = EssentialLabelsTagging(rel2id, max_seq_len, ent2id)
whole_tagger = WholeLabelsTagging(rel2id, max_seq_len, ent2id)
essential_tag_size = essential_tagger.get_tag_size()
whole_tag_size = whole_tagger.get_tag_size()

print("test: {}".format(len(test_data)))

# dataset
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

indexed_test_data = data_maker.get_indexed_data(test_data, max_seq_len, max_tok_len, data_type="valid")
test_dataloader = DataLoader(MyDataset(indexed_test_data),
                                  batch_size = hyper_parameters["batch_size"],
                                  shuffle = False,
                                  num_workers = 6,
                                  drop_last = False,
                                  collate_fn = data_maker.generate_batch,
                                 )

# model
def loadGloveModel(File):
    print("Loading Glove Model")
    import pandas as pd
    encoding = 'ISO-8859-1' if 'mpqa' in config['exp_name'] or 'ds' in config['exp_name'] else 'ISO-8859-1'
    df = pd.read_csv(File, sep=" ", quoting=3, header=None, index_col=0, skiprows=1, encoding=encoding)
    glove = {key: val.values for key, val in df.T.items()}
    return glove


bert_encoder = BertModel.from_pretrained(config["bert_path"], hidden_dropout_prob=0.3)
glove = loadGloveModel(config["pretrained_word_embedding_path"])
# prepare embedding matrix
word_embedding_init_matrix = np.random.normal(-1, 1, size=(len(token2idx), hyper_parameters["word_embedding_dim"]))
count_in = 0

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
rel_extractor.load_state_dict(torch.load(config["model_state_dict_dir"], map_location=device))
rel_extractor.eval()

# metrics
metrics = MetricsCalculator(essential_tagger)

def eval_step(batch_test_data):
    sample_list, batch_input_ids, tok2char_span_list, gold_essential_label, _, batch_rel_lens, batch_char_ids, \
    batch_rel_tok_lens, bert_input_ids_list, bert_attention_mask_list, bert_token_type_ids_list, bert_text_list, batch_pos_ids, batch_lemma_ids, batch_sample_weight_list = batch_test_data

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
            bert_text_list, batch_pos_ids, batch_lemma_ids, bert_input_ids_list, bert_attention_mask_list, bert_token_type_ids_list,
            batch_input_ids, batch_rel_lens, batch_char_ids, batch_rel_tok_lens)

    main_cr_ceil = threshold4pred
    main_cr_low = main_cr_ceil - 2

    pred_essential_label = ((attention_scoring4pred > (main_cr_ceil+main_cr_low)/2) * (attention_scoring4pred != 0)).long()
    sample_acc = metrics.get_sample_accuracy(pred_essential_label,
                                             gold_essential_label)

    cpg_dict = metrics.get_cpg(sample_list,
                               tok2char_span_list,
                               pred_essential_label)
    return sample_acc.item(), cpg_dict

def eval(dataloader):

    t_ep = time.time()
    total_sample_acc = 0.
    total_cpg_dict = {}
    for batch_ind, batch_eval_data in enumerate(tqdm(dataloader, desc="Evaluating")):
        sample_acc, cpg_dict = eval_step(batch_eval_data)
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
    pprint(total_cpg_dict)
    log_dict = metrics.get_prf_scores(total_cpg_dict)
    log_dict["val_shaking_tag_acc"] = avg_sample_acc
    log_dict["time"] = time.time() - t_ep

    return pprint(log_dict)


def pred(text):
    text = text[:500]
    # get codes
    codes = tokenizer.encode_plus(text,
                            return_offsets_mapping = True,
                            add_special_tokens = False,
                            truncation = True,
                            pad_to_max_length = True)
    input_ids = torch.tensor(codes["input_ids"]).long().unsqueeze(0)
    print(input_ids.shape)
    attention_mask = torch.tensor(codes["attention_mask"]).long().unsqueeze(0)
    token_type_ids = torch.tensor(codes["token_type_ids"]).long().unsqueeze(0)
    tok2char_span = codes["offset_mapping"]
    input_ids, attention_mask, token_type_ids = (input_ids.to(device),
                          attention_mask.to(device),
                          token_type_ids.to(device)
                         )

    pred_shaking_outputs, _, _, pred_small_shaking_hiddens4ts, pred_small_shaking_hiddens4graph_ts = rel_extractor(input_ids,
                                                                        attention_mask,
                                                                        token_type_ids,
                                                                        True
                                                                         )
    main_cr_ceil = pred_small_shaking_hiddens4ts
    main_cr_low = main_cr_ceil - 2

    pred_shaking_tag = ((pred_shaking_outputs > (main_cr_ceil+main_cr_low)/2) * (pred_shaking_outputs != 0)).long()[0]
    # print(pred_shaking_tag.shape)
    # print(torch.sum(pred_shaking_tag))
    pred_rel_list, pred_ent_list = handshaking_tagger.decode_rel(text, pred_shaking_tag, tok2char_span, [])
    print("rel: ", pred_rel_list)
    print("ent: ", pred_ent_list)
    return pred_ent_list

eval(test_dataloader)
