import re
from tqdm import tqdm
import torch
import copy
import torch
import torch.nn as nn
import json
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math
from components import TokenPairModeling 
from collections import Counter
import numpy as np
import config
from enhanced_lstm import EnhancedLSTM

config = config.train_config
hyper_parameters = config["hyper_parameters"]

class EssentialLabelsTagging(object):
    def __init__(self, rel2id, max_seq_len, entity_type2id):
        super().__init__()
        self.rel2id = rel2id
        self.id2rel = {ind: rel for rel, ind in rel2id.items()}

        self.separator = "\u2E80"
        self.link_types = {"SH2OH",  # subject head to object head
                           "OH2SH",  # object head to subject head
                           "ST2OT",  # subject tail to object tail
                           "OT2ST",  # object tail to subject tail
                           }
        self.tags = {self.separator.join([rel.lower(), lt]) for rel in self.rel2id.keys() for lt in self.link_types}

        self.ent2id = entity_type2id
        self.id2ent = {ind: ent for ent, ind in self.ent2id.items()}
        self.tags |= {self.separator.join([ent.lower(), "EH2ET"]) for ent in
                      self.ent2id.keys()}  # EH2ET: entity head to entity tail

        self.tags = sorted(self.tags)

        self.tag2id = {t: idx for idx, t in enumerate(self.tags)}
        self.id2tag = {idx: t for t, idx in self.tag2id.items()}
        self.matrix_size = max_seq_len

        # map
        # e.g. [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
        self.shaking_idx2matrix_idx = [(ind, end_ind) for ind in range(self.matrix_size) for end_ind in
                                       list(range(self.matrix_size))[ind:]]

        self.matrix_idx2shaking_idx = [[0 for i in range(self.matrix_size)] for j in range(self.matrix_size)]
        for shaking_ind, matrix_ind in enumerate(self.shaking_idx2matrix_idx):
            self.matrix_idx2shaking_idx[matrix_ind[0]][matrix_ind[1]] = shaking_ind

    def get_tag_size(self):
        return len(self.tag2id)

    def get_spots(self, sample):
        '''
        matrix_spots: [(tok_pos1, tok_pos2, tag_id), ]
        '''
        matrix_spots = []
        spot_memory_set = set()

        def add_spot(spot):
            memory = "{},{},{}".format(*spot)
            if memory not in spot_memory_set:
                matrix_spots.append(spot)
                spot_memory_set.add(memory)

        #         # if entity_list exist, need to distinguish entity types
        #         if self.ent2id is not None and "entity_list" in sample:
        for ent in sample["entity_list"]:
            add_spot(
                (ent["tok_span"][0], ent["tok_span"][1] - 1, self.tag2id[self.separator.join([ent["type"].lower(), "EH2ET"])]))

        for rel in sample["relation_list"]:
            subj_tok_span = rel["subj_tok_span"]
            obj_tok_span = rel["obj_tok_span"]
            rel = rel["predicate"].lower()
            #             if self.ent2id is None: # set all entities to default type
            #                 add_spot((subj_tok_span[0], subj_tok_span[1] - 1, self.tag2id[self.separator.join(["DEFAULT", "EH2ET"])]))
            #                 add_spot((obj_tok_span[0], obj_tok_span[1] - 1, self.tag2id[self.separator.join(["DEFAULT", "EH2ET"])]))
            if subj_tok_span[0] <= obj_tok_span[0]:
                add_spot((subj_tok_span[0], obj_tok_span[0], self.tag2id[self.separator.join([rel, "SH2OH"])]))
            else:
                add_spot((obj_tok_span[0], subj_tok_span[0], self.tag2id[self.separator.join([rel, "OH2SH"])]))
            if subj_tok_span[1] <= obj_tok_span[1]:
                add_spot((subj_tok_span[1] - 1, obj_tok_span[1] - 1, self.tag2id[self.separator.join([rel, "ST2OT"])]))
            else:
                add_spot((obj_tok_span[1] - 1, subj_tok_span[1] - 1, self.tag2id[self.separator.join([rel, "OT2ST"])]))
        return matrix_spots

    def spots2shaking_tag(self, spots):
        '''
        convert spots to matrix tag
        spots: [(start_ind, end_ind, tag_id), ]
        return: 
            shaking_tag: (shaking_seq_len, tag_size)
        '''
        shaking_seq_len = self.matrix_size * (self.matrix_size + 1) // 2
        shaking_tag = torch.zeros(shaking_seq_len, len(self.tag2id)).long()
        for sp in spots:
            shaking_idx = self.matrix_idx2shaking_idx[sp[0]][sp[1]]
            shaking_tag[shaking_idx][sp[2]] = 1
        return shaking_tag

    def spots2shaking_tag4batch(self, batch_spots):
        '''
        batch_spots: a batch of spots, [spots1, spots2, ...]
            spots: [(start_ind, end_ind, tag_id), ]
        return: 
            batch_shaking_tag: (batch_size, shaking_seq_len, tag_size)
        '''
        shaking_seq_len = self.matrix_size * (self.matrix_size + 1) // 2
        batch_shaking_tag = torch.zeros(len(batch_spots), shaking_seq_len, len(self.tag2id)).long()
        for batch_id, spots in enumerate(batch_spots):
            for sp in spots:
                shaking_idx = self.matrix_idx2shaking_idx[sp[0]][sp[1]]
                batch_shaking_tag[batch_id][shaking_idx][sp[2]] = 1
        return batch_shaking_tag

    def get_spots_fr_shaking_tag(self, shaking_tag):
        '''
        shaking_tag -> spots
        shaking_tag: (shaking_seq_len, tag_id)
        spots: [(start_ind, end_ind, tag_id), ]
        '''
        spots = []
        nonzero_points = torch.nonzero(shaking_tag, as_tuple=False)
        for point in nonzero_points:
            shaking_idx, tag_idx = point[0].item(), point[1].item()
            pos1, pos2 = self.shaking_idx2matrix_idx[shaking_idx]
            spot = (pos1, pos2, tag_idx)
            spots.append(spot)
        return spots

    def decode_rel(self,
                   text,
                   shaking_tag,
                   tok2char_span,
                   glod_ent,
                   tok_offset=0, char_offset=0):
        '''
        shaking_tag: (shaking_seq_len, tag_id_num)
        '''
        rel_list = []
        matrix_spots = self.get_spots_fr_shaking_tag(shaking_tag)
        # gold opinion
        gold_opinion_set = set()
        for ent in glod_ent:
            if ent['type'] == 'Opinion':
                gold_opinion_set.add("{}\u2E80{}".format(ent['tok_span'][0], ent["tok_span"][1]))

        # entity
        head_ind2entities = {}
        ent_list = []
        for sp in matrix_spots:
            tag = self.id2tag[sp[2]]
            ent_type, link_type = tag.split(self.separator)
            if link_type != "EH2ET" or sp[0] > sp[
                1]:  # for an entity, the start position can not be larger than the end pos.
                continue

            char_span_list = tok2char_span[sp[0]:sp[1] + 1]
            char_sp = [char_span_list[0][0], char_span_list[-1][1]]
            ent_text = text[char_sp[0]:char_sp[1]]
            entity = {
                "type": ent_type,
                "text": ent_text,
                "tok_span": [sp[0], sp[1] + 1],
                "char_span": char_sp,
            }
            head_key = str(sp[0])  # take ent_head_pos as the key to entity list
            if head_key not in head_ind2entities:
                head_ind2entities[head_key] = []
            head_ind2entities[head_key].append(entity)
            ent_list.append(entity)

        # tail link
        tail_link_memory_set = set()
        for sp in matrix_spots:
            tag = self.id2tag[sp[2]]
            rel, link_type = tag.split(self.separator)
            if link_type == "ST2OT":
                tail_link_memory = self.separator.join([rel, str(sp[0]), str(sp[1])])
                tail_link_memory_set.add(tail_link_memory)
            elif link_type == "OT2ST":
                tail_link_memory = self.separator.join([rel, str(sp[1]), str(sp[0])])
                tail_link_memory_set.add(tail_link_memory)

        # head link
        head_link_memory_set = set()
        for sp in matrix_spots:
            tag = self.id2tag[sp[2]]
            rel, link_type = tag.split(self.separator)
            if link_type == "SH2OH":
                head_link_memory = self.separator.join([rel, str(sp[0]), str(sp[1])])
                head_link_memory_set.add(head_link_memory)
            elif link_type == "OH2SH":
                head_link_memory = self.separator.join([rel, str(sp[1]), str(sp[0])])
                head_link_memory_set.add(head_link_memory)

            # go over all subj-obj pair to check whether the tail link exists
            for subj in ent_list:
                if "expression" not in subj['type']:
                    continue

                for obj in ent_list:
                    if "expression" in obj['type']:
                        continue
                    rel = 'polar_expression-' + obj['type']
                    head_link_memory = self.separator.join([rel, str(subj["tok_span"][0]), str(obj["tok_span"][0])])
                    tail_link_memory = self.separator.join(
                        [rel, str(subj["tok_span"][1] - 1), str(obj["tok_span"][1] - 1)])
                    # or: strict; and: loose
                    if tail_link_memory not in tail_link_memory_set and head_link_memory not in head_link_memory_set:
                        # no such relation
                        continue
                    rel_list.append({
                        "subject": subj["text"],
                        "object": obj["text"],
                        "subj_tok_span": [subj["tok_span"][0] + tok_offset, subj["tok_span"][1] + tok_offset],
                        "obj_tok_span": [obj["tok_span"][0] + tok_offset, obj["tok_span"][1] + tok_offset],
                        "subj_char_span": [subj["char_span"][0] + char_offset, subj["char_span"][1] + char_offset],
                        "obj_char_span": [obj["char_span"][0] + char_offset, obj["char_span"][1] + char_offset],
                        "predicate": rel,
                    })
            # recover the positons in the original text
            for ent in ent_list:
                ent["char_span"] = [ent["char_span"][0] + char_offset, ent["char_span"][1] + char_offset]
                ent["tok_span"] = [ent["tok_span"][0] + tok_offset, ent["tok_span"][1] + tok_offset]
        return rel_list, ent_list


class WholeLabelsTagging(object):
    def __init__(self, rel2id, max_seq_len, entity_type2id):
        super().__init__()
        self.rel2id = rel2id
        self.id2rel = {ind: rel for rel, ind in rel2id.items()}

        self.separator = "\u2E80"

        self.ent2id = entity_type2id
        self.id2ent = {ind: ent for ent, ind in self.ent2id.items()}

        self.tags = {"rel", "ent", "global"} # "rel", "ent", "global"
        self.tags = sorted(self.tags)

        self.tag2id = {t: idx for idx, t in enumerate(self.tags)}
        print(self.tag2id)
        self.id2tag = {idx: t for t, idx in self.tag2id.items()}
        self.matrix_size = max_seq_len

        # map
        # e.g. [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
        self.shaking_idx2matrix_idx = [(ind, end_ind) for ind in range(self.matrix_size) for end_ind in
                                       list(range(self.matrix_size))[ind:]]

        self.matrix_idx2shaking_idx = [[0 for i in range(self.matrix_size)] for j in range(self.matrix_size)]
        for shaking_ind, matrix_ind in enumerate(self.shaking_idx2matrix_idx):
            self.matrix_idx2shaking_idx[matrix_ind[0]][matrix_ind[1]] = shaking_ind

    def get_tag_size(self):
        return len(self.tag2id)

    def get_spots(self, sample):
        '''
        matrix_spots: [(tok_pos1, tok_pos2, tag_id), ]
        '''
        matrix_spots = []
        spot_memory_set = set()

        def add_spot(spot):
            memory = "{},{},{}".format(*spot)
            if memory not in spot_memory_set:
                matrix_spots.append(spot)
                spot_memory_set.add(memory)

        for ent in sample["entity_list"]:
            for i in range(ent["tok_span"][0], ent["tok_span"][1]):
                add_spot((0, i, self.tag2id["global"]))
                for j in range(i+1, ent["tok_span"][1]):
                    add_spot((i, j, self.tag2id["ent"]))

        for rel in sample["relation_list"]:
            subj_tok_span = rel["subj_tok_span"]
            obj_tok_span = rel["obj_tok_span"]
            rel = rel["predicate"]
            for i in range(subj_tok_span[0], subj_tok_span[-1]):
                for j in range(obj_tok_span[0], obj_tok_span[-1]):
                    if i <= j:
                        add_spot((i, j, self.tag2id["rel"]))
                    else:
                        add_spot((j, i, self.tag2id["rel"]))
        return matrix_spots

    def spots2shaking_tag(self, spots):
        '''
        convert spots to matrix tag
        spots: [(start_ind, end_ind, tag_id), ]
        return:
            shaking_tag: (shaking_seq_len, tag_size)
        '''
        shaking_seq_len = self.matrix_size * (self.matrix_size + 1) // 2
        shaking_tag = torch.zeros(shaking_seq_len, len(self.tag2id)).long()
        for sp in spots:
            shaking_idx = self.matrix_idx2shaking_idx[sp[0]][sp[1]]
            shaking_tag[shaking_idx][sp[2]] = 1
        return shaking_tag

    def spots2shaking_tag4batch(self, batch_spots):
        '''
        batch_spots: a batch of spots, [spots1, spots2, ...]
            spots: [(start_ind, end_ind, tag_id), ]
        return:
            batch_shaking_tag: (batch_size, shaking_seq_len, tag_size)
        '''
        shaking_seq_len = self.matrix_size * (self.matrix_size + 1) // 2
        batch_shaking_tag = torch.zeros(len(batch_spots), shaking_seq_len, len(self.tag2id)).long()
        for batch_id, spots in enumerate(batch_spots):
            for sp in spots:
                shaking_idx = self.matrix_idx2shaking_idx[sp[0]][sp[1]]
                batch_shaking_tag[batch_id][shaking_idx][sp[2]] = 1
        return batch_shaking_tag

    def get_spots_fr_shaking_tag(self, shaking_tag):
        '''
        shaking_tag -> spots
        shaking_tag: (shaking_seq_len, tag_id)
        spots: [(start_ind, end_ind, tag_id), ]
        '''
        spots = []
        nonzero_points = torch.nonzero(shaking_tag, as_tuple=False)
        for point in nonzero_points:
            shaking_idx, tag_idx = point[0].item(), point[1].item()
            pos1, pos2 = self.shaking_idx2matrix_idx[shaking_idx]
            spot = (pos1, pos2, tag_idx)
            spots.append(spot)
        return spots

class DataMaker():
    def __init__(self, bert_tokenizer, text2indices, get_tok2char_span_map, shaking_tagger, shaking_tagger4graph):
        self.bert_tokenizer = bert_tokenizer
        self.text2indices = text2indices
        self.shaking_tagger = shaking_tagger
        self.shaking_tagger4graph = shaking_tagger4graph
        self.get_tok2char_span_map = get_tok2char_span_map

    def get_indexed_data(self, data, max_seq_len, max_tok_len, data_type="train"):
        indexed_samples = []
        for ind, sample in tqdm(enumerate(data), desc="Generate indexed train or valid data"):
            text = sample["text"]
            pos_list = sample['pos_list']

            lemma_list = sample['lemma_list']
            tokens = text.split(" ")
            for i, token in enumerate(tokens):
                token_tokenized = self.bert_tokenizer(token, return_tensors='pt', add_special_tokens=False)
                subsubtokens = self.bert_tokenizer.convert_ids_to_tokens(token_tokenized["input_ids"][0])
                if len(subsubtokens) == 0:
                    tokens[i] = '[UNK]'
            tokens.extend(['[PAD]'] * (max_seq_len - len(tokens)))
            bert_text = " ".join(tokens)

            # tagging
            matrix_spots = None
            matrix_spots4graph = None
            if data_type != "test":
                matrix_spots = self.shaking_tagger.get_spots(sample)
                matrix_spots4graph = self.shaking_tagger4graph.get_spots(sample)

            tok2char_span = self.get_tok2char_span_map(text)
            tok2char_span.extend([(-1, -1)] * (max_seq_len - len(tok2char_span)))
            input_ids, pos_ids, lemma_ids, rel_len, char_ids, rel_tok_lens = self.text2indices(text, pos_list, lemma_list, max_seq_len, max_tok_len)

            sample_tp = (sample,
                         input_ids,
                         tok2char_span,
                         matrix_spots,
                         matrix_spots4graph,
                         rel_len,
                         char_ids,
                         rel_tok_lens,
                         bert_text,
                         pos_ids, lemma_ids,
                         data_type
                         )
            indexed_samples.append(sample_tp)
        return indexed_samples

    def generate_batch(self, batch_data, data_type="train"):
        sample_list = []
        input_ids_list = []
        tok2char_span_list = []
        matrix_spots_list = []
        matrix_spots4graph_list = []
        rel_len_list = []
        char_ids_list = []
        rel_tok_lens_list = []
        bert_input_ids_list = []
        bert_attention_mask_list = []
        bert_token_type_ids_list = []
        bert_text_list = []
        pos_ids_list = []
        lemma_ids_list = []
        sample_weight_list = []

        max_batch_len = 0
        for tp in batch_data:
            tokens = self.bert_tokenizer.tokenize(tp[8])
            max_batch_len = max(max_batch_len, len(tokens))
            for _ in range(1):
                sample_list.append(tp[0])
                if tp[11] == "train":
                    sample_weight_list.append(tp[0]['weight'])
                input_ids_list.append(tp[1])
                tok2char_span_list.append(tp[2])
                rel_len_list.append(tp[5])
                char_ids_list.append(tp[6])
                rel_tok_lens_list.append(tp[7])
                bert_text_list.append(tp[8])
                pos_ids_list.append(tp[9])
                lemma_ids_list.append(tp[10])
                if data_type != "test":
                    matrix_spots_list.append(tp[3])
                    matrix_spots4graph_list.append(tp[4])

        for tp in batch_data:
            # codes for bert input

            codes = self.bert_tokenizer.encode_plus(tp[8],
                                    return_offsets_mapping = True,
                                    add_special_tokens = False,
                                    max_length = max_batch_len,
                                    padding = 'max_length')
            bert_input_ids = torch.tensor(codes["input_ids"]).long()
            bert_attention_mask = torch.tensor(codes["attention_mask"]).long()
            bert_token_type_ids = torch.tensor(codes["token_type_ids"]).long()
            for _ in range(1):
                bert_input_ids_list.append(bert_input_ids)
                bert_attention_mask_list.append(bert_attention_mask)
                bert_token_type_ids_list.append(bert_token_type_ids)

        bert_input_ids = torch.stack(bert_input_ids_list, dim=0)
        bert_attention_mask = torch.stack(bert_attention_mask_list, dim=0)
        bert_token_type_ids = torch.stack(bert_token_type_ids_list, dim=0)
        batch_input_ids = torch.stack(input_ids_list, dim=0)
        batch_char_ids = torch.stack(char_ids_list, dim=0)
        batch_pos_ids = torch.stack(pos_ids_list, dim=0)
        batch_lemma_ids = torch.stack(lemma_ids_list, dim=0)
        if tp[11] == "train":
            batch_sample_weight_list = torch.tensor(sample_weight_list)
        else:
            batch_sample_weight_list = torch.zeros(1)
        batch_shaking_tag = None
        batch_shaking_tag4graph = None
        if data_type != "test":
            batch_shaking_tag = self.shaking_tagger.spots2shaking_tag4batch(matrix_spots_list)
            batch_shaking_tag4graph = self.shaking_tagger4graph.spots2shaking_tag4batch(matrix_spots4graph_list)

        return sample_list, \
               batch_input_ids, tok2char_span_list, \
               batch_shaking_tag, batch_shaking_tag4graph, rel_len_list, batch_char_ids, rel_tok_lens_list, \
               bert_input_ids, bert_attention_mask, bert_token_type_ids, bert_text_list, batch_pos_ids, batch_lemma_ids, batch_sample_weight_list

class TGLS(nn.Module):
    def __init__(self, data_set,
                 bert_encoder,
                 bert_tokenizer,
                 init_word_embedding_matrix,
                 emb_dropout_rate,
                 enc_hidden_size,
                 dec_hidden_size,
                 rnn_dropout_rate,
                 essential_tag_size,
                 whole_tag_size,
                 char_size,
                 char_emb_size,
                 pos_size,
                 lemma_size,
                 pos_emb_size=100,
                 lemma_emb_size=100
                 ):
        super().__init__()
        self.bert_hidden_size = bert_encoder.config.hidden_size
        self.bert_encoder = bert_encoder
        self.bert_tokenizer = bert_tokenizer

        self.word_embeds = nn.Embedding.from_pretrained(init_word_embedding_matrix, freeze=True)
        self.char_embeds = nn.Embedding(char_size, char_emb_size)
        self.pos_embeds = nn.Embedding(pos_size, pos_emb_size)
        self.lemma_embeds = nn.Embedding(lemma_size, lemma_emb_size)
        self.emb_dropout = nn.Dropout(emb_dropout_rate)

        # char cnn layer
        # Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1...
        kernerls = [2, 3, 4, 5]
        out_channels = 25
        self.char_cnns = nn.ModuleList(
            [nn.Conv1d(char_emb_size, out_channels, int(kernel_size),
                       stride=1, padding=0, bias=False) for kernel_size in kernerls])
        out_char_emb_size = len(kernerls) * out_channels

        self.enc_lstm = EnhancedLSTM(
            "drop_connect",
            init_word_embedding_matrix.size()[-1] + pos_emb_size + lemma_emb_size + out_char_emb_size,
            enc_hidden_size // 2,
            num_layers=2,
            ff_dropout=0.3,
            recurrent_dropout=0.3,
            bidirectional=True)

        self.dec_lstm = EnhancedLSTM(
            "drop_connect",
            enc_hidden_size,
            dec_hidden_size // 2,
            num_layers=2,
            ff_dropout=0.3,
            recurrent_dropout=0.3,
            bidirectional=True)

        hidden_size = bert_encoder.config.hidden_size + dec_hidden_size

        # handshaking kernel
        self.token_pair_modeling = TokenPairModeling(hidden_size, essential_tag_size, whole_tag_size, data_set=data_set)

    def average_reps(self, text_list, bert_reps, seq_len):

        final_reps = []
        for sent, bert_rep in zip(text_list, bert_reps):
            sent_tokenized = self.bert_tokenizer(sent, return_tensors='pt', add_special_tokens=False)
            subtokens = self.bert_tokenizer.convert_ids_to_tokens(sent_tokenized["input_ids"][0])

            sent_reps = []
            sub_reps = []
            i = 0
            k = 0
            tokens = sent.split(' ')
            # print(len(tokens), len(subtokens), len(bert_rep))
            for j, (tok, rep) in enumerate(zip(subtokens, bert_rep[:len(subtokens)])):
                sub_reps.append(rep)
                token_tokenized = self.bert_tokenizer(tokens[i], return_tensors='pt', add_special_tokens=False)
                subsubtokens = self.bert_tokenizer.convert_ids_to_tokens(token_tokenized["input_ids"][0])
                if k == len(subsubtokens) - 1:
                    ave_rep = torch.stack(sub_reps, dim=0).mean(0)
                    # ave_rep = sub_reps[0]
                    sent_reps.append(ave_rep)
                    sub_reps = []
                    i += 1
                    k = 0
                else:
                    k += 1
            sent_reps.extend([])
            sent_reps = torch.stack(sent_reps, dim=0)
            final_reps.append(sent_reps)
        final_reps = torch.stack(final_reps, dim=0)

        return final_reps

    def init_masks(self, batch_size, lengths, max_sent_length, rel_tok_lens, max_tok_len):
        # max_sent_length = max(lengths)
        indices = torch.arange(0, max_sent_length).unsqueeze(0).expand(batch_size, -1)
        masks = indices < lengths.unsqueeze(1)
        masks = masks.type(torch.FloatTensor)

        char_indices = torch.arange(0, max_tok_len)[None, None, :].repeat(batch_size, max_sent_length, 1)
        char_masks = char_indices < rel_tok_lens.unsqueeze(2)
        char_masks = char_masks.type(torch.FloatTensor)

        return masks, char_masks

    def forward(self, batch_sample_weight_list, text_list, batch_pos_ids, batch_lemma_ids, bert_input_ids_list, bert_attention_mask_list, bert_token_type_ids_list, input_ids,
                rel_lens, char_ids, rel_tok_lens):
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        tok_len = char_ids.size(2)
        encodings = []
        context_outputs = self.bert_encoder(bert_input_ids_list, bert_attention_mask_list, bert_token_type_ids_list,
                                            return_dict=True,
                                            output_hidden_states=True)
        bert_reps = context_outputs.last_hidden_state
        bert_outputs = self.average_reps(text_list, bert_reps, seq_len)
        bert_outputs = bert_outputs.to(input_ids.device)
        masks, char_masks = self.init_masks(batch_size, torch.Tensor(rel_lens), seq_len, torch.Tensor(rel_tok_lens),
                                            tok_len)
        masks = masks.to(input_ids.device)        

        char_masks = char_masks.to(input_ids.device)
        word_embedding = self.word_embeds(input_ids)
        pos_embedding = self.pos_embeds(batch_pos_ids)
        lemma_embedding = self.lemma_embeds(batch_lemma_ids)
        word_embedding = self.emb_dropout(word_embedding)
        pos_embedding = self.emb_dropout(pos_embedding)
        lemma_embedding = self.emb_dropout(lemma_embedding)
        # [b, seq_len, tok_len, char_emb]
        char_emb = self.char_embeds(char_ids.long())
        char_emb = char_emb * char_masks.unsqueeze(-1)

        char_emb = char_emb.view(batch_size * seq_len, tok_len, -1)
        # [Batch_size, chr_emb_size, max_token_len]
        char_emb = char_emb.transpose(1, 2)
        chars_output = []
        for i, cnn in enumerate(self.char_cnns):
            chars_cnn_embedding = torch.relu(cnn.forward(char_emb))
            pooled_chars_cnn_emb, _ = chars_cnn_embedding.max(2)
            chars_output.append(pooled_chars_cnn_emb)
        chars_output_emb = torch.cat(chars_output, 1)
        chars_output_emb = chars_output_emb.view(batch_size, seq_len, -1)

        # bsz * seq_len * (100 + 100 + 100 + 100)
        embedding = torch.cat([word_embedding, pos_embedding, lemma_embedding, chars_output_emb], dim=-1)

        # enhanced lstm
        sorted_lens, idx_sort = torch.sort(torch.Tensor(rel_lens), dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0, descending=False)
        embedding = embedding[idx_sort]
        lstm_outputs = self.enc_lstm(embedding, None, sorted_lens)
        lstm_outputs = self.dec_lstm(lstm_outputs, None, sorted_lens)
        lstm_outputs = lstm_outputs[idx_unsort]
        final_outputs = torch.cat([lstm_outputs, bert_outputs], dim=-1)

        attention_scoring4pred, threshold4pred, total_attention_scoring4graph, total_threshold4graph = self.token_pair_modeling(
            final_outputs, masks[:, :, None])

        return attention_scoring4pred, threshold4pred, total_attention_scoring4graph, total_threshold4graph


class MetricsCalculator():
    def __init__(self, shaking_tagger):
        self.shaking_tagger = shaking_tagger
        self.last_weights = None  # for exponential moving averaging
        
    def get_sample_accuracy(self, pred, truth):
        '''
        计算该batch的pred与truth全等的样本比例
        '''
        pred = pred.view(pred.size()[0], -1)
        truth = truth.view(truth.size()[0], -1)

        # (batch_size, )，每个元素是pred与truth之间tag相同的数量
        correct_tag_num = torch.sum(torch.eq(truth, pred).float(), dim=1)

        # seq维上所有tag必须正确，所以correct_tag_num必须等于seq的长度才算一个correct的sample
        sample_acc_ = torch.eq(correct_tag_num, torch.ones_like(correct_tag_num) * truth.size()[-1]).float()
        sample_acc = torch.mean(sample_acc_)

        return sample_acc        

    # loss func
    def _multilabel_categorical_crossentropy(self, y_pred, y_true, cr_ceil, cr_low, batch_sample_weight_list):
        """
        y_pred: (batch_size, shaking_seq_len, type_size)
        y_true: (batch_size, shaking_seq_len, type_size)
        y_true and y_pred have the same shape，elements in y_true are either 0 or 1，
             1 tags positive classes，0 tags negtive classes(means tok-pair does not have this type of link).
        """

        y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
        y_pred_neg = y_pred - y_true * 1e12  # mask the pred oudtuts of pos classes
        y_pred_pos = y_pred - (1 - y_true) * 1e12  # mask the pred oudtuts of neg classes
        y_pred_neg = torch.cat([y_pred_neg, cr_ceil], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, -cr_low], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

        return ((neg_loss + pos_loss + cr_low.squeeze(-1) - cr_ceil.squeeze(-1))).mean()

    def get_span_cpg(self, gold_ent_list, pred_ent_list, cpg):

        def get_token_set(ent_list):
            # token level set
            o_set, h_set, t_set = set(), set(), set()
            for ent in ent_list:
                if "expression" in ent["type"].lower():
                    if (ent["tok_span"][1] - ent["tok_span"][0]) >= 1 and (
                            ent["tok_span"][1] - ent["tok_span"][0]) <= 4000:
                        o_set.update(list(range(ent["tok_span"][0], ent["tok_span"][1])))
                elif ent["type"].lower() == "source":
                    h_set.update(list(range(ent["tok_span"][0], ent["tok_span"][1])))
                elif ent["type"].lower() == "target":
                    if (ent["tok_span"][1] - ent["tok_span"][0]) >= 1 and (
                            ent["tok_span"][1] - ent["tok_span"][0]) <= 16000:
                        t_set.update(list(range(ent["tok_span"][0], ent["tok_span"][1])))
            return o_set, h_set, t_set

        gold_dict, pred_dict = dict(), dict()
        gold_dict['expression'], gold_dict['holder'], gold_dict['target'] = get_token_set(gold_ent_list)
        pred_dict['expression'], pred_dict['holder'], pred_dict['target'] = get_token_set(pred_ent_list)

        for key in gold_dict.keys():
            cpg['overall_span'][0] += len(pred_dict[key].intersection(gold_dict[key]))
            cpg['overall_span'][1] += len(pred_dict[key])
            cpg['overall_span'][2] += len(gold_dict[key])
            cpg[key][0] += len(pred_dict[key].intersection(gold_dict[key]))
            cpg[key][1] += len(pred_dict[key])
            cpg[key][2] += len(gold_dict[key])

    def get_only_rel_cpg(self, gold_rel_list, pred_rel_list, cpg):
        pred_total_num = 0
        gold_total_num = 0
        pred_true_num = 0
        gold_true_num = 0
        is_choose = True

        if is_choose:
            for gold_rel in gold_rel_list:
                gold_predicate = gold_rel["predicate"].lower()
                gold_subj_set = frozenset(list(range(gold_rel["subj_tok_span"][0], gold_rel["subj_tok_span"][1])))
                gold_obj_set = frozenset(list(range(gold_rel["obj_tok_span"][0], gold_rel["obj_tok_span"][1])))        
                for pred_rel in pred_rel_list:
                    pred_predicate = pred_rel["predicate"].lower()
                    pred_subj_set = frozenset(list(range(pred_rel["subj_tok_span"][0], pred_rel["subj_tok_span"][1])))
                    pred_obj_set = frozenset(list(range(pred_rel["obj_tok_span"][0], pred_rel["obj_tok_span"][1])))
                    if gold_predicate == pred_predicate and len(gold_subj_set.intersection(pred_subj_set)) > 0 and \
                            len(gold_obj_set.intersection(pred_obj_set)) > 0:
                        cpg["only_rel"][2] += 1
                        gold_true_num += 1
                        break
                gold_total_num += 1
                            
            for pred_rel in pred_rel_list:
                pred_predicate = pred_rel["predicate"].lower()
                pred_subj_set = frozenset(list(range(pred_rel["subj_tok_span"][0], pred_rel["subj_tok_span"][1])))
                pred_obj_set = frozenset(list(range(pred_rel["obj_tok_span"][0], pred_rel["obj_tok_span"][1])))            
                for gold_rel in gold_rel_list:
                    gold_predicate = gold_rel["predicate"].lower()
                    gold_subj_set = frozenset(list(range(gold_rel["subj_tok_span"][0], gold_rel["subj_tok_span"][1])))
                    gold_obj_set = frozenset(list(range(gold_rel["obj_tok_span"][0], gold_rel["obj_tok_span"][1])))            
                    if gold_predicate == pred_predicate and len(gold_subj_set.intersection(pred_subj_set)) > 0 and \
                            len(gold_obj_set.intersection(pred_obj_set)) > 0:
                        cpg["only_rel"][0] += 1
                        pred_true_num += 1
                        break
                pred_total_num += 1  
            cpg["only_rel"][1] += pred_total_num - pred_true_num
            cpg["only_rel"][3] += gold_total_num - gold_true_num

    def get_senti_cpg(self, gold_ent_list, pred_ent_list, cpg):
        pred_total_num = 0
        gold_total_num = 0
        pred_true_num = 0
        gold_true_num = 0
        for pred_exp in pred_ent_list:
            if 'expression' in pred_exp["type"].lower():
                pred_total_num += 1
                pred_exp_set = frozenset(list(range(pred_exp["tok_span"][0], pred_exp["tok_span"][1])))
                pred_polar = pred_exp["type"].split(':')[-1]
                for gold_exp in gold_ent_list:
                    if 'expression' in gold_exp["type"].lower():
                        gold_exp_set = frozenset(list(range(gold_exp["tok_span"][0], gold_exp["tok_span"][1])))
                        gold_polar = gold_exp["type"].split(':')[-1]
                        if pred_polar.lower() == gold_polar.lower() and len(gold_exp_set.intersection(pred_exp_set)) > 0:
                            cpg["only_senti"][0] += 1
                            pred_true_num += 1
                            break
        for gold_exp in gold_ent_list:
            if 'expression' in gold_exp["type"].lower():
                gold_total_num += 1
                gold_exp_set = frozenset(list(range(gold_exp["tok_span"][0], gold_exp["tok_span"][1])))
                gold_polar = gold_exp["type"].split(':')[-1]
                for pred_exp in pred_ent_list:
                    if 'expression' in pred_exp["type"].lower():
                        pred_exp_set = frozenset(list(range(pred_exp["tok_span"][0], pred_exp["tok_span"][1])))
                        pred_polar = pred_exp["type"].split(':')[-1]
                        if pred_polar.lower() == gold_polar.lower() and len(gold_exp_set.intersection(pred_exp_set)) > 0:
                            cpg["only_senti"][2] += 1
                            gold_true_num += 1
                            break
        cpg["only_senti"][1] += pred_total_num - pred_true_num
        cpg["only_senti"][3] += gold_total_num - gold_true_num

    def get_targeted_cpg(self, gold_ent_list, gold_rel_list, pred_ent_list, pred_rel_list, cpg):

        def get_targeted_set(ent_list, rel_list):
            targeted_set = set()
            for ent in ent_list:
                if ent["type"].lower() == "target" and (ent["tok_span"][1] - ent["tok_span"][0]) >= 1 and (
                            ent["tok_span"][1] - ent["tok_span"][0]) <= 1200:
                    for rel in rel_list:
                        if rel["predicate"].lower() == 'polar_expression-target' and ent["tok_span"][0] == rel["obj_tok_span"][
                            0] and ent["tok_span"][1] == rel["obj_tok_span"][1]:
                            # find expression to get polar
                            for exp in ent_list:
                                #  for some rare situation, one target may have multiple expression,
                                if 'expression' in exp["type"].lower() and exp["tok_span"][0] == rel["subj_tok_span"][0] and \
                                        exp["tok_span"][1] == rel["subj_tok_span"][1]:
                                    polar = exp["type"].split(':')[-1]
                                    targeted_set.add((ent["tok_span"][0], ent["tok_span"][1], polar.lower()))

            return targeted_set

        gold_targeted_set = get_targeted_set(gold_ent_list, gold_rel_list)
        pred_targeted_set = get_targeted_set(pred_ent_list, pred_rel_list)
        # print(gold_targeted_set, pred_targeted_set)
        cpg["targeted"][0] += len(pred_targeted_set.intersection(gold_targeted_set))
        cpg["targeted"][1] += len(pred_targeted_set)
        cpg["targeted"][2] += len(gold_targeted_set)

    def get_sent_graph_cpg(self, gold_ent_list, gold_rel_list, pred_ent_list, pred_rel_list, cpg, keep_polarity=True,
                           weighted=True):

        def get_sent_tuples_list(ent_list, rel_list):
            sent_tuples_set = set()
            for exp in ent_list:
                current_targets = []
                current_holders = []
                if 'expression' in exp["type"].lower():
                    polar = exp["type"].split(':')[-1]
                    for rel in rel_list:
                        if rel["predicate"].lower() == 'polar_expression-target' and exp["tok_span"][0] == rel["subj_tok_span"][
                            0] and exp["tok_span"][1] == rel["subj_tok_span"][1]:
                            current_targets.append(list(range(rel["obj_tok_span"][0], rel["obj_tok_span"][1])))
                        if rel["predicate"].lower() == 'polar_expression-source' and exp["tok_span"][0] == rel["subj_tok_span"][
                            0] and exp["tok_span"][1] == rel["subj_tok_span"][1]:
                            current_holders.append(list(range(rel["obj_tok_span"][0], rel["obj_tok_span"][1])))
                    if current_targets == []:
                        current_targets = [[]]
                    if current_holders == []:
                        current_holders = [[]]
                    for target in current_targets:
                        for holder in current_holders:
                            avg_dis_set = set()
                            avg_dis_set.update((frozenset(holder)))
                            avg_dis_set.update(frozenset(target))
                            avg_dis_set.update(frozenset(list(range(exp["tok_span"][0], exp["tok_span"][1]))))

                            if len(avg_dis_set) >= 1 and \
                                    len(avg_dis_set) <= 2100:
                                sent_tuples_set.add((frozenset(holder), frozenset(target),
                                                     frozenset(list(range(exp["tok_span"][0], exp["tok_span"][1]))), polar.lower()))
            return list(sent_tuples_set)

        gold_sent_tuples_list = get_sent_tuples_list(gold_ent_list, gold_rel_list)
        pred_sent_tuples_list = get_sent_tuples_list(pred_ent_list, pred_rel_list)

        def sent_tuples_in_list(sent_tuple1, list_of_sent_tuples):
            holder1, target1, exp1, pol1 = sent_tuple1
            if len(holder1) == 0:
                holder1 = frozenset(["_"])
            if len(target1) == 0:
                target1 = frozenset(["_"])
            for holder2, target2, exp2, pol2 in list_of_sent_tuples:
                if len(holder2) == 0:
                    holder2 = frozenset(["_"])
                if len(target2) == 0:
                    target2 = frozenset(["_"])
                if len(holder1.intersection(holder2)) > 0 and len(target1.intersection(target2)) > 0 and len(
                        exp1.intersection(exp2)) > 0:
                    if keep_polarity:
                        if pol1 == pol2:
                            # print(holder1, target1, exp1, pol1)
                            # print(holder2, target2, exp2, pol2)
                            return True
                    else:
                        # print(holder1, target1, exp1, pol1)
                        # print(holder2, target2, exp2, pol2)
                        return True
            return False

        def weighted_score(sent_tuple1, list_of_sent_tuples):
            holder1, target1, exp1, pol1 = sent_tuple1
            if len(holder1) == 0:
                holder1 = frozenset(["_"])
            if len(target1) == 0:
                target1 = frozenset(["_"])
            for holder2, target2, exp2, pol2 in list_of_sent_tuples:
                if len(holder2) == 0:
                    holder2 = frozenset(["_"])
                if len(target2) == 0:
                    target2 = frozenset(["_"])
                if len(holder1.intersection(holder2)) > 0 and len(target1.intersection(target2)) > 0 and len(
                        exp1.intersection(exp2)) > 0:
                    holder_overlap = len(holder1.intersection(holder2)) / len(holder1)
                    target_overlap = len(target1.intersection(target2)) / len(target1)
                    exp_overlap = len(exp1.intersection(exp2)) / len(exp1)
                    return (holder_overlap + target_overlap + exp_overlap) / 3
            return 0

        # sent_graph precision
        for stuple in pred_sent_tuples_list:
            if sent_tuples_in_list(stuple, gold_sent_tuples_list):
                if weighted:
                    partial = weighted_score(stuple, gold_sent_tuples_list)
                    # print(partial)
                    if keep_polarity:
                        cpg["sent_graph"][0] += partial
                    else:
                        cpg["no_sent_graph"][0] += partial
                else:
                    if keep_polarity:
                        cpg["sent_graph"][0] += 1
                    else:
                        cpg["no_sent_graph"][0] += 1
            else:
                if keep_polarity:
                    cpg["sent_graph"][1] += 1
                else:
                    cpg["no_sent_graph"][1] += 1

        # sent_graph recall
        for stuple in gold_sent_tuples_list:
            if sent_tuples_in_list(stuple, pred_sent_tuples_list):
                if weighted:
                    partial = weighted_score(stuple, pred_sent_tuples_list)
                    if keep_polarity:
                        cpg["sent_graph"][2] += partial
                    else:
                        cpg["no_sent_graph"][2] += partial
                else:
                    if keep_polarity:
                        cpg["sent_graph"][2] += 1
                    else:
                        cpg["no_sent_graph"][2] += 1
            else:
                if keep_polarity:
                    cpg["sent_graph"][3] += 1
                else:
                    cpg["no_sent_graph"][3] += 1

    def get_cpg(self, sample_list,
                tok2char_span_list,
                batch_pred_shaking_tag):
        '''
        return correct number, predict number, gold number (cpg)
        '''

        cpg = {
            "expression": [0, 0, 0],
            "holder": [0, 0, 0],
            "target": [0, 0, 0],
            "targeted": [0, 0, 0],
            "no_sent_graph": [0, 0, 0, 0],
            "sent_graph": [0, 0, 0, 0],
            "only_senti": [0, 0, 0, 0],
            "overall_span": [0, 0, 0],
            "only_rel": [0, 0, 0, 0]
        }

        # go through all sentences
        for ind in range(len(sample_list)):
            sample = sample_list[ind]
            text = sample["text"]
            tok2char_span = tok2char_span_list[ind]
            pred_shaking_tag = batch_pred_shaking_tag[ind]
            gold_rel_list = sample["relation_list"]
            gold_ent_list = sample["entity_list"]
            pred_rel_list, pred_ent_list = self.shaking_tagger.decode_rel(text,
                                                                          pred_shaking_tag,
                                                                          tok2char_span,
                                                                          gold_ent_list)  # decoding
            self.get_span_cpg(gold_ent_list, pred_ent_list, cpg)
            self.get_targeted_cpg(gold_ent_list, gold_rel_list
                                  , pred_ent_list, pred_rel_list
                                  , cpg)
            self.get_sent_graph_cpg(gold_ent_list, gold_rel_list, pred_ent_list, pred_rel_list
                                    , cpg, keep_polarity=True, weighted=True)
            self.get_sent_graph_cpg(gold_ent_list, gold_rel_list, pred_ent_list, pred_rel_list
                                    , cpg, keep_polarity=False, weighted=True)
            self.get_senti_cpg(gold_ent_list, pred_ent_list, cpg)
            self.get_only_rel_cpg(gold_rel_list, pred_rel_list, cpg)

        return cpg

    def get_prf_scores(self, total_cpg_dict):
        minimini = 1e-12
        log_dict = {}
        for key in ['expression', 'holder', 'target', 'targeted', 'sent_graph', 'no_sent_graph', 'only_senti', 'overall_span', 'only_rel']:
            if key in ['expression', 'holder', 'target', 'targeted', 'overall_span']:
                precision = total_cpg_dict[key][0] / (total_cpg_dict[key][1] + minimini)
                recall = total_cpg_dict[key][0] / (total_cpg_dict[key][2] + minimini)
            if key in ['sent_graph', 'no_sent_graph', 'only_senti', 'only_rel']:
                precision = total_cpg_dict[key][0] / (total_cpg_dict[key][0] + total_cpg_dict[key][1] + minimini)
                recall = total_cpg_dict[key][2] / (total_cpg_dict[key][2] + total_cpg_dict[key][3] + minimini)
            f1 = 2 * precision * recall / (precision + recall + minimini)
            # log_dict[key + '_' + 'prec'] = precision
            # log_dict[key + '_' + 'recall'] = recall
            log_dict[key + '_' + 'f1'] = f1

        return log_dict
