import string
import random

common = {
    "exp_name": "norec", # ace05_lu
    "rel2id": "rel2id.json",
    "ent2id": "ent2id.json",
    "device_num": 0,
}
common["run_name"] = "{}".format(common["exp_name"]) + ""
seed = 1234

run_id = ''.join(random.sample(string.ascii_letters + string.digits, 8))
train_config = {
    "train_data": "train.json", 
    "valid_data": "valid.json",
    "rel2id": "rel2id.json",

    # if logger is set as default, uncomment the following four lines and comment the line above
    "logger": "default",
    "run_id": run_id,
    "log_path": "./default_log_dir/default.log",
    "path_to_save_model": "./default_log_dir/{}_{}".format(common["exp_name"], seed),

    # when to save the model state dict
    "f1_2_save": 0.25,
    # whether train_config from scratch
    "fr_scratch": True,
    # write down notes here if you want, it will be logged
    "note": "start from scratch",
    # if not fr scratch, set a model_state_dict
    "model_state_dict_path": "", # valid only if "fr_scratch" is False
    "hyper_parameters": {
        "batch_size": 8,
        "epochs": 60,
        "seed": seed, # 1234, 5678, 9101112, 13141516, 17181920
        "log_interval": 10,
        "max_seq_len": 100,
        "sliding_len": 20,
        "scheduler": "CAWR", # Step
    },
}

eval_config = {
    "model_state_dict_dir": "/home/shiwenxuan/workspace/ssa_master/src/default_log_dir/norec_1234/model_state_dict_11_6_0.3876.pt", 
    "test_data": "test.json", 

    "hyper_parameters": {
        "batch_size": 6,
        "force_split": False,
        "max_seq_len": 140,
        "sliding_len": 20,
        "tok_pair_sample_rate": 1,
    },
}

model_config = {
    "bert_path": "bert-base-multilingual-cased", # bert-base-cased， chinese-bert-wwm-ext-hit bert-base-multilingual-cased
    "data_home": "../data",
    "token2idx": "token2idx.json",
    "pos2idx": "pos2idx.json",
    "lemma2idx": "lemma2idx.json",
    "char_dict": "char_dict",
    "pretrained_word_embedding_path": "../pretrained_word_emb/Norwegian_100d.txt", # eu: Basque_100d.txt ca: Catalan_100d.txt Norwegian_100d.txt multi_100d.txt
    "hyper_parameters": {
         "lr": 3e-5,
         "enc_hidden_size": 1000,
         "dec_hidden_size": 800,
         "emb_dropout": 0.4,
         "rnn_dropout": 0.3,
         "word_embedding_dim": 100,
         "char_embedding_dim": 8
    },
}

cawr_scheduler = {
    # CosineAnnealingWarmRestarts
    "T_mult": 1,
    "rewarm_epoch_num": 2,
}
step_scheduler = {
    # StepLR
    "decay_rate": 0.999,
    "decay_steps": 100,
}

# ---------------------------dicts above is all you need to set---------------------------------------------------

hyper_params = {**model_config["hyper_parameters"], **train_config["hyper_parameters"]}
train_config = {**train_config, **common, **model_config}
train_config["hyper_parameters"] = hyper_params
if train_config["hyper_parameters"]["scheduler"] == "CAWR":
    train_config["hyper_parameters"] = {**train_config["hyper_parameters"], **cawr_scheduler}
elif train_config["hyper_parameters"]["scheduler"] == "Step":
    train_config["hyper_parameters"] = {**train_config["hyper_parameters"], **step_scheduler}
    
hyper_params = {**model_config["hyper_parameters"], **eval_config["hyper_parameters"]}
eval_config = {**eval_config, **common, **model_config}
eval_config["hyper_parameters"] = hyper_params
