# TGLS

**Effective Token Graph Modeling using a Novel Labeling Strategy for Structured Sentiment Analysis**

This repository contains the code of the official implementation for the paper: **[Effective Token Graph Modeling using a Novel Labeling Strategy for Structured Sentiment Analysis](https://aclanthology.org/2022.acl-long.291/).** The paper has been accepted to appear at **ACL 2022**. 

## Environment
* python==3.7.10
* torch==1.7.0
* transformers==4.11.3
* CUDA==11.2
* GPU: GeForce RTX 3090

## Data
We provide processed data in `./data` directory.  
### Data format:
```python
[
    {
        "id": "",
        "text": "",
        "pos_list": [],
        "lemma_list": [],
        "entity_list": 
            [
                {
                    "text": "",
                    "type": "",
                    "char_span": [],
                    "tok_span": []
                },
            ],
        "relation_list": 
            [
                {
                    "subject": "",
                    "object": "",
                    "predicate": "",
                    "subj_char_span": [],
                    "subj_tok_span": [],
                    "obj_char_span": [],
                    "obj_tok_span": []
                },
            ]
    }
]
```

## Word Embeddings
We use word embeddings openly available from the [NLPL vector repository](http://vectors.nlpl.eu/repository/), for English (model id 40 in the repo.), Basque (id 32), Catalan (id 34), and Norwegian (id 58).
To run our model, you should put the download `.txt` file of the word vector into the `./pretrained_word_emb` directory.

## Train
Set configuration in `src/config.py`:
```python
common = {
    "exp_name": "norec", # ca, eu, ds, mpqa
    "rel2id": "rel2id.json",
    "ent2id": "ent2id.json",
    "device_num": 0, # cuda idx
}
train_config = {
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
model_config = {
    "bert_path": "bert-base-multilingual-cased", 
    "data_home": "../data",
    "token2idx": "token2idx.json",
    "pos2idx": "pos2idx.json",
    "lemma2idx": "lemma2idx.json",
    "char_dict": "char_dict",
    "pretrained_word_embedding_path": "../pretrained_word_emb/Norwegian_100d.txt", # eu: Basque_100d.txt ca: Catalan_100d.txt Norwegian_100d.txt
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
```

Start training
```
cd src
python train.py
```

## Evaluation
Set configuration in `src/config.py`:
```python
eval_config = {
    "model_state_dict_dir": "./default_log_dir/norec_1234/model_state_dict_16_20_0.4059.pt", 
    "test_data": "test.json", 
    "hyper_parameters": {
        "batch_size": 6,
        "force_split": False,
        "max_seq_len": 140,
        "sliding_len": 20
    },
}
```
Start evaluation
```
cd src
python eval.py
```

## Citation
```
@inproceedings{shi-etal-2022-effective,
    title = "Effective Token Graph Modeling using a Novel Labeling Strategy for Structured Sentiment Analysis",
    author = "Shi, Wenxuan  and
      Li, Fei  and
      Li, Jingye  and
      Fei, Hao  and
      Ji, Donghong",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.291",
    pages = "4232--4241",
}
```

