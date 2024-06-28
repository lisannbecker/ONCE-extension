# ONCE: Boosting Content-based Recommendation with Open- and Closed-source LLMs

--- 

### A. Ibrahimi, L. Becker, R. van Emmerik

---

## Installations
### Requirements
- Linux or Windows with Python = 3.11

### Environment Setup
```bash
conda create --name once python=3.11 
conda activate once

pip install -r requirements_once.txt
```

### Model and Data Download
Navigate to the [Ekstra Bladet](https://recsys.eb.dk) website to download the small and/or large EB-NeRD dataset. Move to TODO XXX

Run the following scripts to download the [LLaMA 7b model](https://huggingface.co/huggyllama/llama-7b?library=transformers) and its embeddings:
```bash
python src/scripts/scripts_dire/1_download_llama_vocab_model.py

python src/scripts/scripts_dire/2_download_llama_embeddings.py
```

## Reproduction
### GENRE Implementation
```bash
conda activate genre
```

Move to the Once repository

```bash
cd src/lib/ONCE/
```

### Overview
We call GPT-3.5-turbo API provided by OpenAI as closed-source LLM. All the request codes are uploaded in this repository.

### Codes and corresponding generated data

|  Dataset  |            Schemes             |           Request Code           |                            Generated Data                            |
|:---------:|:------------------------------:|:--------------------------------:|:--------------------------------------------------------------------:|
|   EB-NeRD |       Content Summarizer       |       `news_summarizer.py`       |           `data/eb-nerd/eb-nerd-outputs/news_summarizer.log`         |
|   EB-NeRD |       User Profiler Train      |       `user_profiler.py`         |           `data/eb-nerd/eb-nerd-outputs/user_profiler.log`           |
|   EB-NeRD |       User Profiler Val        |       `user_profiler.py`         |           `data/eb-nerd/eb-nerd-outputs/user_profiler_val.log`       |
<!-- |   EB-NeRD | Personalized Content Generator | `personalized_news_generator.py` |           `data/eb-nerd/eb-nerd-outputs/generator_v3.log`            | -->

### DIRE Implementation
```bash
conda activate once
```

Move to the Legommenders repository and pre-train:
```bash
cd src/lib/Legommenders/

TODO
```

Training and testing with default parameters:
```bash
TODO
```

### ONCE
```bash
conda activate once
```

Move to the Legommenders repository and pre-train:
```bash
cd src/lib/Legommenders/

python worker.py --embed config/embed/llama-token.yaml --model config/model/llm/llama-fastformer-once.yaml --exp config/exp/llama-split-once.yaml --data config/data/eb-nerd-once.yaml --version small --llm_ver 7b --hidden_size 64 --layer 0 --lora 0 --fast_eval 0 --embed_hidden_size 4096 --page_size 8
```

Training and testing with default parameters:
```bash
python worker.py  --data config/data/eb-nerd-once.yaml --embed config/embed/llama-token.yaml  --model config/model/llm/llama-fastformer-once.yaml --exp config/exp/tt-llm.yaml --embed_hidden_size 4096 --llm_ver 7b --layer 31 --version small --lr 0.0001 --item_lr 0.00001 --batch_size 32 --acc_batch 2 --epoch_batch -4  
```

## Extension
### DIRE with Sentiment
```bash
conda activate once
```

Move to the Legommenders repository and pre-train:
```bash
cd src/lib/Legommenders/

python worker.py --embed config/embed/llama-token.yaml --model config/model/llm/llama-fastformer-sentiment.yaml --exp config/exp/llama-split-sentiment.yaml --data config/data/eb-nerd-sentiment.yaml --version small --llm_ver 7b --hidden_size 64 --layer 0 --lora 0 --fast_eval 0 --embed_hidden_size 4096 --page_size 8 --cuda -1
```

Training and testing with default parameters:
```bash
python worker.py  --data config/data/eb-nerd-sentiment.yaml --embed config/embed/llama-token.yaml  --model config/model/llm/llama-fastformer-sentiment.yaml --exp config/exp/tt-llm.yaml --embed_hidden_size 4096 --llm_ver 7b --layer 31 --version small --lr 0.0001 --item_lr 0.00001 --batch_size 32 --acc_batch 2 --epoch_batch -4 
```
