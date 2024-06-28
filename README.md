# ONCE: Boosting Content-based Recommendation with Open- and Closed-source LLMs

--- 

### A. Ibrahimi, L. Becker, R. van Emmerik

---

This project aims to reproduce and add an extension to the paper ["ONCE: Boosting Content-based Recommendation with Both Open- and Closed-source Large Language Models"](https://dl.acm.org/doi/pdf/10.1145/3616855.3635845?casa_token=OO-BH8lyeAwAAAAA:rtMOl0GST3o9NKwshn3hIYh2eBk_HCkbiIMPeO3gEBP3VQP8vpxT6pXpkutaOyNWseCGVy23iCED9g) by Liu et al. (2024). Please request our report for detailed information on the implementation of our reproduction and extension of the [ONCE architecture](https://github.com/frank-xwang/InstanceDiffusion).



## Installations
### Requirements
- Linux or Windows with Python = 3.11

### Environment Setup
For DIRE and ONCE:
```bash
conda create --name once python=3.11 
conda activate once

pip install -r requirements_once.txt
```
Or for the GENRE implementation:
```bash
conda create --name once python=3.8 
conda activate genre

pip install -r requirements_genre.txt
```

### Model and Data Download
Navigate to the [Ekstra Bladet](https://recsys.eb.dk) website to download the small and/or large EB-NeRD dataset. Move the data to your data to [`src/data/`](src/data/).

Run the following scripts to download the [LLaMA 7b model](https://huggingface.co/huggyllama/llama-7b?library=transformers) and its embeddings to [`src/data/`](src/data/):
```bash
python src/scripts/scripts_dire/1_download_llama_vocab_model.py

python src/scripts/scripts_dire/2_download_llama_embeddings.py
```

## Reproduction
### GENRE Implementation: Prompting Closed-source LLMs for Content-based Recommendation
```bash
conda activate genre
```

Move to the GENRE repository:
```bash
cd src/lib/GENRE/
```

Move or duplicate the EB-NeRD data to this folder for GENRE implementation [`src/lib/GENRE/data/eb-nerd/eb-nerd-data/ebnerd_small`](src/lib/GENRE/data/eb-nerd/eb-nerd-data/ebnerd_small). To convert the EB-NeRD articles to match the ONCE datatype, run:

```bash
python src/lib/GENRE/scripts_genre/articles_process.py
```

### Overview
We use the GPT-3.5-turbo API provided by OpenAI as a closed-source LLM. All the request codes are available in this repository.

### Codes and Corresponding Generated Data

|  Dataset  |            Schemes             |           Request Code           |                            Generated Data                            |
|:---------:|:------------------------------:|:--------------------------------:|:--------------------------------------------------------------------:|
|   EB-NeRD |       Content Summarizer       |       `news_summarizer.py`       |           `data/eb-nerd/eb-nerd-outputs/news_summarizer.log`         |
|   EB-NeRD |       User Profiler Train      |       `user_profiler.py`         |           `data/eb-nerd/eb-nerd-outputs/user_profiler.log`           |
|   EB-NeRD |       User Profiler Val        |       `user_profiler.py`         |           `data/eb-nerd/eb-nerd-outputs/user_profiler_val.log`       |
<!-- |   EB-NeRD | Personalized Content Generator | `personalized_news_generator.py` |           `data/eb-nerd/eb-nerd-outputs/generator_v3.log`            | -->

To update the datasets with the generated data, run these scripts:

```bash
python src/lib/GENRE/scripts_genre/update_articles.py
python src/lib/GENRE/scripts_genre/update_history.py
```

To obtain statistics of the augmented data, run the following script:

```bash
python src/lib/GENRE/scripts_genre/statistics_eb_nerd.py
```

### DIRE Implementation: Finetuning Open-source LLMs for Content-based Recommendation
```bash
conda activate once
```

Tokenize the EB-NeRD data located in [`src/data/`](src/data/) with the basic tokenizer:
```bash
python src/scripts/scripts_dire/3_processor_dire.py
python src/scripts/scripts_dire/4_processor_dire_llama.py
python src/scripts/scripts_dire/5_ebnerd_fusion_dire.py
```

Move to the Legommenders repository and pre-train. Replace {yourModel} with the recommender model you would like to use out of 'fastformer', 'naml', or 'nrms'. You will need to update the config files to point to [`src/data/`](src/data/):
```bash
cd src/lib/Legommenders/

python worker.py --embed config/embed/llama-token.yaml --model config/model/llm/llama-naml.yaml \
    --exp config/exp/llama-split.yaml --data config/data/eb-nerd.yaml --version small --llm_ver 13b \
    --hidden_size 64 --layer 0 --lora 0  --fast_eval 0 --embed_hidden_size 5120 --page_size 32 --cuda -1
```

Training and testing with default parameters:
```bash
cd src/lib/Legommenders/

python  worker.py --data config/data/eb-nerd.yaml --embed config/embed/llama-token.yaml \
     --model config/model/llm/llama-fastformer.yaml --exp config/exp/tt-llm.yaml --embed_hidden_size 4096 \
     --llm_ver 7b --layer 31 --version small --lr 0.0001   --item_lr 0.00001 --batch_size 64 --acc_batch 1 --epoch_batch -4
```

### ONCE
```bash
conda activate once
```

Tokenize the EB-NeRD data located in [`src/data/`](src/data/) with the tokenizer for the basic data and topics and region:
```bash
python src/scripts/scripts_dire/6_processor_once.py
python src/scripts/scripts_dire/7_processor_llama_once.py
python src/scripts/scripts_dire/8_ebnerd_fusion_once.py
```

Move to the Legommenders repository and pre-train. Replace {yourModel} with the recommender model you would like to use out of 'fastformer', 'naml', or 'nrms'. You will need to update the config files to point to [`src/data/`](src/data/):
```bash
cd src/lib/Legommenders/

python worker.py --embed config/embed/llama-token.yaml --model config/model/llm/llama-{yourModel}-once.yaml --exp config/exp/llama-split-once.yaml --data config/data/eb-nerd-once.yaml --version small --llm_ver 7b --hidden_size 64 --layer 0 --lora 0 --fast_eval 0 --embed_hidden_size 4096 --page_size 8
```

Training and testing with default parameters:
```bash
python worker.py  --data config/data/eb-nerd-once.yaml --embed config/embed/llama-token.yaml  --model config/model/llm/llama-{yourModel}-once.yaml --exp config/exp/tt-llm.yaml --embed_hidden_size 4096 --llm_ver 7b --layer 31 --version small --lr 0.0001 --item_lr 0.00001 --batch_size 32 --acc_batch 2 --epoch_batch -4  
```

## Extension
### DIRE with Sentiment
```bash
conda activate once
```

Tokenize the EB-NeRD data located in [`src/data/`](src/data/) with the tokenizer for the basic data and sentiment:
```bash
python src/scripts/scripts_dire/9_processor_sentiment.py
python src/scripts/scripts_dire/10_processor_llama_sentiment.py
python src/scripts/scripts_dire/11_ebnerd_fusion-sentiment.py
```

Prepare script: open [`src/lib/Legommenders/model/inputer/llm_concat_inputer.py`](src/lib/Legommenders/model/inputer/llm_concat_inputer.py) and uncomment line 38. If you are curious what is behind the list check [`/src/scripts/scripts_dire/0_get_col_prompts_additions.py`](/src/scripts/scripts_dire/0_get_col_prompts_additions.py).

Move to the Legommenders repository and pre-train. Replace {yourModel} with the recommender model you would like to use out of 'fastformer', 'naml', or 'nrms'. You will need to update the config files to point to [`src/data/`](src/data/):
```bash
cd src/lib/Legommenders/

python worker.py --embed config/embed/llama-token.yaml --model config/model/llm/llama-{yourModel}-sentiment.yaml --exp config/exp/llama-split-sentiment.yaml --data config/data/eb-nerd-sentiment.yaml --version small --llm_ver 7b --hidden_size 64 --layer 0 --lora 0 --fast_eval 0 --embed_hidden_size 4096 --page_size 8 --cuda -1
```

Training and testing with default parameters:
```bash
python worker.py  --data config/data/eb-nerd-sentiment.yaml --embed config/embed/llama-token.yaml  --model config/model/llm/llama-{yourModel}-sentiment.yaml --exp config/exp/tt-llm.yaml --embed_hidden_size 4096 --llm_ver 7b --layer 31 --version small --lr 0.0001 --item_lr 0.00001 --batch_size 32 --acc_batch 2 --epoch_batch -4 
```
