# ONCE-extension

--- 

### A. Ibrahimi, L. Becker, R. van Emmerik

---

## Installation
### Requirements
- Linux or Windows with Python = 3.11

### Environment Setup
```bash
conda create --name once python=3.11 
conda activate once

pip install -r requirements_once.txt
```

Run the following script to download the [LLaMA 7b model](https://huggingface.co/huggyllama/llama-7b?library=transformers) (to [`src/lib/XXX/`](src/lib/XXX/)):
```bash
bash XXX/download_coco.sh
```

### GENRE Implementation


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


### Extension/ DIRE with Sentiment
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
