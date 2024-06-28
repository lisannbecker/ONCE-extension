"""
Downloading llama 7b or 13b model and tokenizer

https://huggingface.co/huggyllama/llama-7b?library=transformers
"""
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

current_script_dir = os.path.dirname(os.path.abspath(__file__))

vocab_dir = os.path.join(current_script_dir, 'llama_tokenizer')

tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
tokenizer.save_pretrained(vocab_dir)

model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-7b")
#model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-13b")
#model.save_pretrained("/scratch-shared/scur1569/llama-13b")

data_dir = os.path.abspath(os.path.join(current_script_dir, '../../data'))

save_dirs = ['llama-7b', 'llama-7b-once', 'llama-7b-sentiment'] #Modify based on what you want to implement

for save_dir in save_dirs:
    output_dir = os.path.join(data_dir, save_dir)
    model.save_pretrained(output_dir)
    print(f'Model saved to {output_dir}')