"""
Downloading llama 7b or 13b model and tokenizer
"""
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

#https://huggingface.co/huggyllama/llama-7b?library=transformers


#tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
#tokenizer.save_pretrained("./llama_tokenizer")

model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-7b")
#model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-13b")
#model.save_pretrained("/scratch-shared/scur1569/llama-13b")

current_script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(current_script_dir, '../../data'))

save_dirs = ['llama-7b', 'llama-7b-once', 'llama-7b-sentiment']

for save_dir in save_dirs:
    output_dir = os.path.join(data_dir, save_dir)
    model.save_pretrained(output_dir)
    print(f'Model saved to {output_dir}')