"""
Downloading llama 7b or 13b model and tokenizer
"""

from transformers import AutoTokenizer, AutoModelForCausalLM

#https://huggingface.co/huggyllama/llama-7b?library=transformers


#tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")

model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-7b")
#model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-13b")



#tokenizer.save_pretrained("./llama_tokenizer")

model.save_pretrained("/scratch-shared/scur1569/llama-7b-once")
#model.save_pretrained("/scratch-shared/scur1569/llama-13b")