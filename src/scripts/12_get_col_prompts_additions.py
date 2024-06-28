from transformers import LlamaTokenizer

t = LlamaTokenizer.from_pretrained('llama_tokenizer')

print(f"topics={t.encode('<topics>')},")
print(f"region={t.encode('<region>')},")

#print(f"sentiment_label={t.encode('<sentiment_label>')},")