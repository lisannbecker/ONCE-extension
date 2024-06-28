"""
Download and store llama embedding weights.
"""



import torch
import os
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM,LlamaModel

#------------------------------------
#  Solution 1 : Not using the pretrained llama-7b Folder
#--------------------------------------

# model_name = 'huggyllama/llama-7b' 
# # pretrained_dir = '/path/to/llama/'
# # device = 'cuda:1'
# # save_dir = 'ebnerd_data/'
# # save_path = os.path.join(save_dir, 'llama-token.npy')


# # Load the tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # Load pre-trained model (weights)
# model = AutoModelForCausalLM.from_pretrained(model_name)  #.to(device)  # type: LlamaModel
# # print(len(model.layers))

# # Put the model in "evaluation" mode, meaning feed-forward operation.
# model.eval()

# # Extract token embeddings
# embeddings = model.get_input_embeddings().weight.cpu().detach().numpy()

# # Save the embeddings to a numpy file
# np.save('llama-token.npy', embeddings)
# print('Embeddings saved')


#----------------(Old snippet)----------------------
# vocab = tokenizer.get_vocab()
# vocab_size = len(vocab)
# embedding_dim = 4096 #model.config.hidden_size

# embeddings = np.zeros((vocab_size, embedding_dim))

# for token, idx in vocab.items():
#     input_ids = tokenizer.encode(token, return_tensors='pt')
#     with torch.no_grad():
#         outputs = model(input_ids)
#         token_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy() #this doesnt work
#         embeddings[idx] = token_embedding

# np.save('data/llama-token.npy', embeddings)
# print('Embeddings saved to data/llama-token.npy')
#---------------------------------------------------------


#------------------------------------
# (Author's) Solution 2: Using pretrained dir (llama-7b)
#--------------------------------------

import numpy as np
from transformers import LlamaModel

pretrained_dir = '/scratch-shared/scur1569/llama-7b/' 
# device = 'cuda:1'

# Load pre-trained model (weights)
model = LlamaModel.from_pretrained(pretrained_dir) #.to(device)  # type: LlamaModel
print(len(model.layers))

# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()

embeds = model.embed_tokens.weight.cpu().detach().numpy()

#uncomment as needed
np.save('/scratch-shared/scur1569/llama-7b/llama-token.npy', embeds)
np.save('/scratch-shared/scur1569/llama-7b-sentiment/llama-token.npy', embeds)
np.save('/scratch-shared/scur1569/llama-7b-once/llama-token.npy', embeds)
