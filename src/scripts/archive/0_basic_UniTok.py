"""
Example code for tokenization with UniTok
"""


from UniTok import UniTok, Column, Vocab
from UniTok.tok import IdTok, BertTok, EntTok
import pandas as pd

# dataset_name = 'ebnerd_small'
dataset_name = 'ebnerd_testset'
file_path = f'{dataset_name}/articles.parquet'

df = pd.read_parquet(file_path)


# news id vocab, commonly used in news data, history data, and interaction data
nid_vocab = Vocab('nid')

# bert tokenizer,  used in tokenizing title and abstract.
eng_tok = BertTok(vocab_dir='bert-base-uncased', name='eng')

news_ut = UniTok()

news_ut.add_col(Column(
    name='article_id',
    tok=IdTok(vocab=nid_vocab),
)).add_col(Column(
    name='title',
    tok=eng_tok,
    max_length=50  
)).add_col(Column(
    name='subtitle',
    tok=eng_tok,
    max_length=120,
)).add_col(Column(
    name='category_str', #TODO or category
    tok=EntTok(name='category'), 
)).add_col(Column(
    name='subcategory',
    tok=EntTok(name='subcategory'),  
))


news_ut.read(df)
print("check if getting data worked:")
print(news_ut.data)

news_ut.tokenize()

news_ut.store('data_tokenized/test_news')