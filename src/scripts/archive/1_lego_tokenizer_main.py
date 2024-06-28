"""
Modified MIND tokenizer to run on eb-nerd dataset. Not anymore used for our project.
"""

import json
import os
import random
import polars as pl

import numpy as np
import pandas as pd
from UniTok import Vocab, UniTok, Column
from UniTok.tok import IdTok, SplitTok, BertTok, EntTok, BaseTok, NumberTok
from nltk import word_tokenize

from ebrec.utils._descriptive_analysis import (
    min_max_impression_time_behaviors, 
    min_max_impression_time_history
)
from ebrec.utils._polars import slice_join_dataframes
from ebrec.utils._behaviors import (
    create_binary_labels_column,
    sampling_strategy_wu2019,
    truncate_history,
)
from ebrec.utils._constants import (
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_USER_COL
)

class GloveTok(BaseTok):
    def __init__(self, name: str, path: str):
        super().__init__(name)
        self.vocab = Vocab('english').load(path, as_path=True)

    def t(self, obj: str):
        ids = []
        objs = word_tokenize(str(obj).lower())
        for o in objs:
            if o in self.vocab.obj2index:
                ids.append(self.vocab.obj2index[o])
        return ids or [self.vocab.obj2index[',']]


class Processor:
    def __init__(self, data_dir, store_dir, glove=None, imp_list_path: str = None):
        self.data_dir = data_dir
        self.store_dir = store_dir
        self.glove = glove
        self.imp_list = json.load(open(imp_list_path, 'r')) if imp_list_path else None

        # if os.path.exists(self.store_dir):
        #     c = input(f'{self.store_dir} exists, press Y to continue, or press any other to exit.')
        #     if c.upper() != 'Y':
        #         exit(0)

        os.makedirs(self.store_dir, exist_ok=True)

        self.train_store_dir = os.path.join(self.store_dir, 'train')
        self.dev_store_dir = os.path.join(self.store_dir, 'dev')

        self.nid = Vocab(name='nid')
        self.uid = Vocab(name='uid')
        #self.subcat = Vocab(name='subcat')

    def read_news_data(self):
        filepath = os.path.join(self.data_dir, 'articles.parquet') #took mode out
        news_data= pd.read_parquet(
        filepath,
        columns=['article_id', 'category_str', 'subcategory', 'title', 'subtitle']
        )
        news_data.rename(columns={
            'article_id': 'nid',
            'category_str': 'cat',
            'subcategory': 'subcat',
            'title': 'title',
            'subtitle': 'abs'
        }, inplace=True)
        
        return news_data
        """
        return pd.read_parquet(
            filepath_or_buffer=os.path.join(self.data_dir, mode, 'news.tsv'),
            sep='\t',
            names=['nid', 'cat', 'subcat', 'title', 'abs', 'url', 'tit_ent', 'abs_ent'],
            usecols=['nid', 'cat', 'subcat', 'title', 'abs'],
        )
        """ 

    def read_user_data(self, mode):

        filepath_history = os.path.join(self.data_dir, mode, 'history.parquet') #took mode out
        df_history = pl.scan_parquet(filepath_history)

        df_history = df_history.select(DEFAULT_USER_COL, DEFAULT_HISTORY_ARTICLE_ID_COL).pipe(
            truncate_history,
            column=DEFAULT_HISTORY_ARTICLE_ID_COL,
            history_size=30,
            padding_value=0,
            enable_warning=False,)
        
        df_history = df_history.collect()
        df_history = df_history.to_pandas()

        df_history.rename(columns={
            'user_id': 'uid',
            'article_id_fixed': 'history'
        }, inplace=True)

        return df_history


        return pd.read_parquet(
            filepath_or_buffer=os.path.join(self.data_dir, mode, 'behaviors.tsv'),
            sep='\t',
            names=['imp', 'uid', 'time', 'history', 'predict'],
            usecols=['uid', 'history']
        )

    def _read_inter_data(self, mode): 

        #Option 1 (Non Working atm)
        # filepath = os.path.join(self.data_dir, mode, 'behaviors.parquet')
        # inter_data = pd.read_parquet(filepath)

        # inter_data = pl.from_pandas(inter_data) #pandas -> polar

        # if 'article_ids_clicked' in inter_data.columns:
        #     clicked_articles_col = DEFAULT_CLICKED_ARTICLES_COL
        #     inter_data = inter_data.select(["user_id", "impression_id", "article_id", DEFAULT_CLICKED_ARTICLES_COL, DEFAULT_INVIEW_ARTICLES_COL]).pipe(
        #         create_binary_labels_column, shuffle=True, seed=123
        #     ).with_columns(pl.col("labels").list.len().name.suffix("_len"))
            
        #     inter_data = inter_data.with_columns(
        #         pl.struct(["article_ids_inview", "labels"]).map_elements(lambda row: [f"{id}-{label}" for id, label in zip(row["article_ids_inview"], row["labels"])]).alias("predict")
        #     )
        # else:
        #     inter_data = inter_data.select(["user_id", "impression_id", "article_id", DEFAULT_INVIEW_ARTICLES_COL])
        #     inter_data = inter_data.with_columns(pl.lit(None).alias('predict'))
        #     print("Warning: 'article_ids_clicked' column is missing, 'predict' column will be filled with None.")

        
        # inter_data = inter_data.to_pandas() #polar -> pandas

        # inter_data['predict'] = inter_data['predict'].astype(str)

        # # Renaming columns to match the MIND dataset format
        # inter_data.rename(columns={
        #     'impression_id': 'imp',
        #     'user_id': 'uid',
        # }, inplace=True)

        # # Selecting only the required columns
        # inter_data = inter_data[['imp', 'uid', 'predict']]

        # return inter_data	
        
        # Option 2
        filepath = os.path.join(self.data_dir, mode, 'behaviors.parquet')
        inter_data = pd.read_parquet(filepath)

        # Creating the 'predict' column
        def create_predict(row):
            inview_articles = row['article_ids_inview']
            clicked_articles = row.get('article_ids_clicked', [])
            predict = []

            for article_id in inview_articles:
                click = 1 if article_id in clicked_articles else 0
                predict.append(f"{article_id}-{click}")

            return ' '.join(predict)

        inter_data['predict'] = inter_data.apply(create_predict, axis=1)

        # Renaming columns to match the MIND dataset format
        inter_data.rename(columns={
            'impression_id': 'imp',
            'user_id': 'uid',
        }, inplace=True)

        # Selecting only the required columns
        inter_data = inter_data[['imp', 'uid', 'predict']]

        return inter_data	

        # Previous non-compatible columns Code:
        return pd.read_parquet(
            filepath_or_buffer= os.path.join(self.data_dir, mode, 'behaviors.parquet'),
            sep='\t',
            names=['imp', 'uid', 'time', 'history', 'predict'], #history = article_id_fixed; 
                                                                #predict = impressions: "List of news displayed in this impression and user's click behaviors on them (1 for click and 0 for non-click). The orders of news in a impressions have been shuffled."
                                                                #e.g., N129416-0 N26703-1 N120089-1 N53018-0 N89764-0 N91737-0 N29160-0
            usecols=['imp', 'uid', 'predict']
            #columns=['imp', 'uid', 'predict']
        )   

    def read_neg_data(self, mode):
        df = self._read_inter_data(mode)
        data = dict(uid=[], neg=[])
        for line in df.itertuples():
            if line.uid in data['uid']:
                continue
            
            print(f"Processing user: {line.uid}")
            predicts = line.predict.split(' ')
            negs = []
            for predict in predicts:
                nid, click = predict.split('-')
                # click = click.strip()
                # nid = nid.strip().strip('[]\'"') 
                # print(f"Article ID: {nid}, Click: {click}")
                if not int(click):
                    negs.append(nid)

            data['uid'].append(line.uid)
            data['neg'].append(' '.join(negs))
        return pd.DataFrame(data)

    def read_inter_data(self, df) -> pd.DataFrame:
    #def read_inter_data(self, mode) -> pd.DataFrame:

        #df = self._read_inter_data(mode)
        data = dict(imp=[], uid=[], nid=[], click=[])
        for line in df.itertuples():
            predict = line.predict

            if isinstance(predict, np.ndarray):
                predict = predict.tolist()
                predict = ' '.join(map(str, predict))
            predicts = predict.split(' ')
            data['imp'].extend([line.imp] * len(predicts))
            data['uid'].extend([line.uid] * len(predicts))
            for predict in predicts:
                nid, click = predict.split('-')
                data['nid'].append(nid)
                data['click'].append(int(click))
        return pd.DataFrame(data)

    def get_news_tok(self, max_title_len=0, max_abs_len=0):
        if self.glove:
            txt_tok = GloveTok(name='english', path=self.glove)
        else:
            txt_tok = BertTok(name='english', vocab_dir='bert-base-uncased')

        return UniTok().add_col(Column(
            tok=IdTok(vocab=self.nid)
        )).add_col(Column(
            name='cat',
            tok=EntTok,
        )).add_col(Column(
            name='subcat',
            tok=EntTok, #TODO update this - SplitTok(sep=' ', vocab=self.subcat) ?
        )).add_col(Column(
            name='title',
            tok=txt_tok,
            max_length=max_title_len,
        )).add_col(Column(
            name='abs',
            tok=txt_tok,
            max_length=max_abs_len,
        ))

    def get_user_tok(self, max_history: int = 0):
        user_ut = UniTok()
        user_ut.add_col(Column(
            tok=IdTok(vocab=self.uid)
        )).add_col(Column(
            name='history',
            tok=SplitTok(
                sep=' ',
                vocab=self.nid
            ),
            max_length=max_history,
            slice_post=True,
        ))
        return user_ut

    def get_neg_tok(self, max_neg: int = 0):
        neg_ut = UniTok()
        neg_ut.add_col(Column(
            tok=IdTok(vocab=self.uid),
        )).add_col(Column(
            name='neg',
            tok=SplitTok(
                sep=' ',
                vocab=self.nid
            ),
            max_length=max_neg,
            slice_post=True,
        ))
        return neg_ut

    def get_inter_tok(self):
        return UniTok().add_index_col(
            name='index'
        ).add_col(Column(
            name='imp',
            tok=EntTok,
        )).add_col(Column(
            tok=EntTok(vocab=self.uid)
        )).add_col(Column(
            tok=EntTok(vocab=self.nid)
        )).add_col(Column(
            tok=NumberTok(name='click', vocab_size=2)
        ))

    def combine_news_data(self):
        news_df = self.read_news_data()
        """
        news_train_df = self.read_news_data('train')
        news_dev_df = self.read_news_data('dev')
        news_df = pd.concat([news_train_df, news_dev_df])
        news_df = news_df.drop_duplicates(['nid'])
        """
        return news_df

    def combine_user_df(self):
        user_train_df = self.read_user_data('train')
        
        user_dev_df = self.read_user_data('validation')
        #user_dev_df = self.read_user_data('dev') 


        user_df = pd.concat([user_train_df, user_dev_df])
        user_df = user_df.drop_duplicates(['uid'])
        return user_df

    def combine_neg_df(self):
        neg_train_df = self.read_neg_data('train')
        neg_dev_df = self.read_neg_data('validation')

        neg_df = pd.concat([neg_train_df, neg_dev_df])
        neg_df = neg_df.drop_duplicates(['uid'])
        return neg_df

    def combine_inter_df(self):
        inter_train_df = self.read_inter_data('train')
        inter_dev_df = self.read_inter_data('dev')
        inter_dev_df.imp += max(inter_train_df.imp)

        inter_df = pd.concat([inter_train_df, inter_dev_df])
        return inter_df

    def splitter(self, l: list, portions: list):
        if self.imp_list:
            l = self.imp_list
        else:
            random.shuffle(l)
        json.dump(l, open(os.path.join(self.store_dir, 'imp_list.json'), 'w'))

        portions = np.array(portions)
        portions = portions * 1.0 / portions.sum() * len(l)
        portions = list(map(int, portions))
        portions[-1] = len(l) - sum(portions[:-1])

        pos = 0
        parts = []
        for i in portions:
            parts.append(l[pos: pos+i])
            pos += i
        return parts

    def reassign_inter_df_v2(self):
        inter_train_df = self.read_inter_data('train')
        inter_df = self.read_inter_data('validation')

        imp_list = inter_df.imp.drop_duplicates().to_list()

        dev_imps, test_imps = self.splitter(imp_list, [5, 5])
        inter_dev_df, inter_test_df = [], []

        inter_groups = inter_df.groupby('imp')
        for imp, imp_df in inter_groups:
            if imp in dev_imps:
                inter_dev_df.append(imp_df)
            else:
                inter_test_df.append(imp_df)
        return inter_train_df, \
               pd.concat(inter_dev_df, ignore_index=True), \
               pd.concat(inter_test_df, ignore_index=True)

    def analyse_news(self):
        tok = self.get_news_tok(
            max_title_len=0,
            max_abs_len=0
        )
        df = self.combine_news_data()
        tok.read(df).analyse()

    def analyse_user(self):
        tok = self.get_user_tok(max_history=0)
        df = self.combine_user_df()
        tok.read(df).analyse()

    def analyse_inter(self):
        tok = self.get_inter_tok()
        df = self.combine_inter_df()
        tok.read_file(df).analyse()

    def tokenize(self):
        news_tok = self.get_news_tok(
            max_title_len=20,
            max_abs_len=50
        )
        news_df = self.combine_news_data()
        news_tok.read_file(news_df).tokenize().store_data(os.path.join(self.store_dir, 'news'))

        user_tok = self.get_user_tok(max_history=30)
        user_df = self.combine_user_df()
        user_tok.read(user_df).tokenize().store(os.path.join(self.store_dir, 'user'))

        inter_dfs = self.reassign_inter_df_v2()
        for inter_df, mode in zip(inter_dfs, ['train', 'dev', 'test']):
            inter_tok = self.get_inter_tok()
            inter_tok.read_file(inter_df).tokenize().store_data(os.path.join(self.store_dir, mode))

    def transform_ebnerd_data(self, mode):
        filepath_history = os.path.join(self.data_dir, mode, 'history.parquet') 
        df_history = pl.scan_parquet(filepath_history)
        filepath_behavior = os.path.join(self.data_dir, mode, 'behaviors.parquet') 
        df_behaviors = pl.scan_parquet(filepath_behavior)

        #print(df_history.collect().to_pandas())
        #print(df_history.columns)
        #print(df_behaviors.collect().to_pandas())
        #print(df_behaviors.columns)

        df_history = df_history.select(DEFAULT_USER_COL, DEFAULT_HISTORY_ARTICLE_ID_COL).pipe(
            truncate_history,
            column=DEFAULT_HISTORY_ARTICLE_ID_COL,
            history_size=30,
            padding_value=0,
            enable_warning=False,)
        
        #print(df_history.collect().to_pandas())
        #print(df_history.collect().to_pandas())

        #print(df_history.columns)
        
        df = slice_join_dataframes(
            df1=df_behaviors.collect(),
            df2=df_history.collect(),
            on=DEFAULT_USER_COL,
            how="left",)
        #print(df)
        #print(df.columns)
            # Check if 'article_ids_clicked' exists

        if 'article_ids_clicked' in df.columns:
            #print('article ids clicked')
            #print(df['article_ids_clicked'])
            clicked_articles_col = DEFAULT_CLICKED_ARTICLES_COL
            df = df.select(["user_id", "impression_id", "article_id_fixed", clicked_articles_col, DEFAULT_INVIEW_ARTICLES_COL]).pipe(
                create_binary_labels_column, shuffle=True, seed=123
            ).with_columns(pl.col("labels").list.len().name.suffix("_len"))
            
            df_for_once = df.with_columns(
                pl.struct(["article_ids_inview", "labels"]).map_elements(lambda row: [f"{id}-{label}" for id, label in zip(row["article_ids_inview"], row["labels"])]).alias("predict")
            )
        else:
            # Handle missing column
            df_for_once = df.select(["user_id", "impression_id", "article_id_fixed", DEFAULT_INVIEW_ARTICLES_COL])
            df_for_once = df_for_once.with_columns(pl.lit(None).alias('predict'))
            print("Warning: 'article_ids_clicked' column is missing, 'predict' column will be filled with None.")
        
        df_for_once = df_for_once.to_pandas()

        df_for_once.rename(columns={
            'user_id': 'uid',
            'impression_id': 'imp',
            'article_id_fixed': 'history',
            'predict': 'predict'
        }, inplace=True)
        #print(df_for_once)

        return df_for_once


    def tokenize_original_dev(self):
        
         
        """
        Tokenize Articles > generates files in folder 'news'
        """
        
        news_tok = self.get_news_tok(
            max_title_len=20,
            max_abs_len=50
        )
        
        #news_df = self.combine_news_data() # > took out as we only have one file with new data
        news_df = self.read_news_data()
        
        news_tok.read_file(news_df).tokenize().store_data(os.path.join(self.store_dir, 'news')) #TODO fix subcat
        

        """
        Tokenize User data > generates files in folder 'user'
        """
        user_tok = self.get_user_tok(max_history=30)
        
        user_df = self.combine_user_df()
        print(user_df)
        #user_df.to_csv(os.path.join('/home/scur1569', 'user_df.csv'), index=False)
        
        user_tok.read(user_df).tokenize().store_data(os.path.join(self.store_dir, 'user'))
        
        
        
        """
        Tokenize train/validation  
        """
        #L: addition
        dataset = 'train' #'train/validation'
        df_for_inter = self.transform_ebnerd_data(f'{dataset}')

        #inter_df = self.read_inter_data('dev')
        inter_df = self.read_inter_data(df_for_inter)

        inter_tok = self.get_inter_tok()
        inter_tok.read_file(inter_df).tokenize().store_data(os.path.join(self.store_dir, f'{dataset}'))


        dataset = 'validation' #'train/validation'
        df_for_inter = self.transform_ebnerd_data(f'{dataset}')

        #inter_df = self.read_inter_data('dev')
        inter_df = self.read_inter_data(df_for_inter)

        inter_tok = self.get_inter_tok()
        inter_tok.read_file(inter_df).tokenize().store_data(os.path.join(self.store_dir, f'{dataset}'))


    def tokenize_neg(self):
        print('tokenize neg')
        self.uid.load(os.path.join(self.store_dir, 'user'))
        self.nid.load(os.path.join(self.store_dir, 'news'))

        print('combine neg df')
        neg_df = self.combine_neg_df()
        print('get neg tok')
        neg_tok = self.get_neg_tok()
        neg_tok.read(neg_df).tokenize().store(os.path.join(self.store_dir, 'neg'))


if __name__ == '__main__':
    # p = Processor(
    #     data_dir='/data1/qijiong/Data/MIND/',
    #     store_dir='../../data/MIND-small-v3',
    # )
    #

    p = Processor(
        data_dir='ebnerd_small',
        store_dir='ebnerd_small_tokenized',
    )

    p.tokenize_original_dev()
    # p.tokenize()
    # p.tokenize_neg()  #Uncommment to generate only the neg folder