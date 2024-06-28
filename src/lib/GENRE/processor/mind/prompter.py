import json
import os
import numpy as np

import pandas as pd
#from UniTok import UniDep
from tqdm import tqdm


class MindPrompter:
    def __init__(self, data_path):
        self.data_path = data_path

        self.news_df = pd.read_csv(
            filepath_or_buffer=os.path.join(data_path),
            sep='\t',
            header=0,
        )

        self.keys = dict(
            title='title',
            abstract='abs',
            category='cat',
            subcategory='subcat',
        )

        self._news_list = None
        self._news_dict = None

    def stringify(self):
        if self._news_list is not None:
            return self._news_list
        self._news_list = []
        for news in tqdm(self.news_df.iterrows()):
            string = ''
            for key in self.keys:
                string += f'[{key}] {news[1][self.keys[key]]}\n'
            self._news_list.append((news[1]['nid'], string))
        return self._news_list

    def get_news_dict(self):
        if self._news_dict is not None:
            return self._news_dict
        self._news_dict = {}
        for news in tqdm(self.news_df.iterrows()):
            self._news_dict[news[1]['nid']] = news[1]['title']
        return self._news_dict

    def get_news_dict_with_category(self):
        if self._news_dict is not None:
            return self._news_dict
        self._news_dict = {}
        for news in tqdm(self.news_df.iterrows()):
            self._news_dict[news[1]['nid']] = f'({news[1]["cat"]}) {news[1]["title"]}'
        return self._news_dict


class MindUser:
    def __init__(self, history_path, mind_prompter):
        self.news_dict = mind_prompter.get_news_dict()
        self.history_df = pd.read_parquet(history_path)
        self.user_article_dict = self.history_df[['user_id', 'article_id_fixed']].rename(columns={'article_id_fixed': 'history'}).set_index('user_id').to_dict()['history']

        self._user_list = None

    def stringify(self):
        if self._user_list is not None:
            return self._user_list
        self._user_list = []
        for user_id, history in tqdm(self.user_article_dict.items()):
            string = ''
            if not history.size:
                self._user_list.append((user_id, None))
                continue
            for i, n in enumerate(history[:30]):
                if n in self.news_dict:
                    string += f'({i + 1}) {self.news_dict[n]}\n'
            self._user_list.append((user_id, string))
        return self._user_list


class MindColdUser:
    def __init__(self, history_path, mind_prompter):
        self.news_dict = mind_prompter.get_news_dict()
        self.history_df = pd.read_parquet(history_path)
        self.user_article_dict = self.history_df[['user_id', 'article_id_fixed']].rename(columns={'article_id_fixed': 'history'}).set_index('user_id').to_dict()['history']

        self._user_list = None

    def stringify(self):
        if self._user_list is not None:
            return self._user_list
        self._user_list = []
        for user_id, history in tqdm(self.user_article_dict.items()):
            string = ''
            if not history.size or len(history) > 10:
                continue
            for i, n in enumerate(history[:30]):
                if n in self.news_dict:
                    string += f'({i + 1}) {self.news_dict[n]}\n'
            self._user_list.append((user_id, string))
        return self._user_list


class MindCoT_old:
    def __init__(self, data_path, plugin_path, mind_prompter, allowed_user_path):
        self.depot = UniDep(data_path, silent=True)
        self.plugin = UniDep(plugin_path, silent=True)
        self.tv = self.plugin.vocabs['topic']
        self.rv = self.plugin.vocabs['region']

        self.nid = self.depot.vocabs('nid')
        self.news_dict = mind_prompter.get_news_dict_with_category()

        self._user_list = None
        self.allowed_user = json.load(open(allowed_user_path))

    def stringify(self):
        if self._user_list is not None:
            return self._user_list
        self._user_list = []
        for user in tqdm(self.depot):
            if user['uid'] not in self.allowed_user:
                continue
            string = ''
            pg = self.plugin[user['uid']]
            string += 'Interest Topics:\n'
            for t in pg['topic']:
                string += f'- {self.tv[t]}\n'
            string += '\n'
            # string += 'Interest Regions:\n'
            # for r in pg['region']:
            #     string += f'- {self.rv[r]}\n'
            # string += '\n'

            string += 'History:\n'
            for i, n in enumerate(user['history']):
                string += f'({i + 1}) {self.news_dict[self.nid.i2o[n]]}\n'
            self._user_list.append((user['uid'], string))
        return self._user_list

class MindCoT:
    def __init__(self, history_path, mind_prompter, allowed_user_path):
        self.news_dict = mind_prompter.get_news_dict_with_category()
        self.history_df = pd.read_parquet(history_path)
        self.user_article_dict = self.history_df[['user_id', 'article_id_fixed']].rename(columns={'article_id_fixed': 'history'}).set_index('user_id').to_dict()['history']
        
        self.allowed_user = json.load(open(allowed_user_path))
        self._user_list = None

    def stringify(self):
        if self._user_list is not None:
            return self._user_list
        self._user_list = []
        for user_id, history in tqdm(self.user_article_dict.items()):
            if user_id not in self.allowed_user:
                continue
            string = 'Interest Topics:\n'
            # self.plugin` and `self.tv` are not used anymore, removing related code
            string += '\nHistory:\n'
            for i, n in enumerate(history):
                if n in self.news_dict:
                    string += f'({i + 1}) {self.news_dict[n]}\n'
            self._user_list.append((user_id, string))
        return self._user_list