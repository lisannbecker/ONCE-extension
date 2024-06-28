import pandas as pd
import itertools
import os

# Ensure the directory exists
output_dir = 'GENRE/data/eb-nerd/eb-nerd-data/articles_data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

dataset_name = 'ebnerd_small'
# dataset_name = 'ebnerd_testset'

pd.set_option('display.max_columns', None)

articles_path = f"{dataset_name}/articles.parquet"

articles = pd.read_parquet(articles_path)

# print("\nArticles:")
# print(articles.head())
print(articles.columns)
# Extracting specific columns from the DataFrame
# Assuming 'topics' is a string of topics separated by spaces
articles['first_topic'] = articles['topics'].astype(str).str.strip("[]").str.replace("'", "").str.split(', ').str[0]
selected_columns = articles[['article_id', 'category_str', 'first_topic', 'title', 'subtitle']]
selected_columns.columns = ['nid', 'cat', 'subcat', 'title', 'abs']
# print(selected_columns.head())

# Saving the selected columns to a TSV file
selected_columns.to_csv('GENRE/data/eb-nerd/eb-nerd-data/articles_data/news_ebnerd.tsv', sep='\t', index=False)