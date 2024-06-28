"""
Inspect the eb-nerd dataset.
"""

import pandas as pd
import itertools


dataset_name = 'ebnerd_small'
# dataset_name = 'ebnerd_testset'
#dataset_name = 'ebnerd_augmented'

pd.set_option('display.max_columns', None)

# Define the file paths
#train_behaviours_path = f"{dataset_name}/train/behaviors.parquet"
#train_history_path = f"{dataset_name}/train/history.parquet"
train_history_path = f"ebnerd_augmented/train/history_aug.parquet"
# val_behaviours_path = f"{dataset_name}/validation/behaviors.parquet"
# val_history_path = f"{dataset_name}/validation/history.parquet"
#articles_path = f"{dataset_name}/articles.parquet"
#dataset_name = 'ebnerd_augmented'
#articles_aug_path = f"ebnerd_augmented/articles_aug.parquet"

# Read the Parquet files into pandas DataFrames
#train_behaviours = pd.read_parquet(train_behaviours_path)

train_history = pd.read_parquet(train_history_path)
# val_behaviours = pd.read_parquet(val_behaviours_path)
# val_history = pd.read_parquet(val_history_path)
#articles = pd.read_parquet(articles_path)
#articles_aug = pd.read_parquet(articles_aug_path)
#article_type


"""
subcategory_lists = articles['subcategory'].tolist()
flattened_subcategories = list(itertools.chain.from_iterable(subcategory_lists))
unique_subcategories = subcategory_lists #set(flattened_subcategories)
print(sorted(set(flattened_subcategories)))

# Inspect the DataFrames
""" 

#print("Train Behaviours:")
#print(train_behaviours.head())
#print(train_behaviours.columns)
#print("\nTrain History:")
#print(train_history.head(10))
#print(set(train_history['topics']))
#print(set(train_history['region']))
#print(train_history.columns)

"""
print("\nVal Behaviours:")
print(val_behaviours.head())
print("\nVal History:")
print(val_history.head())
"""
# print("\nTest Behaviours:")
# print(train_behaviours.head())
# print("\nTest History:")
# print(train_history.head())
print(f"Average number of topics: {train_history['topics'].apply(lambda x: x.split(', ')).apply(len).mean()}")
print(f"Upper quantile of number of topics: {train_history['topics'].apply(lambda x: x.split(', ')).apply(len).quantile([0.25, 0.5, 0.75])[0.75]}")


#print("\nArticles:")
#print(articles.head(3))
#print(articles.columns)
#print(f"Average title length: {articles['title'].apply(len).mean()}")

#print("\nArticles AUG:")
#print(articles_aug.head(3))
#print(articles_aug.columns)
#print(f"Average title length AUG: {articles_aug['title'].apply(len).mean()}")


#print("\Predictions:")
#print(predictions.head())
#print(predictions.columns)

# Summary statistics
# print(train_behaviours.describe())
# print(train_history.describe())
# print(val_behaviours.describe())
# print(val_history.describe())

# print(test_behaviours.describe())
# print(test_history.describe())
#print(predictions.describe())

# Column names and data types
# print(train_behaviours.info())
# print(train_history.info())
# print(val_behaviours.info())
# print(val_history.info())
# print(test_history.info())
# print(test_behaviours.info())
#print(predictions.info())

