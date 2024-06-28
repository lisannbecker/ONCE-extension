import pandas as pd
import numpy as np
import os

# Define base path
base_path = os.path.expanduser('~')

# Load the data
history_data = pd.read_parquet(f'{base_path}/GENRE/data/eb-nerd/eb-nerd-data/ebnerd_small/train/history.parquet')
history_data_val = pd.read_parquet(f'{base_path}/GENRE/data/eb-nerd/eb-nerd-data/ebnerd_small/validation/history.parquet')
history_data = pd.concat([history_data, history_data_val], ignore_index=True)
articles_df = pd.read_parquet(f'{base_path}/GENRE/data/eb-nerd/eb-nerd-data/ebnerd_small/articles.parquet')

# Load in augmented data
history_data_aug = pd.read_parquet(f'{base_path}/GENRE/data/eb-nerd/eb-nerd-data/eb-nerd_augmented_data/train/history_aug.parquet')
history_data_aug_val = pd.read_parquet(f'{base_path}/GENRE/data/eb-nerd/eb-nerd-data/eb-nerd_augmented_data/validation/history_aug.parquet')
history_data_aug = pd.concat([history_data_aug, history_data_aug_val], ignore_index=True)
articles_aug_df = pd.read_parquet(f'{base_path}/GENRE/data/eb-nerd/eb-nerd-data/eb-nerd_augmented_data/articles_aug.parquet')

# Calculate statistics
num_content = articles_df.shape[0]
tokens_per_title = articles_df['title'].apply(lambda x: len(x.split())).mean()
chars_per_title = articles_df['title'].apply(len).mean()
num_users = history_data['user_id'].nunique()
user_history_size = history_data.groupby('user_id')['article_id_fixed'].apply(lambda x: len(x.iloc[0]))

# Calculate number of new users (users with no more than five contents in browsing history)
new_users = user_history_size[(user_history_size <= 5) & (user_history_size > 0)]
num_new_users = len(new_users)

content_per_user = user_history_size.mean()
# Calculate content per new user
content_per_user_new = new_users.mean()

# Define a threshold for positive interaction (e.g., 10 seconds)
POSITIVE_THRESHOLD = 10

# Count positive and negative interactions
history_data['is_positive'] = history_data['read_time_fixed'].apply(lambda x: np.any(np.array(x) >= POSITIVE_THRESHOLD))
num_pos = history_data['is_positive'].sum()
num_neg = (~history_data['is_positive']).sum()


# Calculate statistics for augmented data
tokens_per_title_aug = articles_aug_df['title'].apply(lambda x: len(x.split())).mean()
chars_per_title_aug = articles_aug_df['title'].apply(len).mean()


# Calculate topics and regions per user
def count_unique_items(x):
    return len(set([item for sublist in x for item in sublist.split(', ')]))

topics_per_user = history_data_aug.groupby('user_id')['topics'].apply(count_unique_items).mean()

regions_per_user = history_data_aug[history_data_aug['region'] != 'Unknown'].groupby('user_id')['region'].nunique().mean()



# Print results
print(f"# content: {num_content}")
print(f"tokens/title: {tokens_per_title:.2f}")
print(f"chars/title: {chars_per_title:.2f}")
print(f"tokens/title (augmented): {tokens_per_title_aug:.2f}")
print(f"chars/title (augmented): {chars_per_title_aug:.2f}")
print(f"# users: {num_users}")
print(f"# new users: {num_new_users}")
print(f"content/user: {content_per_user:.2f}")
print(f"content/user_new: {content_per_user_new:.2f}")
print(f"# pos: {num_pos}")
print(f"# neg: {num_neg}")
print(f"topics/user (augmented): {topics_per_user:.2f}")
print(f"regions/user (augmented): {regions_per_user:.2f}")
