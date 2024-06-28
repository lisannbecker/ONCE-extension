import pandas as pd
import os

def load_log_file(file_path):
    with open(file_path, 'r') as file:
        log_data = file.readlines()
    return log_data

def parse_log_line(line):
    if '\t[newtitle] ' in line:
        nid, title = line.split('\t[newtitle] ')
        return {'article_id': nid, 'title': title.strip()}
    else:
        print(line)

def process_files(log_file_path, articles_file_path, output_file_path):
    log_data = load_log_file(log_file_path)
    parsed_data = [parse_log_line(line) for line in log_data]
    parsed_data = [entry for entry in parsed_data if entry is not None]
    df = pd.DataFrame(parsed_data)
    # It may contains some duplicates
    df = df.drop_duplicates(subset='article_id')

    articles_df = pd.read_parquet(articles_file_path)

    #print(df)
    df['article_id'] = df['article_id'].astype('int32')
    #Merge the two DataFrames on the 'nid' column
    merged_df = articles_df.merge(df, on='article_id', how='left')

    # Replace the articles_df["title"] with df["title"]
    merged_df['title'] = merged_df['title_y'].combine_first(merged_df['title_x'])
    merged_df = merged_df.drop(columns=['title_x', 'title_y'])


    print(merged_df)

    merged_df.to_parquet(output_file_path)

# Define base path
base_path = os.path.expanduser('~')

# Paths for training files
train_log_file_path = f'{base_path}/GENRE/data/eb-nerd/eb-nerd-outputs/news_summarizer.log'
train_articles_file_path = f'{base_path}/GENRE/data/eb-nerd/eb-nerd-data/ebnerd_small/articles.parquet'
train_output_file_path = f'{base_path}/GENRE/data/eb-nerd/eb-nerd-data/eb-nerd_augmented_data/articles_aug.parquet'

# Process training files
process_files(train_log_file_path, train_articles_file_path, train_output_file_path)
