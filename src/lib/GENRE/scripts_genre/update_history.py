import json
import pandas as pd
import os

def load_log_file(file_path):
    with open(file_path, 'r') as file:
        log_data = [json.loads(line) for line in file]
    return log_data
    
def parse_interest(interest_str):
    topics = []
    region = None
    lines = interest_str.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('- '):
            if region is None:
                topics.append(line[2:])
        elif line.startswith('[region]'):
            if i + 1 < len(lines) and lines[i + 1].startswith('- '):
                region = lines[i + 1][2:]
    return ', '.join(topics), region

def process_files(log_file_path, behavior_file_path, output_file_path):
    log_data = load_log_file(log_file_path)

    parsed_data = []
    for entry in log_data:
        topics, region = parse_interest(entry['interest'])
        parsed_data.append({'user_id': entry['uid'], 'topics': topics, 'region': region})

    df = pd.DataFrame(parsed_data)
    # It may contains some duplicates
    df = df.drop_duplicates(subset='user_id')
    behaviors_df = pd.read_parquet(behavior_file_path)

    print(df)
    print(behaviors_df)

    # Merge the two DataFrames on the 'uid' column
    merged_df = behaviors_df.merge(df, on='user_id', how='left')
    # sometimes chatgpt could not predict the region
    merged_df['region'] = merged_df['region'].replace(
        ['N/A', 'None', 'Not applicable', '(not available)', '(none)', '(unknown)', 'blank', 'n/a', '(blank)', 'No region mentioned', 
        'not available', '(not applicable)', '(unable to determine)', '(unable to predict)', '(no specific region)', 'unspecified',
        '(not enough data to determine)', 'Not determined', 'No specific region mentioned', 'unknown', '(Not available)', 'not specific',
        '(not enough information to determine)', '(Not applicable)', 'N/A  ', '(could not predict)', '(no specific region)', '(None)',
        '(No region information can be inferred)', '(uncertain)', 'not available ', 'blank ', '(None provided)', '_', 
        '(not enough information for specific regions)', '[No specific region identified]', 'no specific region is indicated', 
        '(difficult to predict based on the provided news list)', '<blank>', '(Unable to determine)', '[unpredictable]', '[empty]',
        ' ', '[no specific region]', '(Not enough information to determine specific region)', 'Not applicable.', 
        '*unable to predict region based on the provided news list*', '(n/a)', '(unable to predict region)', 'regions are hard to predict',
        '[No specific region information available]', 'there is no specific region mentioned in the news list', 
        'Not able to determine with provided information', '(none) ', 'not specific to any region', '(not available) ',
        'specific region not determined', '(no specific region mentioned)', '(not enough information to determine specific region)',
        'No specific region is indicated.', '*Unable to predict specific region*', '(difficult to predict)', 'not applicable',
        '(unpredictable)', 'Not predictable', '(No specific region)', 'N/A ', '(empty)', '<<unable to determine>>',
        'Blank', 'not provided', 'not specified', 'Not determinable', '(No specific region can be determined)',
        '(Predicting region is not applicable based on the provided news list)', '(hard to predict) ', '(blank) ', 'unavailable',
        'non-specific', '(not enough information to determine regions)', 'unspecified or hard to predict', '(Unavailable)', '[not available]',
        '(not enough information)', 'regions not applicable', 'No specific region is indicated', 'No specific region. ',
        'region: N/A', 'Does not apply', 'Blank (region is hard to determine)', 'unidentifiable', 
        'No specific region information can be identified.', '*Unable to determine*', '(not enough information to determine a specific region)',
        '(not enough information provided)', '(cannot be determined)', '(unable to predict based on provided news list)', '(difficult to determine)',
        '(Blank)', 'unidentifiable in this context', '(hard to predict)', 'No specific region', 'none', 'null', 'no specific region', None,
        'Not available', 'not determinable', '(N/A)', '(optional)', '', '(not provided)', '[Not available]', 'No specific region mentioned.',
        'no specific region mentioned', '(hard to determine)', 'specific regions are hard to determine from the provided news list',
        '(No specific region stands out)', '(missing)', ],
        'Unknown')
    
    print(merged_df)
    merged_df.to_parquet(output_file_path)

# Define base path
base_path = os.path.expanduser('~')

# Paths for training files
train_log_file_path = f'{base_path}/GENRE/data/eb-nerd/eb-nerd-outputs/user_profiler.log'
train_behavior_file_path = f'{base_path}/GENRE/data/eb-nerd/eb-nerd-data/ebnerd_small/train/history.parquet'
train_output_file_path = f'{base_path}/GENRE/data/eb-nerd/eb-nerd-data/eb-nerd_augmented_data/train/history_aug.parquet'

# Paths for validation files
val_log_file_path = f'{base_path}/GENRE/data/eb-nerd/eb-nerd-outputs/user_profiler_val.log'
val_behavior_file_path = f'{base_path}/GENRE/data/eb-nerd/eb-nerd-data/ebnerd_small/validation/history.parquet'
val_output_file_path = f'{base_path}/GENRE/data/eb-nerd/eb-nerd-data/eb-nerd_augmented_data/validation/history_aug.parquet'

# Process training files
process_files(train_log_file_path, train_behavior_file_path, train_output_file_path)

# Process validation files
process_files(val_log_file_path, val_behavior_file_path, val_output_file_path)