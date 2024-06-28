import json
import time


from tqdm import tqdm

from processor.mind.prompter import MindPrompter, MindColdUser
from utils.openai.chat_service import ChatService

MIN_INTERVAL = 0

# concise

mind_prompter = MindPrompter('/home/scur1569/ebnerd_data/genre_data/news_ebnerd.tsv')
user_list = MindColdUser('/home/scur1569/ONCE/data/eb-nerd/eb-nerd-data/ebnerd_small/train/history.parquet', mind_prompter).stringify()

system = """You are asked to capture user's interest based on his/her browsing history, and generate a piece of news that he/she may be interested. The format of history is as below:

(1) (the category of the first news) the title of the first news
...
(n) (the category of the n-th news) the title of the n-th news

You can only generate a piece of news (only one) in the following json format:

{"title": <title>, "abstract": <news abstract>, "category": <news category>}

where <news category> is limited to the following options:

- lifestyle
- health
- news
- sports
- weather
- entertainment
- autos
- travel
- foodanddrink
- tv
- finance
- movies
- video
- music
- kids
- middleeast
- northamerica
- games

"title", "abstract", and "category" should be the only keys in the json dict. The news should be diverse, that is not too similar with the original provided news list. You are not allowed to response any other words for any explanation or note. JUST GIVE ME JSON-FORMAT NEWS. Now, the task formally begins. Any other information should not disturb you."""

save_path = '/home/scur1569/ONCE/data/eb-nerd/eb-nerd-outputs/generator_v4_10.log'

exist_set = set()
with open(save_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        exist_set.add(data['uid'])

for uid, content in tqdm(user_list):
    start_time = time.time()
    if uid in exist_set:
        continue

    if not content:
        continue

    try:
        service = ChatService(system)
        enhanced = service.ask(content)  # type: str
        enhanced = enhanced.rstrip('\n')

        with open(save_path, 'a') as f:
            f.write(json.dumps({'uid': uid, 'news': enhanced}) + '\n')
    except Exception as e:
        print(e)
    time.sleep(0.5)
    interval = time.time() - start_time
    if interval <= MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - interval)
