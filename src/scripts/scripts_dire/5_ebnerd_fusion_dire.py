from UniTok import UniDep
import os

#Update as needed
current_script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(current_script_dir, '../../data/ebnerd_small_tokenized'))
news_path = os.path.join(data_dir, 'news')
news_llama_path = os.path.join(data_dir, 'news-llama')

news = UniDep(news_path)
news_llama = UniDep(news_llama_path)
#news = UniDep('/scratch-shared/scur1569/ebnerd_small_tokenized/news')
#news_llama = UniDep('/scratch-shared/scur1569/ebnerd_small_tokenized/news-llama')

news.rename_col('title', 'title-bert')
news.rename_col('subtitle', 'subtitle-bert')
news.rename_col('body', 'body-bert')
news.rename_col('category', 'category-token')

news_llama.rename_col('title', 'title-llama')
news_llama.rename_col('subtitle', 'subtitle-llama')
news_llama.rename_col('body', 'body-llama')
news_llama.rename_col('category', 'category-llama')

news.inject(news_llama, ['title-llama', 'subtitle-llama', 'body-llama', 'category-llama'])

#Update as needed
news_fusion_path = os.path.join(data_dir, 'news-fusion')
news.export(news_fusion_path)
#news.export('ebnerd_small_tokenized/news-fusion')
