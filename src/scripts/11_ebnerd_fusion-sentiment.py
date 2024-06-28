from UniTok import UniDep


news = UniDep('/scratch-shared/scur1569/ebnerd_large_tokenized-sentiment/news')
news_llama = UniDep('/scratch-shared/scur1569/ebnerd_large_tokenized-sentiment/news-llama')


news.rename_col('title', 'title-bert')
news.rename_col('subtitle', 'subtitle-bert')
news.rename_col('body', 'body-bert')
news.rename_col('category', 'category-token')
news.rename_col('sentiment_label', 'sentiment_label-token')

news_llama.rename_col('title', 'title-llama')
news_llama.rename_col('subtitle', 'subtitle-llama')
news_llama.rename_col('body', 'body-llama')
news_llama.rename_col('category', 'category-llama')
news_llama.rename_col('sentiment_label', 'sentiment_label-llama')

news.inject(news_llama, ['title-llama', 'subtitle-llama', 'body-llama', 'category-llama', 'sentiment_label-llama'])

news.export('/scratch-shared/scur1569/ebnerd_large_tokenized-sentiment/news-fusion')
