from UniTok import UniDep

#Update as needed
news = UniDep('/scratch-shared/scur1569/ebnerd_small_tokenized/news')
news_llama = UniDep('/scratch-shared/scur1569/ebnerd_small_tokenized/news-llama')

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
news.export('ebnerd_small_tokenized/news-fusion')
