from UniTok import UniDep


news = UniDep('ebnerd_small_tokenized_2/news')
news_llama = UniDep('ebnerd_small_tokenized_2/news-llama')

news.rename_col('title', 'title-bert')
news.rename_col('subtitle', 'subtitle-bert')
news.rename_col('body', 'body-bert')
news.rename_col('category', 'category-token')

news_llama.rename_col('title', 'title-llama')
news_llama.rename_col('subtitle', 'subtitle-llama')
news_llama.rename_col('body', 'body-llama')
news_llama.rename_col('category', 'category-llama')

news.inject(news_llama, ['title-llama', 'subtitle-llama', 'body-llama', 'category-llama'])


news.export('ebnerd_small_tokenized_2/news-fusion')
