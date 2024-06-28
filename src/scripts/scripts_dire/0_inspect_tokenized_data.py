"""
Use to check if all columns were tokenized.
"""



from UniTok import UniDep

print("\nSENTIMENT:")
news = UniDep('/scratch-shared/scur1569/ebnerd_small_tokenized-sentiment/news') #directory path - no specific file
print(f"News:\n{news[0]}")



print("\nGENRE:")
train = UniDep('/scratch-shared/scur1569/ebnerd_small_tokenized-genre/train') #directory path - no specific file
print(f"Train:\n{train[0]}")

user = UniDep('/scratch-shared/scur1569/ebnerd_small_tokenized-genre/user') #directory path - no specific file
print(f"Train:\n{user[0]}")

news = UniDep('/scratch-shared/scur1569/ebnerd_small_tokenized-genre/news') #directory path - no specific file
print(f"News:\n{news[0]}")