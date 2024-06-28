"""
Experimenting with the eb-nerd loader
"""


"""
Outputs one row per impression = article clicked on 

impression_id
article_id
impression_time: timestamp of accessing article
read_time
scroll_percentage: 0.0 to 100.0
device_type
article_ids_inview: all articles the user got a preview from
article_ids_clicked: often = article_id
user_id
... more user data like location and subscriber status...
session_id
next_read_time
next_scroll_percentage
article_id_fixed

"""


from pathlib import Path
import polars as pl

from ebrec.utils._descriptive_analysis import (
    min_max_impression_time_behaviors, 
    min_max_impression_time_history
)
from ebrec.utils._polars import slice_join_dataframes
from ebrec.utils._behaviors import (
    create_binary_labels_column,
    sampling_strategy_wu2019,
    truncate_history,
)
from ebrec.utils._constants import (
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_USER_COL
)


PATH = Path("ebnerd_small/")
data_split = "train"

df_behaviors = pl.scan_parquet(PATH.joinpath(data_split, "behaviors.parquet"))
df_history = pl.scan_parquet(PATH.joinpath(data_split, "history.parquet"))


#check min max timestamp
print(f"History: {min_max_impression_time_history(df_history).collect()}")
print(f"Behaviors: {min_max_impression_time_behaviors(df_behaviors).collect()}")




#selects relevant columns from the history df and truncates the history to a fixed size (30) per user.
df_history = df_history.select(DEFAULT_USER_COL, DEFAULT_HISTORY_ARTICLE_ID_COL).pipe(
    truncate_history,
    column=DEFAULT_HISTORY_ARTICLE_ID_COL,
    history_size=30,
    padding_value=0,
    enable_warning=False,
)
df_history.head(5)


df_history_p = df_history.collect()
pl.Config.set_tbl_cols(100)
pl.Config.set_tbl_rows(4) 
print('df_history:\n')
print(df_history_p)
print('\n\n')


df = slice_join_dataframes(
    df1=df_behaviors.collect(),
    df2=df_history.collect(),
    on=DEFAULT_USER_COL,
    how="left",
)

print(df)


df = df.select(["user_id", "impression_id", "article_id_fixed", DEFAULT_CLICKED_ARTICLES_COL, DEFAULT_INVIEW_ARTICLES_COL]).pipe(
    create_binary_labels_column, shuffle=True, seed=123
).with_columns(pl.col("labels").list.len().name.suffix("_len")).head(5)

# Set display configuration and print all column names
pl.Config.set_tbl_cols(100)
pl.Config.set_tbl_rows(2)
print("All column names:\n", df.columns)
print(df.head(4))

print("Output as needed for ONCE:")

df_labeled = df.with_columns(
    pl.struct(["article_ids_inview", "labels"]).apply(lambda row: [f"{id}-{label}" for id, label in zip(row["article_ids_inview"], row["labels"])]).alias("predict")
)
print(df_labeled)
exit()

#downsample strategy (wu)
NPRATIO = 2
df.select(DEFAULT_CLICKED_ARTICLES_COL, DEFAULT_INVIEW_ARTICLES_COL).pipe(
    sampling_strategy_wu2019, npratio=NPRATIO, shuffle=False, with_replacement=True, seed=123
).pipe(create_binary_labels_column, shuffle=True, seed=123).with_columns(pl.col("labels").list.len().name.suffix("_len")).head(5)