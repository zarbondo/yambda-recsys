# загрузка yambda с huggingface и приведение типов
import polars as pl
from huggingface_hub import hf_hub_download

REPO = "yandex/yambda"
LISTEN_COLS = ["uid", "item_id", "timestamp", "is_organic", "played_ratio_pct", "track_length_seconds"]
LIKE_COLS = ["uid", "item_id", "timestamp", "is_organic"]


def download(filename, raw_dir):
    return hf_hub_download(repo_id=REPO, repo_type="dataset", filename=filename, local_dir=raw_dir)


def load_yambda(cfg):
    size, raw_dir = cfg["data"]["size"], cfg["data"]["raw_dir"]
    listens = pl.read_parquet(download(f"flat/{size}/listens.parquet", raw_dir), columns=LISTEN_COLS)
    likes = pl.read_parquet(download(f"flat/{size}/likes.parquet", raw_dir), columns=LIKE_COLS)
    return listens, likes


def load_artists(cfg):
    # маппинг трек -> артист лежит в корне репозитория датасета; схему проверяем на всякий случай
    try:
        df = pl.read_parquet(download("artist_item_mapping.parquet", cfg["data"]["raw_dir"]))
        item_col = "item_id" if "item_id" in df.columns else [c for c in df.columns if "item" in c][0]
        art_col = [c for c in df.columns if "artist" in c][0]
        if isinstance(df.schema[art_col], pl.List):
            df = df.with_columns(pl.col(art_col).list.first())  # у трека бывает несколько артистов
        return (df.select(pl.col(item_col).cast(pl.UInt32).alias("item_id"),
                          pl.col(art_col).cast(pl.UInt32).alias("artist_id"))
                  .drop_nulls().unique("item_id"))
    except Exception as e:
        print("маппинг артистов не загрузился, фичи по артистам пропускаю:", e)
        return None


def prepare(listens, likes, threshold):
    listens = listens.select(
        pl.col("uid").cast(pl.UInt32),
        pl.col("item_id").cast(pl.UInt32),
        pl.col("timestamp").cast(pl.Int64),
        pl.col("is_organic").cast(pl.UInt8),
        pl.col("played_ratio_pct").cast(pl.UInt16),
        pl.col("track_length_seconds").cast(pl.UInt32),
    ).with_columns((pl.col("played_ratio_pct") >= threshold).alias("is_plus"))
    likes = likes.select(
        pl.col("uid").cast(pl.UInt32),
        pl.col("item_id").cast(pl.UInt32),
        pl.col("timestamp").cast(pl.Int64),
        pl.col("is_organic").cast(pl.UInt8),
    )
    return listens, likes
