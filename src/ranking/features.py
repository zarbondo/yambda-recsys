# фичи для ранкера; всё считается только по истории до cutoff, будущего тут нет
import polars as pl

from src.data.split import DAY


def item_features(hist_l, hist_k, cutoff):
    f = hist_l.group_by("item_id").agg(
        pl.len().alias("item_listens"),
        pl.col("is_plus").sum().alias("item_plus"),
        pl.col("uid").n_unique().alias("item_users"),
        (pl.col("is_plus") & (pl.col("timestamp") >= cutoff - 7 * DAY)).sum().alias("item_plus_7d"),
        (pl.col("is_plus") & (pl.col("timestamp") >= cutoff - DAY)).sum().alias("item_plus_1d"),
        pl.col("is_organic").mean().alias("item_organic_share"),
        pl.col("track_length_seconds").median().alias("track_len"),
        pl.col("timestamp").min().alias("first_ts"),
    )
    likes = hist_k.group_by("item_id").agg(pl.len().alias("item_likes"))
    f = f.join(likes, on="item_id", how="left").with_columns(pl.col("item_likes").fill_null(0))
    return f.with_columns(
        (1 - pl.col("item_plus") / pl.col("item_listens")).alias("item_skip_rate"),
        (pl.col("item_plus_1d") / (pl.col("item_plus_7d") + 1)).alias("item_trend"),
        (pl.col("item_likes") / pl.col("item_users")).alias("item_like_rate"),
        ((cutoff - pl.col("first_ts")) / DAY).alias("item_age_days"),
    ).drop("first_ts")


def user_features(hist_l, hist_k, cutoff):
    f = hist_l.group_by("uid").agg(
        pl.len().alias("user_listens"),
        pl.col("is_plus").mean().alias("user_plus_rate"),
        pl.col("item_id").n_unique().alias("user_items"),
        (pl.col("timestamp") >= cutoff - 7 * DAY).sum().alias("user_listens_7d"),
        pl.col("is_organic").mean().alias("user_organic_share"),
        ((cutoff - pl.col("timestamp").max()) / DAY).alias("user_days_inactive"),
    )
    likes = hist_k.group_by("uid").agg(pl.len().alias("user_likes"))
    return f.join(likes, on="uid", how="left").with_columns(pl.col("user_likes").fill_null(0))


def artist_affinity(hist_l, artists):
    # какая доля listen+ юзера приходится на каждого артиста
    ua = (hist_l.filter(pl.col("is_plus")).select("uid", "item_id")
                .join(artists, on="item_id")
                .group_by("uid", "artist_id").agg(pl.len().alias("artist_cnt")))
    return ua.with_columns((pl.col("artist_cnt") / pl.col("artist_cnt").sum().over("uid")).alias("artist_share"))


def build(cands, hist_l, hist_k, cutoff, als, artists=None, targets=None):
    df = cands.join(als.score(cands.select("uid", "item_id")), on=["uid", "item_id"], how="left")
    df = df.join(item_features(hist_l, hist_k, cutoff), on="item_id", how="left")
    df = df.join(user_features(hist_l, hist_k, cutoff), on="uid", how="left")
    if artists is not None:
        df = (df.join(artists, on="item_id", how="left")
                .join(artist_affinity(hist_l, artists), on=["uid", "artist_id"], how="left")
                .with_columns(pl.col("artist_cnt").fill_null(0), pl.col("artist_share").fill_null(0.0))
                .drop("artist_id"))
    if targets is not None:
        df = (df.join(targets.with_columns(pl.lit(1, dtype=pl.UInt8).alias("target")),
                      on=["uid", "item_id"], how="left")
                .with_columns(pl.col("target").fill_null(0)))
    return df


def feature_columns(df):
    return [c for c in df.columns if c not in ("uid", "item_id", "target")]
