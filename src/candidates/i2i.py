# item2item по переходам: какие треки слушают сразу после данного в рамках одной сессии
import polars as pl

from src.data.split import DAY, MINUTE


def fit_transitions(hist_l, cutoff, cfg):
    df = (hist_l.filter(pl.col("is_plus") & (pl.col("timestamp") >= cutoff - cfg["days"] * DAY))
                .select("uid", "item_id", "timestamp")
                .sort(["uid", "timestamp"]))
    df = df.with_columns(
        pl.col("item_id").shift(-1).over("uid").alias("next_item"),
        (pl.col("timestamp").shift(-1).over("uid") - pl.col("timestamp")).alias("dt"),
    )
    pairs = df.filter(pl.col("next_item").is_not_null()
                      & (pl.col("dt") <= cfg["session_gap_minutes"] * MINUTE)
                      & (pl.col("next_item") != pl.col("item_id")))
    cnt = (pairs.group_by("item_id", "next_item").agg(pl.len().alias("n"))
                .filter(pl.col("n") >= cfg["min_count"]))
    # p = доля переходов item -> next среди всех переходов из item
    cnt = cnt.with_columns((pl.col("n") / pl.col("n").sum().over("item_id")).alias("p"))
    nb = (cnt.sort(["item_id", "n"], descending=[False, True])
             .group_by("item_id", maintain_order=True).head(cfg["neighbors"]))
    print("i2i: %d пар переходов" % nb.height)
    return nb.select("item_id", pl.col("next_item").alias("nb_item"), "p")


def i2i_candidates(hist_l, nb, seen, users, cutoff, cfg):
    # последние треки юзера; чем свежее трек, тем больше вес его соседей
    last = (hist_l.filter(pl.col("is_plus") & (pl.col("timestamp") >= cutoff - cfg["days"] * DAY))
                  .join(users, on="uid", how="semi")
                  .group_by("uid", "item_id").agg(pl.col("timestamp").max())
                  .with_columns(pl.col("timestamp").rank("ordinal", descending=True).over("uid").alias("pos"))
                  .filter(pl.col("pos") <= cfg["history_len"]))
    cand = (last.join(nb, on="item_id")
                .with_columns((pl.col("p") / pl.col("pos")).alias("w"))
                .group_by("uid", "nb_item").agg(pl.col("w").sum().alias("i2i_score"))
                .rename({"nb_item": "item_id"}))
    cand = cand.join(seen, on=["uid", "item_id"], how="anti")
    cand = cand.with_columns(pl.col("i2i_score").rank("ordinal", descending=True).over("uid").alias("i2i_rank"))
    return cand.filter(pl.col("i2i_rank") <= cfg["n"])
