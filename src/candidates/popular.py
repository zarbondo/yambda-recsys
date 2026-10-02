# популярное за последние дни: и бейзлайн MostPop, и кандидаты для юзеров, где остальным не хватает сигнала
import polars as pl

from src.data.split import DAY


def fit_popular(hist_l, cutoff, days, top):
    return (hist_l.filter(pl.col("is_plus") & (pl.col("timestamp") >= cutoff - days * DAY))
                  .group_by("item_id").agg(pl.len().alias("pop_cnt"))
                  .sort(["pop_cnt", "item_id"], descending=[True, False])
                  .head(top))


def popular_candidates(users, pop, seen, n):
    # каждому юзеру топ популярного, которого он ещё не слушал
    cand = users.join(pop.select("item_id", "pop_cnt"), how="cross")
    cand = cand.join(seen, on=["uid", "item_id"], how="anti")
    cand = cand.with_columns(pl.col("pop_cnt").rank("ordinal", descending=True).over("uid").alias("pop_rank"))
    return cand.filter(pl.col("pop_rank") <= n).select("uid", "item_id", "pop_rank")
