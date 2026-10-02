# ndcg@k и recall@k с бинарной релевантностью
import numpy as np
import polars as pl


def evaluate(recs, targets, ks):
    """recs: uid, item_id, rank (с 1); targets: uid, item_id.
    усредняем по всем юзерам с таргетами: нет рекомендаций — метрики юзера нулевые"""
    n_t = targets.group_by("uid").agg(pl.len().alias("n_t"))
    hits = recs.join(targets, on=["uid", "item_id"], how="semi").select("uid", "rank")
    res = {}
    for k in ks:
        h = (hits.filter(pl.col("rank") <= k)
                 .group_by("uid").agg(pl.len().alias("hits"),
                                      (1 / (pl.col("rank") + 1).log(2)).sum().alias("dcg")))
        per = n_t.join(h, on="uid", how="left").fill_null(0)
        n = per["n_t"].to_numpy()
        # идеальный dcg: все min(n_t, k) попаданий стоят в начале списка
        ideal = np.concatenate([[0.0], np.cumsum(1 / np.log2(np.arange(2, k + 2)))])
        res[f"ndcg@{k}"] = float(np.mean(per["dcg"].to_numpy() / ideal[np.minimum(n, k)]))
        res[f"recall@{k}"] = float(np.mean(per["hits"].to_numpy() / n))
    return res


def pool_recall(cands, targets):
    # какую долю таргетов вообще видит ранкер
    n_t = targets.group_by("uid").agg(pl.len().alias("n_t"))
    h = cands.join(targets, on=["uid", "item_id"], how="semi").group_by("uid").agg(pl.len().alias("hits"))
    per = n_t.join(h, on="uid", how="left").fill_null(0)
    size = cands.group_by("uid").agg(pl.len().alias("size"))["size"].mean()
    return {"recall": float((per["hits"] / per["n_t"]).mean()), "avg_size": float(size)}
