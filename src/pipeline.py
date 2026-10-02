# одна фаза: история до cutoff -> генераторы -> кандидаты -> фичи (+ таргеты из окна)
import polars as pl

from src.candidates.als import ALSCandidates
from src.candidates.i2i import fit_transitions, i2i_candidates
from src.candidates.popular import fit_popular, popular_candidates
from src.data.split import slice_phase
from src.ranking.features import build


def build_phase(listens, likes, hist_end, win_start, win_end, cfg, artists, seed):
    c = cfg["candidates"]
    hist_l, hist_k, seen, targets = slice_phase(listens, likes, hist_end, win_start, win_end)
    users = targets.select("uid").unique()
    print("юзеров с таргетами: %d, таргетов: %d" % (users.height, targets.height))

    als = ALSCandidates(c["als"], seed).fit(hist_l, hist_k)
    nb = fit_transitions(hist_l, hist_end, c["i2i"])
    pop = fit_popular(hist_l, hist_end, c["popular"]["days"], top=max(1000, 10 * max(cfg["eval"]["k"])))

    als_c = als.recommend(users, seen, c["als"]["n"])
    i2i_c = i2i_candidates(hist_l, nb, seen, users, hist_end, c["i2i"])
    pop_c = popular_candidates(users, pop, seen, c["popular"]["n"])

    cands = (als_c.join(i2i_c, on=["uid", "item_id"], how="full", coalesce=True)
                  .join(pop_c, on=["uid", "item_id"], how="full", coalesce=True))
    cands = cands.with_columns(
        (pl.col("als_rank").is_not_null().cast(pl.UInt8)
         + pl.col("i2i_rank").is_not_null().cast(pl.UInt8)
         + pl.col("pop_rank").is_not_null().cast(pl.UInt8)).alias("n_sources"))

    feats = build(cands, hist_l, hist_k, hist_end, als, artists, targets)
    return {"features": feats, "targets": targets, "users": users, "seen": seen, "pop": pop,
            "als": als_c, "i2i": i2i_c, "cands": cands.select("uid", "item_id")}
