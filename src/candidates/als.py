# ALS (implicit) по матрице юзер x трек: вес = log(1 + число listen+) + бонус за лайк
import numpy as np
import polars as pl
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares
from threadpoolctl import threadpool_limits


def _np(x):
    return x.to_numpy() if hasattr(x, "to_numpy") else np.asarray(x)


class ALSCandidates:
    def __init__(self, cfg, seed):
        self.cfg = cfg
        self.seed = seed

    def fit(self, hist_l, hist_k):
        c = self.cfg
        plus = hist_l.filter(pl.col("is_plus")).group_by("uid", "item_id").agg(pl.len().alias("cnt"))
        liked = hist_k.select("uid", "item_id").unique().with_columns(pl.lit(1.0).alias("liked"))
        inter = plus.join(liked, on=["uid", "item_id"], how="full", coalesce=True).fill_null(0)

        # редкие треки только шумят и раздувают матрицу
        keep = (inter.group_by("item_id").agg(pl.len().alias("n"))
                     .filter(pl.col("n") >= c["min_item_count"]).select("item_id"))
        inter = inter.join(keep, on="item_id", how="semi")

        self.user_map = inter.select("uid").unique().sort("uid").with_row_index("uidx")
        self.item_map = inter.select("item_id").unique().sort("item_id").with_row_index("iidx")
        inter = inter.join(self.user_map, on="uid").join(self.item_map, on="item_id")

        w = (np.log1p(inter["cnt"].to_numpy().astype(np.float32))
             + c["like_weight"] * inter["liked"].to_numpy().astype(np.float32))
        shape = (self.user_map.height, self.item_map.height)
        self.ui = sp.csr_matrix((w, (inter["uidx"].to_numpy(), inter["iidx"].to_numpy())), shape=shape)

        self.model = AlternatingLeastSquares(
            factors=c["factors"], regularization=c["regularization"], alpha=c["alpha"],
            iterations=c["iterations"], random_state=self.seed)
        with threadpool_limits(1, "blas"):  # implicit сам параллелит, blas-потоки ему мешают
            self.model.fit(self.ui, show_progress=True)
        self.user_f = _np(self.model.user_factors)
        self.item_f = _np(self.model.item_factors)
        print("als: %d юзеров x %d треков, %d ненулевых" % (shape[0], shape[1], self.ui.nnz))
        return self

    def recommend(self, users, seen, n):
        u = users.join(self.user_map, on="uid")
        uidx = u["uidx"].to_numpy()
        # implicit не знает про пропущенные треки — берём с запасом и чистим по seen сами
        n_req = min(n + 50, self.item_map.height)
        ids, _ = self.model.recommend(uidx, self.ui[uidx], N=n_req, filter_already_liked_items=True)
        ids = np.asarray(ids)
        flat = ids.ravel()
        ok = flat >= 0
        df = pl.DataFrame({
            "uid": np.repeat(u["uid"].to_numpy(), ids.shape[1])[ok],
            "iidx": flat[ok].astype(np.uint32),
            "pos": np.tile(np.arange(ids.shape[1], dtype=np.uint32), len(uidx))[ok],
        })
        df = df.join(self.item_map, on="iidx").drop("iidx")
        df = df.join(seen, on=["uid", "item_id"], how="anti")
        df = df.with_columns(pl.col("pos").rank("ordinal").over("uid").alias("als_rank")).drop("pos")
        return df.filter(pl.col("als_rank") <= n)

    def score(self, pairs):
        # скор als для любых пар юзер-трек (nan, если юзера или трека нет в матрице)
        p = pairs.join(self.user_map, on="uid", how="left").join(self.item_map, on="item_id", how="left")
        ui = p["uidx"].fill_null(0).to_numpy()
        ii = p["iidx"].fill_null(0).to_numpy()
        ok = (p["uidx"].is_not_null() & p["iidx"].is_not_null()).to_numpy()
        out = np.empty(p.height, dtype=np.float32)
        step = 500_000
        for s in range(0, p.height, step):
            out[s:s + step] = np.einsum("ij,ij->i", self.user_f[ui[s:s + step]], self.item_f[ii[s:s + step]])
        out[~ok] = np.nan
        return p.select("uid", "item_id").with_columns(pl.Series("als_score", out))
