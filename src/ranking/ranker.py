# ранжирование кандидатов catboost'ом с лоссом YetiRank, группа = юзер
import numpy as np
import polars as pl
from catboost import CatBoostRanker, Pool


def _pool(df, feats):
    return Pool(df.select(feats).cast(pl.Float32).to_numpy(),
                label=df["target"].to_numpy().astype(np.float32),
                group_id=df["uid"].to_numpy())


def train_ranker(df, feats, cfg, seed, train_dir):
    # юзеры без позитивов среди кандидатов ранкеру ничего не дают
    good = df.group_by("uid").agg(pl.col("target").max()).filter(pl.col("target") == 1).select("uid")
    users = good["uid"].shuffle(seed=seed)
    val_users = users[:max(1, int(len(users) * cfg["val_users_frac"]))].to_frame()
    df = df.join(good, on="uid", how="semi")
    tr = df.join(val_users, on="uid", how="anti").sort("uid", maintain_order=True)
    va = df.join(val_users, on="uid", how="semi").sort("uid", maintain_order=True)
    print("ранкер: %d юзеров, %d строк в трейне, %d в валидации, доля позитивов %.3f"
          % (len(users), tr.height, va.height, tr["target"].mean()))

    model = CatBoostRanker(
        loss_function="YetiRank", eval_metric="NDCG:top=10",
        iterations=cfg["iterations"], learning_rate=cfg["learning_rate"], depth=cfg["depth"],
        task_type=cfg["task_type"], random_seed=seed, verbose=50, train_dir=train_dir)
    model.fit(_pool(tr, feats), eval_set=_pool(va, feats),
              early_stopping_rounds=cfg["early_stopping"], use_best_model=True)
    return model


def rank(model, df, feats):
    score = model.predict(df.select(feats).cast(pl.Float32).to_numpy())
    df = df.select("uid", "item_id").with_columns(pl.Series("score", score))
    return df.with_columns(pl.col("score").rank("ordinal", descending=True).over("uid").alias("rank"))
