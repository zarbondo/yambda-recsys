# двухэтапная рексис на yambda: кандидаты (als + i2i + популярное) -> catboost yetirank
# запуск: python main.py            — полный прогон на yambda-50m
#         python main.py --smoke    — быстрый прогон на синтетике
import argparse
import gc
import time
from pathlib import Path

import mlflow
import polars as pl
import yaml

from src.candidates.popular import popular_candidates
from src.data import load, synthetic
from src.data.split import DAY, make_borders
from src.evaluation.metrics import evaluate, pool_recall
from src.pipeline import build_phase
from src.ranking.features import feature_columns
from src.ranking.ranker import rank, train_ranker
from src.utils import plots, report


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten(v, f"{prefix}{k}."))
        else:
            out[prefix + k] = v
    return out


def fill_missing(recs, fallback, users):
    # юзерам, которым генератор ничего не выдал, отдаём популярное
    missing = users.join(recs.select("uid").unique(), on="uid", how="anti")
    return pl.concat([recs.select("uid", "item_id", "rank"),
                      fallback.join(missing, on="uid", how="semi").select("uid", "item_id", "rank")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--smoke", action="store_true", help="прогон на синтетике, чтобы проверить пайплайн")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path("configs/smoke.yaml" if args.smoke else args.config).read_text(encoding="utf-8"))
    seed, ks = cfg["seed"], cfg["eval"]["k"]
    out = Path(cfg["results_dir"])
    (out / "plots").mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if "synthetic" in cfg["data"]:
        listens, likes, artists = synthetic.generate(cfg["data"]["synthetic"], seed)
    else:
        listens, likes = load.load_yambda(cfg)
        artists = load.load_artists(cfg) if cfg["data"]["use_artists"] else None
    listens, likes = load.prepare(listens, likes, cfg["data"]["listen_threshold"])
    t_max = max(listens["timestamp"].max(), likes["timestamp"].max())
    b = make_borders(t_max, cfg)
    print("данные: %d прослушиваний, %d лайков, %.0f дней" % (listens.height, likes.height, t_max / DAY))

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    with mlflow.start_run(run_name="smoke" if args.smoke else cfg["data"]["size"]):
        mlflow.log_params({k: str(v) for k, v in flatten(cfg).items()})

        # фаза A: генераторы видят историю до дня ранкера, таргеты — сам этот день
        print("\n== фаза A: учим ранкер")
        a = build_phase(listens, likes, b["val_hist_end"], b["val_start"], b["val_end"], cfg, artists, seed)
        feats = feature_columns(a["features"])
        ranker = train_ranker(a["features"], feats, cfg["ranker"], seed, str(out / "catboost_info"))
        del a
        gc.collect()

        # фаза B: генераторы переобучаем на свежей истории, меряем на тестовом дне
        print("\n== фаза B: тест")
        tb = build_phase(listens, likes, b["test_hist_end"], b["test_start"], b["test_end"], cfg, artists, seed)
        users, targets = tb["users"], tb["targets"]

        pop_recs = popular_candidates(users, tb["pop"], tb["seen"], max(ks)).rename({"pop_rank": "rank"})
        recs = {
            "mostpop": pop_recs,
            "als": fill_missing(tb["als"].rename({"als_rank": "rank"}), pop_recs, users),
            "i2i": tb["i2i"].select("uid", "item_id", pl.col("i2i_rank").alias("rank")),
            "ranker": rank(ranker, tb["features"], feats),
        }
        metrics = {name: evaluate(r, targets, ks) for name, r in recs.items()}
        pool = pool_recall(tb["cands"], targets)
        als10 = metrics["als"]["ndcg@10"]
        summary = {
            "metrics": metrics,
            "candidate_pool": pool,
            "ndcg@10_gain_vs_als": metrics["ranker"]["ndcg@10"] / als10 - 1 if als10 > 0 else None,
            "n_test_users": users.height,
            "features": feats,
        }

        tbl = report.table(metrics, ks)
        print("\n" + tbl)
        print("\nrecall пула кандидатов: %.4f, в среднем %.0f на юзера" % (pool["recall"], pool["avg_size"]))
        if summary["ndcg@10_gain_vs_als"] is not None:
            print("NDCG@10 ранкера относительно ALS: %+.1f%%" % (100 * summary["ndcg@10_gain_vs_als"]))

        report.save(summary, out)
        imp = ranker.get_feature_importance(type="PredictionValuesChange")
        plots.feature_importance(feats, imp, out / "plots" / "feature_importance.png")
        plots.metrics_bar(metrics, "ndcg@10", out / "plots" / "ndcg_at_10.png")
        ranker.save_model(str(out / "ranker.cbm"))
        if not args.smoke and summary["ndcg@10_gain_vs_als"] is not None:
            report.update_readme(tbl, summary)  # в README попадают только цифры с настоящих данных

        for name, m in metrics.items():
            for k, v in m.items():
                mlflow.log_metric(f"{name}.{k.replace('@', '_at_')}", v)
        mlflow.log_metric("pool.recall", pool["recall"])
        mlflow.log_metric("pool.avg_size", pool["avg_size"])
        mlflow.log_artifacts(str(out / "plots"), "plots")
        mlflow.log_artifact(str(out / "metrics.json"))

    print("\nготово за %.1f мин, результаты в %s/" % ((time.time() - t0) / 60, out))


if __name__ == "__main__":
    main()
