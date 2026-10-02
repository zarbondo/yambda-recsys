# быстрый взгляд на данные: python -m src.eda (или --smoke на синтетике)
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml

from src.data import load, synthetic
from src.data.split import DAY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path("configs/smoke.yaml" if args.smoke else args.config).read_text(encoding="utf-8"))
    if "synthetic" in cfg["data"]:
        listens, likes, _ = synthetic.generate(cfg["data"]["synthetic"], cfg["seed"])
    else:
        listens, likes = load.load_yambda(cfg)
    listens, likes = load.prepare(listens, likes, cfg["data"]["listen_threshold"])
    out = Path(cfg["results_dir"]) / "eda"
    out.mkdir(parents=True, exist_ok=True)

    listens = listens.sort(["uid", "timestamp"])
    repeat = 1 - listens.select(pl.struct("uid", "item_id").is_first_distinct().mean()).item()
    print("юзеров: %d, треков: %d" % (listens["uid"].n_unique(), listens["item_id"].n_unique()))
    print("прослушиваний: %d, лайков: %d, период: %.0f дней"
          % (listens.height, likes.height, listens["timestamp"].max() / DAY))
    print("доля listen+: %.3f, доля органики: %.3f" % (listens["is_plus"].mean(), listens["is_organic"].mean()))
    print("доля повторных прослушиваний (трек уже был у юзера): %.3f" % repeat)

    hist = listens.group_by("uid").agg(pl.len().alias("n"))["n"].to_numpy()
    pop = np.sort(listens.group_by("item_id").agg(pl.len().alias("n"))["n"].to_numpy())[::-1]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].hist(np.log10(hist), bins=50)
    ax[0].set_title("длина истории юзера, log10")
    ax[1].loglog(np.arange(1, len(pop) + 1), pop)
    ax[1].set_title("популярность треков (rank-frequency)")
    ax[2].hist(listens["played_ratio_pct"].clip(upper_bound=150).to_numpy(), bins=50)
    ax[2].axvline(cfg["data"]["listen_threshold"], color="r", ls="--")
    ax[2].set_title("played_ratio_pct, красная линия — порог listen+")
    plt.tight_layout()
    plt.savefig(out / "overview.png", dpi=110)
    print("графики в", out)


if __name__ == "__main__":
    main()
