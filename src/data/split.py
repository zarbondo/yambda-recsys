# глобальный временной сплит, как в статье yambda: история / 30 минут зазора / 1 день теста
import polars as pl

DAY = 24 * 60 * 60        # timestamp в yambda — в секундах
MINUTE = 60


def make_borders(t_max, cfg):
    s = cfg["split"]
    gap = s["gap_minutes"] * MINUTE
    test_start = t_max - s["test_days"] * DAY
    test_hist_end = test_start - gap          # история для фазы теста
    val_start = test_hist_end - s["val_days"] * DAY
    val_hist_end = val_start - gap            # история для фазы обучения ранкера
    return {
        "val_hist_end": val_hist_end, "val_start": val_start, "val_end": test_hist_end,
        "test_hist_end": test_hist_end, "test_start": test_start, "test_end": t_max + 1,
    }


def slice_phase(listens, likes, hist_end, win_start, win_end):
    hist_l = listens.filter(pl.col("timestamp") < hist_end)
    hist_k = likes.filter(pl.col("timestamp") < hist_end)

    # всё, с чем юзер уже сталкивался (включая пропущенные треки), не рекомендуем
    seen = pl.concat([hist_l.select("uid", "item_id"), hist_k.select("uid", "item_id")]).unique()

    # таргет: listen+ или лайк в окне, причём трек для юзера новый
    in_win = pl.col("timestamp").is_between(win_start, win_end, closed="left")
    targets = pl.concat([
        listens.filter(in_win & pl.col("is_plus")).select("uid", "item_id"),
        likes.filter(in_win).select("uid", "item_id"),
    ]).unique()
    targets = targets.join(seen, on=["uid", "item_id"], how="anti")
    # юзеров без истории выкидываем, как в статье
    targets = targets.join(hist_l.select("uid").unique(), on="uid", how="semi")

    assert hist_l["timestamp"].max() < win_start, "утечка: история заходит в окно таргетов"
    return hist_l, hist_k, seen, targets
