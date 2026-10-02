import polars as pl

from src.data.split import DAY, make_borders, slice_phase

EMPTY_LIKES = pl.DataFrame(schema={"uid": pl.Int64, "item_id": pl.Int64, "timestamp": pl.Int64})


def test_borders_go_in_order():
    b = make_borders(100 * DAY, {"split": {"test_days": 1, "val_days": 1, "gap_minutes": 30}})
    assert b["val_hist_end"] < b["val_start"] < b["val_end"] == b["test_hist_end"] < b["test_start"] < b["test_end"]


def test_history_stops_before_window_and_targets_are_new():
    listens = pl.DataFrame({
        "uid": [1, 1, 1, 2, 2], "item_id": [10, 11, 12, 10, 13],
        "timestamp": [0, DAY, 3 * DAY, 0, 3 * DAY], "is_plus": [True] * 5,
    })
    likes = pl.DataFrame({"uid": [1], "item_id": [14], "timestamp": [3 * DAY]})
    hist_l, _, _, targets = slice_phase(listens, likes, 2 * DAY, 2 * DAY + 10, 4 * DAY)
    assert hist_l["timestamp"].max() < 2 * DAY
    assert set(targets.rows()) == {(1, 12), (1, 14), (2, 13)}


def test_already_seen_track_is_not_a_target():
    listens = pl.DataFrame({"uid": [1, 1], "item_id": [10, 10], "timestamp": [0, 3 * DAY], "is_plus": [True, True]})
    *_, targets = slice_phase(listens, EMPTY_LIKES, 2 * DAY, 2 * DAY, 4 * DAY)
    assert targets.height == 0


def test_skipped_listen_is_not_a_target():
    listens = pl.DataFrame({"uid": [1, 1], "item_id": [10, 11], "timestamp": [0, 3 * DAY], "is_plus": [True, False]})
    *_, targets = slice_phase(listens, EMPTY_LIKES, 2 * DAY, 2 * DAY, 4 * DAY)
    assert targets.height == 0
