import math

import polars as pl

from src.evaluation.metrics import evaluate, pool_recall


def test_perfect_ranking():
    recs = pl.DataFrame({"uid": [1, 1, 1], "item_id": [10, 11, 12], "rank": [1, 2, 3]})
    targets = pl.DataFrame({"uid": [1, 1], "item_id": [10, 11]})
    m = evaluate(recs, targets, [10])
    assert math.isclose(m["ndcg@10"], 1.0)
    assert m["recall@10"] == 1.0


def test_hit_on_second_place():
    recs = pl.DataFrame({"uid": [1, 1], "item_id": [5, 7], "rank": [1, 2]})
    targets = pl.DataFrame({"uid": [1], "item_id": [7]})
    m = evaluate(recs, targets, [10])
    assert math.isclose(m["ndcg@10"], 1 / math.log2(3))


def test_user_without_recs_counts_as_zero():
    recs = pl.DataFrame({"uid": [1], "item_id": [7], "rank": [1]})
    targets = pl.DataFrame({"uid": [1, 2], "item_id": [7, 8]})
    assert math.isclose(evaluate(recs, targets, [10])["recall@10"], 0.5)


def test_k_cuts_the_tail():
    recs = pl.DataFrame({"uid": [1, 1], "item_id": [1, 2], "rank": [1, 11]})
    targets = pl.DataFrame({"uid": [1], "item_id": [2]})
    m = evaluate(recs, targets, [10, 100])
    assert m["recall@10"] == 0.0 and m["recall@100"] == 1.0


def test_pool_recall():
    cands = pl.DataFrame({"uid": [1, 1, 2], "item_id": [1, 2, 3]})
    targets = pl.DataFrame({"uid": [1, 2], "item_id": [2, 4]})
    p = pool_recall(cands, targets)
    assert math.isclose(p["recall"], 0.5) and math.isclose(p["avg_size"], 1.5)
