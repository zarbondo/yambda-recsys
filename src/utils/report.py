# сохранение результатов и таблица в README
import json
import re
from pathlib import Path

import polars as pl

NAMES = {
    "mostpop": "MostPop (7 дней)",
    "als": "ALS",
    "i2i": "i2i (переходы)",
    "ranker": "ALS + i2i + pop → CatBoost YetiRank",
}


def table(metrics, ks):
    cols = [f"{m}@{k}" for k in ks for m in ("ndcg", "recall")]
    head = "| модель | " + " | ".join(c.replace("ndcg", "NDCG").replace("recall", "Recall") for c in cols) + " |"
    lines = [head, "|---" * (len(cols) + 1) + "|"]
    for key, m in metrics.items():
        lines.append(f"| {NAMES[key]} | " + " | ".join(f"{m[c]:.4f}" for c in cols) + " |")
    return "\n".join(lines)


def save(summary, out):
    out = Path(out)
    (out / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = [{"model": NAMES[k], **v} for k, v in summary["metrics"].items()]
    pl.DataFrame(rows).write_csv(out / "comparison.csv")


def update_readme(tbl, summary, path="README.md"):
    pool = summary["candidate_pool"]
    block = (f"{tbl}\n\n"
             f"NDCG@10 ранкера относительно ALS: {summary['ndcg@10_gain_vs_als']:+.1%}. "
             f"Recall пула кандидатов: {pool['recall']:.4f} "
             f"(в среднем {pool['avg_size']:.0f} кандидатов на юзера, {summary['n_test_users']} юзеров в тесте).")
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    text = re.sub(r"<!-- results -->.*?<!-- /results -->",
                  lambda _: f"<!-- results -->\n{block}\n<!-- /results -->", text, flags=re.S)
    p.write_text(text, encoding="utf-8")
