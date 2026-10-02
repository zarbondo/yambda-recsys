# синтетика в схеме yambda, чтобы прогнать весь пайплайн без скачивания 50M:
# у юзеров есть любимые артисты, альбомы слушают подряд, популярность с длинным хвостом
import numpy as np
import polars as pl

from src.data.split import DAY

ALBUM_SIZE = 10
ALBUMS_PER_ARTIST = 3


def generate(cfg, seed):
    rng = np.random.default_rng(seed)
    n_albums = cfg["albums"]
    n_items = n_albums * ALBUM_SIZE
    album_artist = np.arange(n_albums) // ALBUMS_PER_ARTIST
    n_artists = album_artist.max() + 1
    album_pop = rng.permutation(1 / np.arange(1, n_albums + 1) ** 0.8)
    track_len = rng.integers(120, 360, n_items)

    L = {k: [] for k in ["uid", "item_id", "timestamp", "is_organic", "played_ratio_pct", "track_length_seconds"]}
    K = {k: [] for k in ["uid", "item_id", "timestamp", "is_organic"]}
    for u in range(cfg["users"]):
        taste = rng.dirichlet(np.full(n_artists, 0.05))  # пара-тройка любимых артистов
        w = album_pop * (0.2 + 5 * taste[album_artist])
        w /= w.sum()
        activity = rng.uniform(0.5, 3)
        for d in range(cfg["days"]):
            for _ in range(rng.poisson(activity)):
                t = d * DAY + int(rng.integers(0, DAY - 3000))
                album = rng.choice(n_albums, p=w)
                fav = taste[album_artist[album]] > 0.05
                start = rng.integers(0, ALBUM_SIZE)
                for j in range(rng.integers(2, 9)):
                    item = album * ALBUM_SIZE + (start + j) % ALBUM_SIZE
                    ratio = int(np.clip(rng.normal(80 if fav else 45, 30), 1, 120))
                    organic = int(rng.random() < 0.6)
                    for k, v in zip(L, [u, item, t, organic, ratio, track_len[item]]):
                        L[k].append(v)
                    if ratio >= 50 and fav and rng.random() < 0.05:
                        for k, v in zip(K, [u, item, t, organic]):
                            K[k].append(v)
                    t += track_len[item] * min(ratio, 100) // 100 // 5 + 1

    listens = pl.DataFrame(L).sort(["uid", "timestamp"])
    likes = pl.DataFrame(K).sort(["uid", "timestamp"])
    artists = pl.DataFrame({
        "item_id": np.arange(n_items, dtype=np.uint32),
        "artist_id": album_artist[np.arange(n_items) // ALBUM_SIZE].astype(np.uint32),
    })
    print("синтетика: %d прослушиваний, %d лайков" % (listens.height, likes.height))
    return listens, likes, artists
