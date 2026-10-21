"""City visualization from TencentVectorDemo-cn.ipynb, cells 19-23."""
import json
from pathlib import Path

import numpy as np

from vector_engine import VectorEngine

LOCATIONS = tuple(json.loads(Path(__file__).with_name("notebook_locations.json").read_text(encoding="utf-8")))
PROJECTION_VERSION = 3

# Selected on the original 134 places; revalidated after list changes, using
# cosine-neighbor recall and trustworthiness
# at k=5,10,15. No location names or geographic coordinates enter the fit.
PROJECTION_PARAMS = {
    2: dict(perplexity=30, learning_rate=25, random_state=0),
    3: dict(perplexity=20, learning_rate=10, random_state=3),
}


def location_vectors(engine: VectorEngine) -> np.ndarray:
    """Keep the Notebook's order and jieba-average fallback for missing names."""
    import jieba

    vectors = []
    for place in LOCATIONS:
        if place in engine.model:
            vectors.append(engine.vector(place))
        else:
            words = jieba.lcut(place)
            vector = np.zeros(engine.model.vector_size, dtype="float32")
            for word in words:
                vector += engine.vector(word)
            vectors.append(vector / len(words))
    return np.asarray(vectors)


def reduce_dimensions(vectors: np.ndarray, num_dimensions: int = 2) -> np.ndarray:
    from sklearn.manifold import TSNE
    from threadpoolctl import threadpool_limits

    if num_dimensions not in (2, 3):
        raise ValueError("仅支持二维或三维可视化。")
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[0] < 4 or vectors.shape[1] < 1 or not np.isfinite(vectors).all():
        raise ValueError("降维需要至少4个有效的非空向量。")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms < 1e-10):
        raise ValueError("零向量无法按方向降维。")
    # On unit vectors, squared Euclidean distance = 2 * cosine distance.
    # Normalize a small copy; the memory-mapped model is never modified.
    unit = vectors / norms
    params = PROJECTION_PARAMS[num_dimensions]
    with threadpool_limits(limits=1):
        return TSNE(n_components=num_dimensions, init="random", early_exaggeration=4,
                    max_iter=2000, method="barnes_hut", metric="euclidean",
                    perplexity=min(params["perplexity"], len(unit) - 1),
                    learning_rate=params["learning_rate"], random_state=params["random_state"]).fit_transform(unit)


def cluster_vectors(vectors: np.ndarray, num_groups: int = 8) -> list[str]:
    """Group normalized original vectors; labels are shared by all projections."""
    from sklearn.cluster import KMeans
    from threadpoolctl import threadpool_limits

    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("分组需要有效的二维向量矩阵。")
    if not 2 <= num_groups <= len(vectors):
        raise ValueError("分组数量需在2和词语数量之间。")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms < 1e-10):
        raise ValueError("零向量无法按方向分组。")
    with threadpool_limits(limits=1):
        labels = KMeans(n_clusters=num_groups, random_state=0, n_init=10).fit_predict(vectors / norms)
    # Number groups by first appearance in the fixed place list for stable colors.
    numbers = {label: index + 1 for index, label in enumerate(dict.fromkeys(labels))}
    return [f"第{numbers[label]}组" for label in labels]
