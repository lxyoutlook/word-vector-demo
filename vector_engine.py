"""Pure word-vector operations, independent of Streamlit and mutable UI state."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gensim.models import KeyedVectors
import numpy as np


def unit(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if not np.isfinite(length) or length < 1e-10:
        raise ValueError("组合后的向量接近零，无法计算方向。请调整输入词语。")
    return vector / length


@dataclass(frozen=True)
class VectorEngine:
    model: KeyedVectors

    @classmethod
    def load(cls, path: str | Path) -> "VectorEngine":
        path = Path(path)
        companion = Path(str(path) + ".vectors.npy")
        if not path.is_file() or not companion.is_file():
            raise FileNotFoundError("请将原始 .bin 和 .bin.vectors.npy 两个配套文件放在同一文件夹。")
        model = KeyedVectors.load(str(path), mmap="r")
        if not isinstance(model, KeyedVectors):
            raise ValueError("需要 Gensim KeyedVectors 原生模型文件。")
        model.fill_norms()
        # No cached shared model is modified during user requests.
        model.vectors.flags.writeable = False
        model.norms.flags.writeable = False
        return cls(model)

    def missing(self, words: list[str]) -> list[str]:
        return [word for word in dict.fromkeys(words) if word not in self.model]

    def require(self, words: list[str]) -> None:
        if not words or any(not word.strip() for word in words):
            raise ValueError("请至少输入一个词语。")
        missing = self.missing(words)
        if missing:
            raise ValueError("这些词不在当前模型词表中：" + "、".join(missing) + "。请换词或使用预设示例。")

    def vector(self, word: str) -> np.ndarray:
        self.require([word])
        return self.model.get_vector(word)

    def nearest(self, query: np.ndarray, topn: int = 10, exclude: tuple[str, ...] = ()) -> list[tuple[str, float]]:
        if not 1 <= topn <= 50:
            raise ValueError("返回数量需在 1 到 50 之间。")
        query = np.asarray(query, dtype=np.float32)
        if query.shape != (self.model.vector_size,):
            raise ValueError("查询向量维度与模型不一致。")
        # Dot first, divide norms afterward: avoid allocating a full normalized
        # vocabulary matrix for every request.
        scores = (self.model.vectors @ unit(query)) / np.maximum(self.model.norms, 1e-10)
        scores[self.model.norms < 1e-10] = -np.inf
        for word in exclude:
            if word in self.model:
                scores[self.model.key_to_index[word]] = -np.inf
        count = min(topn, int(np.count_nonzero(np.isfinite(scores))))
        if count == 0:
            raise ValueError("没有可返回的候选词，请调整输入。")
        indices = np.argpartition(-scores, count - 1)[:count]
        indices = indices[np.argsort(-scores[indices], kind="stable")]
        return [(self.model.index_to_key[index], float(np.clip(scores[index], -1, 1))) for index in indices]

    def similar(self, word: str, topn: int = 10, opposite: bool = False) -> list[tuple[str, float]]:
        vector = self.vector(word)
        return self.nearest(-vector if opposite else vector, topn, (word,))

    def similarity(self, left: str, right: str) -> float:
        self.require([left, right])
        return float(np.clip(np.dot(unit(self.vector(left)), unit(self.vector(right))), -1, 1))

    def analogy(self, positive: list[str], negative: list[str], topn: int = 10) -> list[tuple[str, float]]:
        self.require(positive + negative)
        overlap = set(positive) & set(negative)
        if overlap:
            raise ValueError("同一个词不能同时出现在加项和减项中：" + "、".join(sorted(overlap)))
        vectors = [unit(self.vector(word)) for word in positive]
        vectors += [-unit(self.vector(word)) for word in negative]
        combined = np.mean(vectors, axis=0)
        return self.nearest(combined, topn, tuple(positive + negative))

    def outlier(self, words: list[str]) -> list[tuple[str, float]]:
        words = list(dict.fromkeys(words))
        if len(words) < 3:
            raise ValueError("找不同至少需要 3 个不同的词语。")
        self.require(words)
        normalized = np.array([unit(self.vector(word)) for word in words])
        center = unit(normalized.mean(axis=0))
        scores = normalized @ center
        # This matches Gensim doesnt_match: least similar to the normalized mean.
        return sorted(zip(words, map(float, scores)), key=lambda item: item[1])
