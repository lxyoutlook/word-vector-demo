from pathlib import Path
import json
import sys

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vector_engine import VectorEngine
from model_config import model_path, model_files, model_identity
from notebook_space import LOCATIONS, PROJECTION_VERSION, location_vectors, reduce_dimensions, cluster_vectors


@pytest.fixture(scope="module")
def real():
    return VectorEngine.load(model_path())


def test_model_integrity_and_shape(real):
    assert all(file.is_file() for file in model_files(model_path()))
    assert real.model.vectors.shape == (2000000, 100)
    assert isinstance(real.model.vectors, np.memmap)
    assert not real.model.vectors.flags.writeable
    assert not real.missing(["宋江", "武松", "刘备", "毽子", "文具盒", "不倒翁"])


def test_missing_word(real):
    with pytest.raises(ValueError, match="不在当前模型词表"):
        real.similar("这是不存在于词表中的测试词语XYZ")


def test_search_matches_gensim(real):
    expected = real.model.most_similar("人工智能", topn=10)
    actual = real.similar("人工智能", 10)
    assert [word for word, _ in actual] == [word for word, _ in expected]
    np.testing.assert_allclose([score for _, score in actual], [score for _, score in expected], atol=1e-6)
    assert "人工智能" not in [word for word, _ in actual]


def test_analogy_matches_gensim(real):
    expected = real.model.most_similar(positive=["国王", "女人"], negative=["男人"], topn=10)
    actual = real.analogy(["国王", "女人"], ["男人"], 10)
    assert [word for word, _ in actual] == [word for word, _ in expected]
    np.testing.assert_allclose([score for _, score in actual], [score for _, score in expected], atol=1e-6)


def test_pair_and_outlier_match_reference(real):
    assert real.similarity("北京", "北京") == pytest.approx(1, abs=1e-6)
    assert real.similarity("北京", "上海") == pytest.approx(real.model.similarity("北京", "上海"), abs=1e-6)
    words = ["博物馆", "展览馆", "科技馆", "超市"]
    assert real.outlier(words)[0][0] == real.model.doesnt_match(words) == "超市"


def test_invalid_combinations_and_zero_vectors(real):
    with pytest.raises(ValueError, match="同时"):
        real.analogy(["国王"], ["国王"])
    with pytest.raises(ValueError, match="至少需要 3"):
        real.outlier(["北京", "北京", "上海"])
    with pytest.raises(ValueError, match="接近零"):
        real.nearest(np.zeros(real.model.vector_size, dtype=np.float32))
    with pytest.raises(ValueError, match="维度"):
        real.nearest(np.zeros(real.model.vector_size + 1, dtype=np.float32))


@pytest.mark.parametrize("dimensions", [2, 3])
def test_notebook_projection_reproducible(real, dimensions):
    from sklearn.manifold import TSNE, trustworthiness
    from sklearn.metrics import pairwise_distances
    from threadpoolctl import threadpool_limits

    vectors = location_vectors(real)
    unchanged = vectors.copy()
    first = reduce_dimensions(vectors, num_dimensions=dimensions)
    second = reduce_dimensions(vectors, num_dimensions=dimensions)
    assert first.shape == (135, dimensions) and np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(vectors, unchanged)
    # Independent behavioral check: keep cosine-nearest neighbors across ALL
    # locations, rather than making a selected city look geographically right.
    with threadpool_limits(limits=1):
        baseline = TSNE(n_components=dimensions, random_state=0).fit_transform(vectors)
        distances = pairwise_distances(vectors, metric="cosine")
        np.fill_diagonal(distances, np.inf)
        original = distances.argsort(axis=1)
        def recalls(coords):
            distances = pairwise_distances(coords)
            np.fill_diagonal(distances, np.inf)
            order = distances.argsort(axis=1)
            return [np.mean([len(set(a[:k]) & set(b[:k])) / k for a, b in zip(original, order)])
                    for k in (5, 10, 15)]
        before, after = recalls(baseline), recalls(first)
        assert all(new >= old for old, new in zip(before, after))
        assert np.mean(after) - np.mean(before) > (0.005 if dimensions == 2 else 0.20)
        for k in (5, 10, 15):
            assert trustworthiness(vectors, first, n_neighbors=k, metric="cosine") >= trustworthiness(
                vectors, baseline, n_neighbors=k, metric="cosine")


def test_streamlit_pages_and_user_inputs():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=40).run()
    assert not at.exception
    assert [(metric.label, metric.value) for metric in at.sidebar.metric] == [
        ("可查询词语", "2,000,000"), ("向量维度", "100")]
    assert not at.main.metric
    assert at.text_input(key="neighbor_word").value == "宋江"
    assert "neighbors_result" not in at.session_state
    assert not any("本次结果" in caption.value for caption in at.caption)
    assert not any("本地运行" in element.value for element in at.sidebar.markdown)
    assert not at.sidebar.expander
    at.text_input(key="neighbor_word").set_value("杭州")
    at.button[0].click().run()
    assert not at.exception
    data = at.session_state["neighbors_result"]["data"]
    assert data["word"] == "杭州" and len(data["vector"]) == 100 and data["rows"]
    assert "..." not in at.code[0].value
    at.text_input(key="neighbor_word").set_value("这是不存在于词表中的测试词语XYZ")
    at.button[0].click().run()
    assert not at.exception and len(at.warning) == 1
    assert "error" in at.session_state["neighbors_result"]
    # Preset callback must replace the submitted input and clear stale errors.
    at.selectbox(key="neighbor_preset").select("北京").run()
    assert not at.exception and not at.warning
    assert "neighbors_result" not in at.session_state
    at.button[0].click().run()
    assert at.session_state["neighbors_result"]["label"].startswith("北京")
    at.radio(key="page").set_value("两词相似度").run()
    assert not at.exception and at.session_state["pair_result"]["data"] > 0
    at.radio(key="page").set_value("找不同").run()
    assert not at.exception and len(at.button) == 4
    assert not at.selectbox and not at.text_area
    assert not at.main.metric and not at.get("plotly_chart")
    assert not any("本次结果" in caption.value for caption in at.caption)
    for index in range(4):
        assert f"outlier_result_{index}" not in at.session_state
        at.button(key=f"outlier_submit_{index}").click().run()
        assert not at.exception and not at.warning
        assert len(at.main.metric) == index + 1
        assert len(at.get("plotly_chart")) == index + 1
        assert len(at.session_state[f"outlier_result_{index}"]["data"]) == 4
    assert at.session_state["outlier_result_0"]["data"][0][0] == "超市"
    at.radio(key="page").set_value("词语的加减法").run()
    assert not at.exception and len(at.button) == 2
    assert not at.selectbox and not at.text_input and not at.main.metric
    assert not at.slider and not at.get("plotly_chart") and not at.info
    assert [heading.value for heading in at.subheader] == ["国王 − 男人 ＋ 女人", "父亲 − 男人 ＋ 女人"]
    for index in range(2):
        assert f"analogy_result_{index}" not in at.session_state
        assert at.button(key=f"analogy_submit_{index}").label == "计算结果"
        at.button(key=f"analogy_submit_{index}").click().run()
        assert not at.exception and len(at.session_state[f"analogy_result_{index}"]["data"]) == 1
        assert len(at.main.metric) == index + 1
        assert at.main.metric[index].value == at.session_state[f"analogy_result_{index}"]["data"][0][0]
    assert at.main.metric[0].value == "王后"
    at.radio(key="page").set_value("词向量空间").run()
    assert not at.exception
    assert "向量查询" not in at.radio(key="page").options
    names = next(element.value for element in at.markdown if element.value.startswith("新余市、梧州市"))
    assert names.split("、") == list(LOCATIONS)
    assert not at.get("html")
    assert not at.multiselect and not at.selectbox and not at.text_area and not at.slider
    assert not at.image and not at.expander
    assert len(at.button) == 2 and at.button(key="space_visualize").label == "二维可视化"
    assert at.button(key="space_visualize_3d").label == "三维可视化"
    assert not at.get("plotly_chart")
    assert "space_visualization_result" not in at.session_state
    assert "space_3d_result" not in at.session_state
    at.button(key="space_visualize").click().run()
    assert not at.exception and not at.image and len(at.get("plotly_chart")) == 1
    city_frame = at.session_state["space_visualization_result"]["data"]
    assert city_frame["词语"].tolist() == list(LOCATIONS)
    assert len(city_frame) == 135 and np.isfinite(city_frame[["X", "Y"]]).all().all()
    assert {"南京市", "澳门"} <= set(city_frame["词语"]) and "未知位置" not in set(city_frame["词语"])
    spec_2d = json.loads(at.get("plotly_chart")[0].proto.spec)
    assert spec_2d["data"][0]["type"] == "scatter"
    assert spec_2d["layout"]["yaxis"]["scaleanchor"] == "x"
    assert spec_2d["layout"]["yaxis"]["scaleratio"] == 1
    hover_2d = spec_2d["data"][0]["hovertemplate"]
    assert all(f"%{{{axis}:.4f}}" in hover_2d for axis in ("x", "y"))
    assert "%{hovertext}" in hover_2d and "%{z" not in hover_2d
    at.button(key="space_visualize_3d").click().run()
    assert not at.exception and not at.image and len(at.get("plotly_chart")) == 2
    space_3d = at.session_state["space_3d_result"]["data"]
    assert space_3d["词语"].tolist() == list(LOCATIONS)
    assert np.isfinite(space_3d[["X", "Y", "Z"]]).all().all()
    spec = json.loads(at.get("plotly_chart")[1].proto.spec)
    assert all(trace["type"] == "scatter3d" for trace in spec["data"])
    assert len(spec["data"]) == len(spec_2d["data"]) == 8
    colors_2d = {trace["name"]: trace["marker"]["color"] for trace in spec_2d["data"]}
    assert colors_2d == {trace["name"]: trace["marker"]["color"] for trace in spec["data"]}
    groups_2d = {word: trace["name"] for trace in spec_2d["data"] for word in trace["hovertext"]}
    groups_3d = {word: trace["name"] for trace in spec["data"] for word in trace["hovertext"]}
    assert groups_2d == groups_3d and set(groups_2d) == set(LOCATIONS)
    hover = spec["data"][0]["hovertemplate"]
    assert all(f"%{{{axis}:.4f}}" in hover for axis in ("x", "y", "z"))
    assert "%{hovertext}" in hover
    at.button(key="space_visualize").click().run()
    assert not at.exception and not at.image and len(at.get("plotly_chart")) == 2
    np.testing.assert_array_equal(at.session_state["space_3d_result"]["data"][["X", "Y", "Z"]], space_3d[["X", "Y", "Z"]])
    at.radio(key="page").set_value("相近词探索").run()
    at.radio(key="page").set_value("词向量空间").run()
    assert not at.exception and not at.image and len(at.get("plotly_chart")) == 2


def test_free_word_query_vector_and_empty_input(real):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception
    # Custom dropdown entries are verified in the browser: AppTest's selectbox
    # serializer only accepts original options, even with accept_new_options.
    at.selectbox(key="neighbor_preset").select("咖啡").run()
    assert not at.exception and at.text_input(key="neighbor_word").value == "咖啡"
    assert "neighbors_result" not in at.session_state
    at.button[0].click().run()
    data = at.session_state["neighbors_result"]["data"]
    assert data["word"] == "咖啡"
    np.testing.assert_array_equal(data["vector"], real.vector("咖啡"))
    # Replacing the text query must not be overridden by the selector value.
    at.text_input(key="neighbor_word").set_value("杭州")
    at.button[0].click().run()
    assert not at.exception
    data = at.session_state["neighbors_result"]["data"]
    assert data["word"] == "杭州"
    np.testing.assert_array_equal(data["vector"], real.vector("杭州"))
    at.text_input(key="neighbor_word").set_value("   ")
    at.button[0].click().run()
    assert not at.exception and at.warning
    assert not at.code


def test_missing_model_has_actionable_ui(monkeypatch):
    monkeypatch.setenv("WORD_VECTOR_MODEL_PATH", str(ROOT / "data" / "missing-test-file.bin"))
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not at.exception
    assert len(at.error) == 1
    assert "缺少" in at.error[0].value


def test_companion_file_is_required(monkeypatch):
    # An existing .bin on its own must not silently load a different resource.
    monkeypatch.setenv("WORD_VECTOR_MODEL_PATH", str(ROOT / "app.py"))
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not at.exception and len(at.error) == 1
    assert "app.py.vectors.npy" in at.error[0].value
    with pytest.raises(FileNotFoundError, match="配套文件"):
        VectorEngine.load(ROOT / "app.py")


def test_cache_identity_tracks_both_files(tmp_path):
    # These tiny files only exercise cache invalidation, never model loading.
    binary = tmp_path / "model.bin"
    array = tmp_path / "model.bin.vectors.npy"
    binary.write_bytes(b"metadata")
    array.write_bytes(b"one")
    before = model_identity(binary)
    array.write_bytes(b"changed-array")
    assert model_identity(binary) != before


def test_model_change_clears_all_page_results(monkeypatch, real):
    # Simulate changed file metadata without replacing or reloading the large model.
    monkeypatch.setattr(VectorEngine, "load", classmethod(lambda cls, path: real))
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=40).run()
    assert not at.exception
    keys = ["neighbors_result", "pair_result", "space_visualization_result", "space_3d_result"]
    keys += [f"outlier_result_{index}" for index in range(4)]
    keys += [f"analogy_result_{index}" for index in range(2)]
    for key in keys:
        at.session_state[key] = {"label": "旧模型", "data": "旧结果"}
    identity = at.session_state["_active_model_identity"]
    changed = (*identity[:-1], identity[-1] + 1)
    monkeypatch.setattr("model_config.model_identity", lambda path: changed)
    at.run()
    assert not at.exception
    assert at.session_state["_active_model_identity"] == changed
    assert all(key not in at.session_state for key in keys)


def test_locations_use_exact_model_vectors(real):
    assert len(LOCATIONS) == len(set(LOCATIONS)) == 135
    assert LOCATIONS[0] == "新余市" and LOCATIONS[-1] == "南京市"
    assert LOCATIONS.count("南京市") == LOCATIONS.count("澳门") == 1
    assert "未知位置" not in LOCATIONS
    vectors = location_vectors(real)
    assert vectors.shape == (135, 100)
    assert not real.missing(list(LOCATIONS))
    for index, word in enumerate(LOCATIONS):
        np.testing.assert_array_equal(vectors[index], real.vector(word))
    coords = reduce_dimensions(vectors)
    assert coords.shape == (135, 2) and np.isfinite(coords).all()
    coords_3d = reduce_dimensions(vectors, num_dimensions=3)
    assert coords_3d.shape == (135, 3) and np.isfinite(coords_3d).all()
    with pytest.raises(ValueError, match="二维或三维"):
        reduce_dimensions(vectors, num_dimensions=4)


def test_cluster_vectors_use_original_directions(real):
    vectors = location_vectors(real)
    groups = cluster_vectors(vectors)
    assert len(groups) == 135 and len(set(groups)) == 8
    assert groups == cluster_vectors(vectors)
    # Length changes must not change direction-based grouping.
    np.testing.assert_array_equal(groups, cluster_vectors(vectors * 2))
    sample = np.array([[1, 0], [2, 0], [0, 1], [0, 3]], dtype="float32")
    assigned = cluster_vectors(sample, num_groups=2)
    assert assigned[0] == assigned[1] and assigned[2] == assigned[3] and assigned[0] != assigned[2]
    with pytest.raises(ValueError, match="零向量"):
        cluster_vectors(np.zeros((3, 2)), num_groups=2)


def test_projection_rejects_invalid_vectors():
    for vectors in (np.ones(4), np.ones((3, 10)), np.ones((4, 0)), np.full((4, 10), np.nan)):
        with pytest.raises(ValueError, match="有效"):
            reduce_dimensions(vectors)
    with pytest.raises(ValueError, match="零向量"):
        reduce_dimensions(np.zeros((4, 10)))
    # The perplexity adapts safely for a small valid set.
    for dimensions in (2, 3):
        assert np.isfinite(reduce_dimensions(np.eye(4), dimensions)).all()


def test_projection_update_clears_only_projection_results():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=40).run()
    at.session_state["space_visualization_result"] = {"data": "旧二维结果"}
    at.session_state["space_3d_result"] = {"data": "旧三维结果"}
    at.session_state["_active_projection_version"] = PROJECTION_VERSION - 1
    at.session_state["outlier_result_0"] = {"data": "另一页的结果"}
    at.run()
    assert not at.exception
    assert "space_visualization_result" not in at.session_state and "space_3d_result" not in at.session_state
    assert at.session_state["_active_projection_version"] == PROJECTION_VERSION
    assert at.session_state["outlier_result_0"]["data"] == "另一页的结果"
