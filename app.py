from __future__ import annotations

from html import escape
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from threadpoolctl import threadpool_limits

from presets import ANALOGIES, NEIGHBORS, OUTLIERS
from notebook_space import LOCATIONS, PROJECTION_VERSION, location_vectors, reduce_dimensions, cluster_vectors
from model_config import model_path, model_files, model_identity
from vector_engine import VectorEngine

MODEL_PATH = model_path()
PAGES = ["相近词探索", "两词相似度", "找不同", "词语的加减法", "词向量空间"]
SPACE_COLORS = dict(zip([f"第{index}组" for index in range(1, 9)],
                       ["#087F8C", "#D9822B", "#7C59A4", "#C95070", "#357ABD", "#589636", "#8C564B", "#8E8D20"]))

st.set_page_config(page_title="中文词向量实验室", page_icon="🧭", layout="wide")
st.markdown("""
<style>
.block-container {max-width:1240px; padding-top:2rem; padding-bottom:2rem;}
[data-testid="stSidebar"] {border-right:1px solid #E2E8EE;}
.hero {background:#132F3E; border-radius:18px; padding:26px 32px; margin-bottom:22px; color:white;}
.hero .eyebrow {color:#77D6CC; letter-spacing:3px; font-size:12px; font-weight:600;}
.hero h1 {color:white; font-size:31px; padding:9px 0 5px; margin:0;}
.hero p {color:#CDDCE4; font-size:15px; margin:4px 0 0; line-height:1.8;}
[data-testid="stMetric"] {background:white; border:1px solid #E2E8EE; border-radius:12px; padding:14px 20px;}
[data-testid="stMetricValue"] {font-size:25px;}
[data-testid="stForm"] {background:white; border-color:#E2E8EE; border-radius:14px;}
[data-testid="stTextInputRootElement"]:has(input[aria-label="第一个词"]),
[data-testid="stTextInputRootElement"]:has(input[aria-label="第二个词"]) {
    border-color:#AABCC8; background-color:#F7F9FC;
}
[data-testid="stTextInputRootElement"]:has(input[aria-label="第一个词"]):focus-within,
[data-testid="stTextInputRootElement"]:has(input[aria-label="第二个词"]):focus-within {
    border-color:#087F8C; box-shadow:0 0 0 1px #087F8C;
}
.outlier-words {display:flex; flex-wrap:wrap; gap:12px; margin:4px 0 16px;}
.outlier-words span {
    font-size:26px; font-weight:700; line-height:1.5; color:#13545C;
    background:#E7F3F2; border:1px solid #C9E2DF; border-radius:10px; padding:8px 16px;
}
h2 {font-size:24px !important;}
h3 {font-size:19px !important;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="正在加载原 Notebook 的词向量，首次打开需要稍等片刻…")
def load_engine(path: str, *file_versions) -> VectorEngine:
    # File sizes and timestamps are cache-key inputs even though loading uses only the path.
    return VectorEngine.load(path)


@st.cache_data(show_spinner=False, max_entries=128)
def cached_neighbors(identity: tuple, word: str, topn: int, opposite: bool):
    with threadpool_limits(limits=1):
        return engine.similar(word, topn, opposite)


@st.cache_data(show_spinner="正在计算二维地名分布…", max_entries=4)
def cached_notebook_space(identity: tuple, projection_version: int):
    # Imported algorithm changes also need a cache-key version.
    coords = reduce_dimensions(cached_location_vectors(identity, projection_version))
    return pd.DataFrame({"词语": LOCATIONS, "X": coords[:, 0], "Y": coords[:, 1]})


@st.cache_data(show_spinner="正在计算三维地名分布…", max_entries=4)
def cached_space_3d(identity: tuple, projection_version: int):
    coords = reduce_dimensions(cached_location_vectors(identity, projection_version), num_dimensions=3)
    return pd.DataFrame({"词语": LOCATIONS, "X": coords[:, 0], "Y": coords[:, 1], "Z": coords[:, 2]})


@st.cache_data(show_spinner=False, max_entries=4)
def cached_location_vectors(identity: tuple, projection_version: int):
    return location_vectors(engine)


@st.cache_data(show_spinner=False, max_entries=4)
def cached_location_groups(identity: tuple, projection_version: int):
    return cluster_vectors(cached_location_vectors(identity, projection_version), num_groups=len(SPACE_COLORS))


def set_neighbor_word():
    st.session_state["neighbor_word"] = st.session_state["neighbor_preset"] or ""
    st.session_state.pop("neighbors_result", None)


def result(key: str, submitted: bool, label: str, operation, *, auto_run: bool = True):
    if submitted or (auto_run and key not in st.session_state):
        started = time.perf_counter()
        try:
            with threadpool_limits(limits=1):
                data = operation()
            st.session_state[key] = {"label": label, "data": data,
                                     "ms": (time.perf_counter() - started) * 1000}
        except ValueError as error:
            st.session_state[key] = {"label": label, "error": str(error)}
    if key not in st.session_state:
        return None
    saved = st.session_state[key]
    if "error" in saved:
        st.warning(saved["error"])
        return None
    st.caption(f"本次结果 · {saved['label']}　｜　处理耗时 {saved['ms']:.0f} ms（含缓存命中）")
    return saved["data"]


def ranking_chart(rows: list[tuple[str, float]], score_label: str = "余弦相似度"):
    frame = pd.DataFrame(rows, columns=["词语", score_label])
    figure = go.Figure(go.Bar(
        x=frame[score_label], y=frame["词语"], orientation="h",
        marker_color=["#087F8C"] + ["#A6DAD7"] * (len(rows) - 1),
        text=[f"{score:.4f}" for score in frame[score_label]], textposition="auto",
        hovertemplate="%{y}<br>得分 %{x:.4f}<extra></extra>",
    ))
    figure.update_layout(height=max(300, len(rows) * 30 + 65), margin=dict(l=10, r=30, t=15, b=30),
                         yaxis=dict(autorange="reversed"), xaxis_title=score_label,
                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(figure, width="stretch")


def vector_chart(values: np.ndarray):
    frame = pd.DataFrame({"维度": np.arange(1, len(values) + 1), "数值": values})
    figure = px.bar(frame, x="维度", y="数值", color="数值",
                    color_continuous_scale="Tealrose", color_continuous_midpoint=0)
    figure.update_traces(hovertemplate="维度 %{x}<br>数值 %{y:.6f}<extra></extra>")
    figure.update_layout(height=330, showlegend=False, coloraxis_showscale=False,
                         margin=dict(l=10, r=10, t=15, b=20), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(figure, width="stretch")


with st.sidebar:
    st.markdown("### 🧭 中文词向量实验室")
    st.caption("把抽象的向量，变成可探索的关系")
    page = st.radio("选择实验", PAGES, label_visibility="collapsed", key="page")

st.markdown("""
<div class="hero"><div class="eyebrow">WORD VECTOR LAB / 中文语义探索</div>
<h1>探索词语之间的关系</h1>
<p>寻找相近词、比较语义距离，让词语做加减法，再到二维空间里观察它们。</p></div>
""", unsafe_allow_html=True)

missing_files = [file.name for file in model_files(MODEL_PATH) if not file.is_file()]
if missing_files:
    st.error("缺少原模型配套文件：" + "、".join(missing_files))
    st.write("请将 Tencent_AILab_ChineseEmbedding.bin 和 Tencent_AILab_ChineseEmbedding.bin.vectors.npy 放在同一文件夹。")
    st.write("可放入项目的 data 文件夹，或通过 WORD_VECTOR_MODEL_PATH 指定 .bin 的完整路径。")
    st.caption("当前读取位置：" + str(MODEL_PATH))
    st.stop()

identity = model_identity(MODEL_PATH)
if st.session_state.get("_active_model_identity") != identity:
    for state_key in list(st.session_state):
        if state_key.endswith("_result") or state_key.startswith(("outlier_result_", "analogy_result_")):
            del st.session_state[state_key]
    st.session_state["_active_model_identity"] = identity
if st.session_state.get("_active_projection_version") != PROJECTION_VERSION:
    for state_key in ("space_visualization_result", "space_3d_result"):
        st.session_state.pop(state_key, None)
    st.session_state["_active_projection_version"] = PROJECTION_VERSION
try:
    engine = load_engine(*identity)
except (OSError, ValueError, EOFError, UnicodeError):
    st.error("原模型无法读取。请确认两个文件完整、来自同一次保存，且 .npy 文件名与 .bin 配套。")
    st.stop()

with st.sidebar:
    st.divider()
    st.metric("可查询词语", f"{len(engine.model):,}")
    st.metric("向量维度", str(engine.model.vector_size))

if page == "相近词探索":
    st.header("相近词探索")
    st.write("输入一个词，看看哪些词与它拥有接近的向量方向。")
    examples = [word for word in NEIGHBORS if word in engine.model]
    st.selectbox("示例或自定义词", examples, key="neighbor_preset", accept_new_options=True,
                 on_change=set_neighbor_word,
                 help="可选择示例，也可输入新词后按回车确认；下方查询框同样支持自由输入。")
    st.session_state.setdefault("neighbor_word", examples[0] if examples else "")
    with st.form("neighbors_form"):
        left, right = st.columns([3, 2])
        word = left.text_input("查询词语（可自由输入）", key="neighbor_word", max_chars=80).strip()
        topn = right.slider("返回数量", 5, 30, 10, key="neighbor_topn")
        opposite = st.checkbox("探索相反方向", help="向量相反方向不等于语言学上的反义词。")
        submitted = st.form_submit_button("开始探索", type="primary")
    def explore_word():
        return {"word": word, "vector": engine.vector(word).copy(),
                "rows": cached_neighbors(identity, word, topn, opposite), "opposite": opposite}
    data = result("neighbors_result", submitted, f"{word} · {'相反方向' if opposite else '相近方向'} · Top {topn}",
                  explore_word, auto_run=False)
    if data is not None:
        st.subheader("相近词结果" if not data["opposite"] else "相反方向的候选词")
        ranking_chart(data["rows"])
        if data["opposite"]:
            st.caption("相反方向模式的得分，是候选词与输入词负向量的余弦相似度。")
        values = data["vector"]
        st.subheader(f"{data['word']} 的词向量 · {len(values)} 维")
        vector_chart(values)
        st.code(np.array2string(values, precision=8, separator=", ", threshold=len(values),
                               max_line_width=56), language="json", wrap_lines=True)

elif page == "两词相似度":
    st.header("两词相似度")
    st.write("用余弦相似度比较两个词。接近 1 表示向量方向更接近。")
    st.session_state.setdefault("pair_left", "宋江")
    st.session_state.setdefault("pair_right", "武松")
    with st.form("pair_form"):
        left, right = st.columns(2)
        first = left.text_input("第一个词", key="pair_left", max_chars=80).strip()
        second = right.text_input("第二个词", key="pair_right", max_chars=80).strip()
        submitted = st.form_submit_button("比较相似度", type="primary")
    score = result("pair_result", submitted, f"{first} ↔ {second}", lambda: engine.similarity(first, second))
    if score is not None:
        figure = go.Figure(go.Indicator(mode="gauge+number", value=score,
            number={"valueformat": ".4f"}, gauge={"axis": {"range": [-1, 1]}, "bar": {"color": "#087F8C"},
                "steps": [{"range": [-1, 0], "color": "#E8EDF2"}, {"range": [0, 1], "color": "#D7EEEB"}]}))
        figure.update_layout(height=280, margin=dict(l=45, r=45, t=25, b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(figure, width="stretch")
        st.info("这是语料中学到的关联程度，不代表事实判断，也不等于百分比概率。")

elif page == "找不同":
    st.header("找不同")
    st.write("找出与整组平均向量方向最不接近的那个词。")
    for index, words in enumerate(OUTLIERS.values()):
        with st.container(border=True):
            st.subheader(f"第 {index + 1} 题")
            word_labels = "".join(f"<span>{escape(word)}</span>" for word in words)
            st.markdown(f'<div class="outlier-words">{word_labels}</div>', unsafe_allow_html=True)
            submitted = st.button("找出不同的词", key=f"outlier_submit_{index}", type="primary")
            rows = result(f"outlier_result_{index}", submitted, "、".join(words),
                          lambda words=words: engine.outlier(words), auto_run=False)
            if rows:
                st.metric("模型选出的不同词", rows[0][0])
                st.caption("下面的分数越低，表示该词与组内平均方向越不相近。结果取决于语料，并非唯一正确答案。")
                ranking_chart(rows, "与组内平均向量的余弦相似度")

elif page == "词语的加减法":
    st.header("词语的加减法")
    st.write("将多个词向量相加或相减，找到与计算结果最接近的词。")
    for index, (formula, (positive, negative)) in enumerate(ANALOGIES.items()):
        with st.container(border=True):
            st.subheader(formula)
            submitted = st.button("计算结果", key=f"analogy_submit_{index}", type="primary")
            rows = result(f"analogy_result_{index}", submitted, formula,
                          lambda positive=positive, negative=negative: engine.analogy(positive, negative, 1),
                          auto_run=False)
            if rows:
                st.metric("最接近的词", rows[0][0])

elif page == "词向量空间":
    st.header("词向量空间")
    st.write(f"{len(LOCATIONS)} 个地名，包含城市、省份、区县等")
    with st.container(border=True):
        st.write("、".join(LOCATIONS))
    submitted = st.button("二维可视化", key="space_visualize", type="primary")
    submitted_3d = st.button("三维可视化", key="space_visualize_3d", type="primary")
    if submitted or submitted_3d or "space_visualization_result" in st.session_state or "space_3d_result" in st.session_state:
        st.caption("颜色按原始100维词向量归一化后分为8组，用于辅助观察；二维、三维使用相同分组和配色。")
    data = result("space_visualization_result", submitted, "二维可视化 · t-SNE",
                  lambda: cached_notebook_space(identity, PROJECTION_VERSION), auto_run=False)
    if data is not None:
        frame = data.assign(分组=cached_location_groups(identity, PROJECTION_VERSION))
        st.subheader("二维可视化结果")
        st.caption("拖动平移，滚轮缩放，悬停查看地名及二维坐标。")
        figure = px.scatter(frame, x="X", y="Y", text="词语", hover_name="词语", color="分组",
                            color_discrete_map=SPACE_COLORS, category_orders={"分组": list(SPACE_COLORS)}, custom_data=["分组"])
        figure.update_traces(marker=dict(size=7),
                             textfont=dict(size=11, color="black"), textposition="top right",
                             hovertemplate="<b>%{hovertext}</b><br>%{customdata[0]}<br>二维坐标<br>X：%{x:.4f}<br>Y：%{y:.4f}<extra></extra>")
        figure.update_layout(height=680, margin=dict(l=35, r=20, t=20, b=35),
                             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="white",
                             xaxis=dict(title="投影坐标 1", showgrid=False, zeroline=False,
                                        showline=True, mirror=True),
                             yaxis=dict(title="投影坐标 2", showgrid=False, zeroline=False,
                                        showline=True, mirror=True, scaleanchor="x", scaleratio=1),
                             legend=dict(orientation="h", title_text="向量分组", y=1.02, yanchor="bottom"),
                             dragmode="pan", uirevision=f"space-2d-{PROJECTION_VERSION}")
        st.plotly_chart(figure, width="stretch", config={"scrollZoom": True}, key="space_2d_chart")
    frame = result("space_3d_result", submitted_3d, "三维可视化 · t-SNE",
                   lambda: cached_space_3d(identity, PROJECTION_VERSION), auto_run=False)
    if frame is not None:
        frame = frame.assign(分组=cached_location_groups(identity, PROJECTION_VERSION))
        st.subheader("三维可视化结果")
        st.caption("拖动旋转，滚轮缩放，悬停查看地名及三维坐标。")
        figure = px.scatter_3d(frame, x="X", y="Y", z="Z", text="词语", hover_name="词语", color="分组",
                               color_discrete_map=SPACE_COLORS, category_orders={"分组": list(SPACE_COLORS)}, custom_data=["分组"])
        figure.update_traces(marker=dict(size=4),
                             textfont=dict(size=11, color="black"), textposition="top center",
                             hovertemplate="<b>%{hovertext}</b><br>%{customdata[0]}<br>三维坐标<br>X：%{x:.4f}<br>Y：%{y:.4f}<br>Z：%{z:.4f}<extra></extra>")
        figure.update_layout(height=680, margin=dict(l=0, r=0, t=10, b=0),
                             paper_bgcolor="rgba(0,0,0,0)",
                             scene=dict(xaxis_title="投影坐标 1", yaxis_title="投影坐标 2",
                                        zaxis_title="投影坐标 3", aspectmode="data"),
                             legend=dict(orientation="h", title_text="向量分组", y=1.02, yanchor="bottom"),
                             uirevision=f"space-3d-{PROJECTION_VERSION}")
        st.plotly_chart(figure, width="stretch", config={"scrollZoom": True}, key="space_3d_chart")

