import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st


# 获取当前脚本所在的绝对路径（即仓库根目录）
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# 【已修正】直接指向根目录下的模型文件，去掉了不存在的 modeling_results 文件夹
MODEL_PACKAGE_PATH = os.path.join(PROJECT_DIR, "capacitance_model_package.pkl")


@st.cache_resource
def load_model_package():
    """加载模型包，使用缓存避免重复加载"""
    # 增加一个检查，如果文件不存在会提示具体路径，方便排查
    if not os.path.exists(MODEL_PACKAGE_PATH):
        st.error(f"找不到模型文件！请检查文件名是否正确。\n当前查找路径: {MODEL_PACKAGE_PATH}")
        return None
    
    with open(MODEL_PACKAGE_PATH, "rb") as f:
        return pickle.load(f)


def predict_capacity(values, package):
    """执行预测逻辑"""
    raw = np.asarray(values, dtype=float).reshape(1, -1)
    scaled = package["scaler"].transform(raw)
    scaled_df = pd.DataFrame(
        scaled,
        columns=[f"Feature_{i}" for i in range(len(package["feature_names"]))],
    )
    return float(np.asarray(package["model"].predict(scaled_df)).ravel()[0])


def read_r2(row):
    """兼容不同的 R2 键名写法"""
    return row.get("R2", row.get("R²", row.get("R虏", 0)))


# --- 页面配置 ---
st.set_page_config(
    page_title="Capacitance Prediction Platform",
    page_icon="⚡",  # 建议用个图标，比 "C" 好看点
    layout="wide",
)

# 加载模型
package = load_model_package()

# 如果模型加载失败（比如文件没找到），就不渲染后面的界面了，防止报错
if package is None:
    st.stop()

# --- CSS 样式美化 ---
st.markdown(
    """
    <style>
    :root {
        --ink: #152033;
        --muted: #667085;
        --line: #d8e0ec;
        --blue: #1f5fbf;
        --cyan: #0ea5b7;
        --green: #0f8f6f;
        --violet: #6d5bd0;
        --amber: #d88626;
    }
    .stApp {
        background:
            radial-gradient(circle at 8% 4%, rgba(31,95,191,0.20), transparent 28%),
            radial-gradient(circle at 88% 10%, rgba(15,143,111,0.16), transparent 28%),
            linear-gradient(180deg, #f8fbff 0%, #edf4fb 100%);
        color: var(--ink);
    }
    [data-testid="stHeader"] {
        background: rgba(248, 251, 255, 0.72);
        backdrop-filter: blur(8px);
    }
    .block-container {
        padding-top: 1.35rem;
        padding-bottom: 2.5rem;
        max-width: 1380px;
    }
    .hero {
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(216,224,236,0.95);
        border-radius: 14px;
        padding: 26px 30px;
        margin-bottom: 22px;
        color: white;
        background:
            linear-gradient(135deg, rgba(12,42,91,0.95), rgba(31,95,191,0.92) 52%, rgba(14,165,183,0.86));
        box-shadow: 0 18px 42px rgba(22, 45, 90, 0.22);
    }
    .hero:after {
        content: "";
        position: absolute;
        width: 420px;
        height: 420px;
        right: -160px;
        top: -190px;
        border-radius: 50%;
        background: rgba(255,255,255,0.16);
    }
    .hero-title {
        position: relative;
        font-size: 34px;
        line-height: 1.12;
        font-weight: 900;
        margin-bottom: 8px;
        letter-spacing: 0;
    }
    .hero-subtitle {
        position: relative;
        font-size: 15px;
        color: rgba(255,255,255,0.86);
        max-width: 980px;
    }
    .pill-row {
        position: relative;
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 18px;
    }
    .pill {
        border: 1px solid rgba(255,255,255,0.25);
        background: rgba(255,255,255,0.13);
        border-radius: 999px;
        padding: 7px 12px;
        font-size: 13px;
        font-weight: 700;
    }
    .panel {
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 18px 18px 16px;
        background: rgba(255,255,255,0.90);
        box-shadow: 0 14px 34px rgba(32, 55, 100, 0.10);
        margin-bottom: 16px;
    }
    .panel-title {
        font-size: 17px;
        font-weight: 850;
        color: #1f2a44;
        margin-bottom: 4px;
    }
    .panel-note {
        color: var(--muted);
        font-size: 13px;
        margin-bottom: 14px;
    }
    div[data-testid="stMetric"] {
        background:
            linear-gradient(180deg, #ffffff 0%, #f5f9ff 100%);
        border: 1px solid #cfd9e8;
        padding: 1.05rem 1.05rem 0.85rem;
        border-radius: 12px;
        box-shadow: 0 14px 30px rgba(22, 45, 90, 0.12);
    }
    div[data-testid="stMetricValue"] {
        color: #0b3d78;
        font-size: 2.25rem;
        font-weight: 900;
    }
    div[data-testid="stNumberInput"] label {
        font-weight: 750;
        color: #22314d;
    }
    div[data-baseweb="input"] {
        border-radius: 8px;
    }
    .stButton > button {
        height: 46px;
        border-radius: 9px;
        font-weight: 850;
        letter-spacing: 0;
        box-shadow: 0 12px 24px rgba(31, 95, 191, 0.20);
    }
    .small-card {
        border: 1px solid #d9e1ee;
        background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
        border-radius: 10px;
        padding: 13px 14px;
        margin-bottom: 10px;
    }
    .small-card b {
        color: #172033;
    }
    .small-card span {
        color: #667085;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- 头部展示区 ---
st.markdown(
    f"""
    <div class="hero">
      <div class="hero-title">Capacitance Prediction Platform</div>
      <div class="hero-subtitle">
        Enter material descriptors and discharge condition, then run the trained model to predict
        {package['target_name']}.
      </div>
      <div class="pill-row">
        <div class="pill">Trained model: {package['best_model_name']}</div>
        <div class="pill">Feature count: {len(package['feature_names'])}</div>
        <div class="pill">Target range: {package['target_min']:.3f} - {package['target_max']:.3f}</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- 主布局 ---
left, right = st.columns([2.15, 1], gap="large")

with left:
    st.markdown(
        """
        <div class="panel">
          <div class="panel-title">Input Features</div>
          <div class="panel-note">Defaults are filled with training-set means. Hover each field to see the training range.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    values = []
    tabs = st.tabs(["Pore and Defect", "Composition", "Dopant and Test"])
    
    # 根据特征数量动态分组，防止索引越界
    total_features = len(package["feature_stats"])
    groups = [
        package["feature_stats"][:4],
        package["feature_stats"][4:9],
        package["feature_stats"][9:],
    ]
    start_indices = [0, 4, 9]

    for tab, group, start_idx in zip(tabs, groups, start_indices):
        with tab:
            cols = st.columns(2)
            for local_idx, item in enumerate(group):
                idx = start_idx + local_idx
                # 安全检查：确保索引没有超出总特征数
                if idx < total_features:
                    with cols[local_idx % 2]:
                        values.append(
                            st.number_input(
                                label=f"{idx + 1}. {item['name']}",
                                value=float(item["mean"]),
                                format="%.6f",
                                help=f"Training range: {item['min']:.6g} - {item['max']:.6g}",
                                key=f"feature_{idx}",
                            )
                        )

    # 重新排序输入值以匹配模型要求的特征顺序
    ordered_values = [None] * len(package["feature_names"])
    cursor = 0
    for group_start, group in zip(start_indices, groups):
        for local_idx, _ in enumerate(group):
            if group_start + local_idx < len(ordered_values):
                ordered_values[group_start + local_idx] = values[cursor]
                cursor += 1

    predict_clicked = st.button("Predict Capacity", type="primary", use_container_width=True)

with right:
    st.markdown(
        """
        <div class="panel">
          <div class="panel-title">Prediction Output</div>
          <div class="panel-note">The value below is produced by the saved trained model package.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if predict_clicked:
        try:
            prediction = predict_capacity(ordered_values, package)
            st.metric(package["target_name"], f"{prediction:.4f}")
            st.success(f"Prediction completed with trained model: {package['best_model_name']}")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
    else:
        st.metric(package["target_name"], "--")
        st.info("Input features and click Predict Capacity.")

    st.markdown(
        f"""
        <div class="small-card"><b>Model package</b><br><span>capacitance_model_package.pkl</span></div>
        <div class="small-card"><b>Target mean in training data</b><br><span>{package['target_mean']:.4f}</span></div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Test Performance")
    perf_rows = []
    for row in package.get("test_performance", []):
        perf_rows.append(
            {
                "Model": row.get("Model", ""),
                "R2": round(float(read_r2(row)), 4),
                "RMSE": round(float(row.get("RMSE", 0)), 3),
                "MAE": round(float(row.get("MAE", 0)), 3),
            }
        )
    if perf_rows:
        st.dataframe(pd.DataFrame(perf_rows), hide_index=True, use_container_width=True)
    else:
        st.caption("No performance table stored in model package.")
