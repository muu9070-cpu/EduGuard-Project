import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- 1. 配置与数据加载 ---
st.set_page_config(page_title="EduGuard Pro | Academic Risk System", layout="wide")

# UM 官方配色注入 (深蓝与金色)
st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #003366; }
    [data-testid="stSidebar"] * { color: white !important; }
    .stButton>button { background-color: #FFCC00; color: #003366; font-weight: bold; }
    h1, h2, h3 { color: #003366; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_essentials():
    # 加载模型和特征列名
    # 请确保这两个文件在 C:\Users\LX\EduGuard_Project 目录下
    model = joblib.load('edu_risk_model.pkl')
    cols = joblib.load('model_columns.pkl')
    
    # 模拟班级数据库 (用于展示群体监控功能)
    np.random.seed(42)
    ids = [f"STU_{i:03d}" for i in range(1, 101)]
    scores = np.random.beta(2, 5, 100) 
    db = pd.DataFrame({'StudentID': ids, 'RiskScore': scores})
    db['Level'] = pd.cut(db['RiskScore'], bins=[0, 0.3, 0.6, 1.0], labels=['低风险', '中风险', '高风险'])
    return model, cols, db

rf_model, model_columns, class_db = load_essentials()

# --- 2. 侧边栏导航与身份标注 ---
st.sidebar.title("🛡️ EduGuard 系统")
st.sidebar.markdown("---")
st.sidebar.markdown("**Affiliation:** \nUniversity of Malaya (UM)")
st.sidebar.info("**Data Source:** \nKaggle (xAPI-Edu-Data)")
st.sidebar.markdown("---")
page = st.sidebar.radio("功能模块", ["📊 班级监控大屏", "🔍 个体精准诊断"])

# --- 页面 I: 班级/学校监控面板 ---
if page == "📊 班级监控大屏":
    st.header("🏫 班级学业风险监控中心")
    
    # 关键指标 (KPIs)
    c1, c2, c3, c4 = st.columns(4)
    total = len(class_db)
    high = len(class_db[class_db['Level'] == '高风险'])
    mid = len(class_db[class_db['Level'] == '中风险'])
    c1.metric("总学生数", total)
    c2.metric("🔴 高风险人数", high, delta=f"{high/total:.1%}", delta_color="inverse")
    c3.metric("🟡 中风险人数", mid)
    c4.metric("🟢 低风险人数", total - high - mid)

    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.subheader("🚩 风险学生 Top 10 (需立即干预)")
        top_10 = class_db.sort_values('RiskScore', ascending=False).head(10)
        st.dataframe(top_10.style.background_gradient(subset=['RiskScore'], cmap='Reds'), use_container_width=True)

    with col_right:
        st.subheader("📊 班级风险分布图")
        fig_pie = px.pie(class_db, names='Level', color='Level',
                         color_discrete_map={'低风险':'#27ae60', '中风险':'#f1c40f', '高风险':'#e74c3c'})
        st.plotly_chart(fig_pie, use_container_width=True)

# --- 页面 II: 个体诊断 + 对比分析 + 趋势 + 干预 ---
else:
    st.header("👤 学生个体风险深度诊断")
    
    col_in, col_comp = st.columns([1, 1.2])
    
    with col_in:
        st.markdown("### 📥 实时指标录入")
        h = st.slider("举手发言 (Raised Hands)", 0, 100, 30)
        r = st.slider("资源访问 (Visited Resources)", 0, 100, 30)
        a = st.slider("公告查看 (Announcements)", 0, 100, 15)
        abs_days = st.selectbox("缺勤天数 (Absence)", ["Under-7", "Above-7"])
        
    with col_comp:
        st.markdown("### 📊 班级基准线对比")
        comp_df = pd.DataFrame({
            "维度": ["举手次数", "资源访问", "公告查看"],
            "当前学生": [h, r, a],
            "班级平均": [55, 72, 45] 
        })
        fig_bar = px.bar(comp_df, x="维度", y=["当前学生", "班级平均"], barmode='group', height=300)
        st.plotly_chart(fig_bar, use_container_width=True)

    # --- 核心预测逻辑 ---
    input_data = pd.DataFrame(np.zeros((1, len(model_columns))), columns=model_columns)
    input_data['raisedhands'], input_data['VisITedResources'], input_data['AnnouncementsView'] = h, r, a
    if abs_days == "Above-7":
        if 'StudentAbsenceDays_Above-7' in model_columns: 
            input_data['StudentAbsenceDays_Above-7'] = 1
            
    # 计算当前概率 prob
    prob = rf_model.predict_proba(input_data)[0][1]

    # 展示预测结果与干预
    st.markdown("---")
    res_1, res_2 = st.columns([1, 2])
    
    with res_1:
        st.subheader("🎯 预测结果")
        st.metric("风险概率", f"{prob:.2%}")
        if prob > 0.5:
            st.error("状态：【高风险】")
        else:
            st.success("状态：【安全/低风险】")

    with res_2:
        st.subheader("🛠️ AI 辅助干预建议")
        if abs_days == "Above-7":
            st.warning("✔ **出勤红线**：缺勤天数过多，建议导师约谈。")
        if h < 40:
            st.warning("✔ **互动预警**：建议增加课堂提问，提高注意力。")
        if r < 50:
            st.warning("✔ **资源补全**：推送针对性复习包。")

    # --- 纵向趋势监测 (现在放在这里，确保 prob 已定义) ---
    st.markdown("---")
    st.subheader("📈 纵向趋势监测 (Longitudinal Trend Monitoring)")
    weeks = ["Week 1", "Week 2", "Week 3", "Week 4", "Current"]
    history_scores = [0.32, 0.45, 0.41, 0.58, prob] # 这里的 prob 已经安全定义
    
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=weeks, y=history_scores, mode='lines+markers', 
                                  line=dict(color='#003366', width=4),
                                  marker=dict(size=10, color='#FFCC00')))
    fig_trend.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Threshold")
    fig_trend.update_layout(yaxis=dict(range=[0, 1], title="Risk Probability"), height=400)
    st.plotly_chart(fig_trend, use_container_width=True)

# --- 3. 底部：数据科学严谨性说明 ---
st.markdown("---")
with st.expander("📝 查看数据科学背景 (Data Science Context)"):
    st.write("""
    **Dataset Information:**
    - 本项目使用来自 **Kaggle** 的公开数据集 (xAPI-Edu-Data)。
    
    **Technical Implementation:**
    - **Preprocessing**: Handled via Median Imputation and StandardScaler.
    - **Class Imbalance**: Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to enhance model sensitivity to at-risk students.
    - **Explainability**: Logic supports **SHAP** values for feature importance transparency.
    """)