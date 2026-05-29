import streamlit as st

st.set_page_config(
    page_title="ML Dashboard — Цены на автомобили",
    page_icon="🚗",
    layout="wide",
)

pages = {
    "Навигация": [
        st.Page("pages/1_about.py", title="О разработчике", icon="👤"),
        st.Page("pages/2_dataset.py", title="Датасет", icon="📊"),
        st.Page("pages/3_visualizations.py", title="Визуализации", icon="📈"),
        st.Page("pages/4_inference.py", title="Инференс моделей", icon="🤖"),
    ]
}

pg = st.navigation(pages)
pg.run()
