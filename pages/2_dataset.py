import streamlit as st
import pandas as pd

@st.cache_data
def load_data():
    return pd.read_csv("data/dataset.csv")

st.title("Датасет: цены на автомобили")

df = load_data()

st.header("Описание предметной области")
st.markdown("""
Датасет содержит информацию об автомобилях, выставленных на продажу.
Задача — предсказать **цену автомобиля в USD** (`price_usd`) на основе его характеристик.

Данные включают технические параметры авто, информацию о состоянии, регионе продажи
и дополнительные бинарные признаки.
""")

st.header("Признаки датасета")

feature_descriptions = {
    "manufacturer_name": "Производитель (закодирован числом)",
    "transmission": "Тип трансмиссии (0 — механика, 1 — автомат)",
    "odometer_value": "Пробег, км",
    "year_produced": "Год выпуска",
    "engine_has_gas": "Газовое оборудование (0/1)",
    "engine_capacity": "Объём двигателя, л",
    "has_warranty": "Наличие гарантии (0/1)",
    "state": "Состояние автомобиля",
    "drivetrain": "Тип привода",
    "price_usd": "🎯 Цена в USD (целевая переменная)",
    "is_exchangeable": "Возможен обмен (0/1)",
    "number_of_photos": "Количество фото в объявлении",
    "up_counter": "Число поднятий объявления",
    "feature_0..9": "Дополнительные бинарные признаки комплектации",
    "duration_listed": "Дней в продаже",
    "loc_*": "One-hot кодировка региона (6 областей Беларуси)",
    "body_*": "One-hot кодировка типа кузова",
    "fuel_*": "One-hot кодировка типа топлива",
}

feat_df = pd.DataFrame(
    [(k, v) for k, v in feature_descriptions.items()],
    columns=["Признак", "Описание"]
)
st.dataframe(feat_df, use_container_width=True, hide_index=True)

st.header("Предобработка данных")
st.markdown("""
- **Пропуски:** отсутствуют (датасет уже очищен)
- **Кодирование категорий:** применено One-Hot Encoding для региона, типа кузова и типа топлива
- **Масштабирование:** StandardScaler применяется внутри моделей, которые чувствительны к масштабу (SVR, KNN, MLP, ElasticNet)
- **Разбивка:** train/test split — 80% / 20%, `random_state=42`
""")

st.header("EDA — разведочный анализ данных")

col1, col2 = st.columns(2)
with col1:
    st.metric("Строк", f"{df.shape[0]:,}")
    st.metric("Признаков", df.shape[1])
with col2:
    st.metric("Пропусков", df.isnull().sum().sum())
    st.metric("Средняя цена", f"${df['price_usd'].mean():,.0f}")

st.subheader("Первые строки датасета")
st.dataframe(df.head(10), use_container_width=True)

st.subheader("Статистика числовых признаков")
st.dataframe(df.describe().T.round(2), use_container_width=True)

st.subheader("Распределение целевой переменной (price_usd)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Мин", f"${df['price_usd'].min():,.0f}")
col2.metric("Медиана", f"${df['price_usd'].median():,.0f}")
col3.metric("Среднее", f"${df['price_usd'].mean():,.0f}")
col4.metric("Макс", f"${df['price_usd'].max():,.0f}")
