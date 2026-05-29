import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

MODELS = {
    "ML1 — ElasticNet": "models/elasticnet.pkl",
    "ML2 — XGBoost": "models/xgboost.json",
    "ML3 — CatBoost": "models/catboost.cbm",
    "ML4 — RandomForest": "models/randomforest.pkl",
    "ML5 — StackingRegressor": "models/stacking.pkl",
    "ML6 — MLPRegressor": "models/mlp.pkl",
}

# Признаки в том же порядке, что и при обучении (включая engineered)
FEATURE_COLS = [
    "manufacturer_name", "transmission", "odometer_value", "year_produced",
    "engine_has_gas", "engine_capacity", "has_warranty", "state", "drivetrain",
    "is_exchangeable", "number_of_photos", "up_counter",
    "feature_0", "feature_1", "feature_2", "feature_3", "feature_4",
    "feature_5", "feature_6", "feature_7", "feature_8", "feature_9",
    "duration_listed",
    "loc_Брестская обл.", "loc_Витебская обл.", "loc_Гомельская обл.",
    "loc_Гродненская обл.", "loc_Минская обл.", "loc_Могилевская обл.",
    "body_cabriolet", "body_coupe", "body_hatchback", "body_liftback",
    "body_limousine", "body_minibus", "body_minivan", "body_pickup",
    "body_sedan", "body_suv", "body_universal", "body_van",
    "fuel_diesel", "fuel_electric", "fuel_gas", "fuel_gasoline",
    "fuel_hybrid-diesel", "fuel_hybrid-petrol",
    "car_age", "log_odometer",
]

def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["car_age"]      = 2024 - df["year_produced"]
    df["log_odometer"] = np.log1p(df["odometer_value"])
    return df

@st.cache_resource
def load_model(name, path):
    if not os.path.exists(path):
        return None
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)
    if path.endswith(".json"):
        import xgboost as xgb
        model = xgb.XGBRegressor()
        model.load_model(path)
        return model
    if path.endswith(".cbm"):
        from catboost import CatBoostRegressor
        model = CatBoostRegressor()
        model.load_model(path)
        return model
    return None

@st.cache_resource
def load_scaler():
    path = "models/scaler.pkl"
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

def predict(model_name, model, scaler, X: pd.DataFrame) -> float:
    needs_scaling = any(k in model_name for k in ["ElasticNet", "Stacking", "MLP"])
    if needs_scaling and scaler is not None:
        X = pd.DataFrame(scaler.transform(X), columns=X.columns)
    return float(model.predict(X)[0])


st.title("Инференс моделей ML")
st.markdown("Получите прогноз **цены автомобиля в USD** с помощью одной из 6 обученных моделей.")

models_ready = all(os.path.exists(p) for p in MODELS.values())
if not models_ready:
    st.warning("Модели ещё не обучены. Запустите `notebooks/train_and_save.ipynb` и перезапустите приложение.")

model_name = st.selectbox("Выберите модель", list(MODELS.keys()))
model = load_model(model_name, MODELS[model_name])
scaler = load_scaler()

tab_manual, tab_csv = st.tabs(["Ручной ввод", "Загрузка CSV"])

with tab_manual:
    st.subheader("Введите параметры автомобиля")

    col1, col2, col3 = st.columns(3)

    with col1:
        manufacturer_name = st.number_input("Производитель (код)", min_value=1, max_value=100, value=1)
        transmission = st.selectbox("Трансмиссия", [0, 1], format_func=lambda x: "Механика" if x == 0 else "Автомат")
        odometer_value = st.number_input("Пробег, км", min_value=0, max_value=1_000_000, value=100_000, step=5_000)
        year_produced = st.number_input("Год выпуска", min_value=1990, max_value=2024, value=2015)
        engine_capacity = st.number_input("Объём двигателя, л", min_value=0.5, max_value=8.0, value=2.0, step=0.1)

    with col2:
        engine_has_gas = st.selectbox("Газовое оборудование", [0, 1], format_func=lambda x: "Нет" if x == 0 else "Есть")
        has_warranty = st.selectbox("Гарантия", [0, 1], format_func=lambda x: "Нет" if x == 0 else "Есть")
        state = st.number_input("Состояние (1–5)", min_value=1, max_value=5, value=3)
        drivetrain = st.number_input("Привод (1–3)", min_value=1, max_value=3, value=2)
        is_exchangeable = st.selectbox("Обмен возможен", [0, 1], format_func=lambda x: "Нет" if x == 0 else "Да")

    with col3:
        number_of_photos = st.number_input("Кол-во фото", min_value=0, max_value=50, value=8)
        up_counter = st.number_input("Поднятий объявления", min_value=0, max_value=500, value=10)
        duration_listed = st.number_input("Дней в продаже", min_value=0, max_value=1000, value=30)

        region = st.selectbox("Регион", [
            "Брестская обл.", "Витебская обл.", "Гомельская обл.",
            "Гродненская обл.", "Минская обл.", "Могилевская обл."
        ])
        body_type = st.selectbox("Тип кузова", [
            "sedan", "hatchback", "suv", "universal", "minivan",
            "coupe", "pickup", "liftback", "cabriolet", "limousine", "minibus", "van"
        ])
        fuel_type = st.selectbox("Тип топлива", [
            "gasoline", "diesel", "gas", "electric", "hybrid-petrol", "hybrid-diesel"
        ])

    if st.button("Рассчитать цену", type="primary", disabled=model is None):
        base_cols = [c for c in FEATURE_COLS if c not in ("car_age", "log_odometer")]
        row = {col: 0 for col in base_cols}
        row.update({
            "manufacturer_name": manufacturer_name,
            "transmission": transmission,
            "odometer_value": odometer_value,
            "year_produced": year_produced,
            "engine_has_gas": engine_has_gas,
            "engine_capacity": engine_capacity,
            "has_warranty": has_warranty,
            "state": state,
            "drivetrain": drivetrain,
            "is_exchangeable": is_exchangeable,
            "number_of_photos": number_of_photos,
            "up_counter": up_counter,
            "duration_listed": duration_listed,
            f"loc_{region}": 1,
            f"body_{body_type}": 1,
            f"fuel_{fuel_type}": 1,
        })
        X = add_engineered(pd.DataFrame([row]))[FEATURE_COLS]
        price = predict(model_name, model, scaler, X)
        price = max(0, price)
        st.success(f"### Прогнозируемая цена: **${price:,.0f}**")
        st.caption(f"Модель: {model_name}")

with tab_csv:
    st.subheader("Загрузите CSV-файл")
    st.markdown("Файл должен содержать исходные признаки датасета (без `price_usd`). `car_age` и `log_odometer` вычислятся автоматически.")

    uploaded = st.file_uploader("Выберите файл CSV", type="csv")
    if uploaded is not None:
        try:
            df_upload = pd.read_csv(uploaded)
            df_upload = add_engineered(df_upload)
            missing = [c for c in FEATURE_COLS if c not in df_upload.columns]
            if missing:
                st.error(f"Отсутствуют столбцы: {missing}")
            else:
                X = df_upload[FEATURE_COLS]
                if model is None:
                    st.error("Модель не загружена.")
                else:
                    needs_scaling = any(k in model_name for k in ["ElasticNet", "Stacking", "MLP"])
                    X_proc = X.copy()
                    if needs_scaling and scaler is not None:
                        X_proc = pd.DataFrame(scaler.transform(X_proc), columns=X_proc.columns)
                    preds = model.predict(X_proc)
                    preds = np.maximum(preds, 0)
                    df_upload["predicted_price_usd"] = preds.round(0).astype(int)
                    st.dataframe(df_upload[["predicted_price_usd"] + FEATURE_COLS[:5]], use_container_width=True)
                    csv_out = df_upload.to_csv(index=False).encode("utf-8")
                    st.download_button("Скачать результаты", csv_out, "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Ошибка при обработке файла: {e}")
