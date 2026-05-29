import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

@st.cache_data
def load_data():
    return pd.read_csv("data/dataset.csv")

st.title("Визуализации датасета")

df = load_data()

# 1. Гистограмма целевой переменной
st.header("1. Распределение цены автомобиля")
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(df["price_usd"], bins=80, color="#4C72B0", edgecolor="white", linewidth=0.5)
ax.set_xlabel("Цена, USD")
ax.set_ylabel("Количество объявлений")
ax.set_title("Распределение цены автомобиля (price_usd)")
ax.axvline(df["price_usd"].median(), color="red", linestyle="--", label=f"Медиана: ${df['price_usd'].median():,.0f}")
ax.legend()
plt.tight_layout()
st.pyplot(fig)
plt.close()

# 2. Корреляционная тепловая карта (числовые признаки)
st.header("2. Корреляционная матрица (числовые признаки)")
num_cols = ["odometer_value", "year_produced", "engine_capacity",
            "number_of_photos", "up_counter", "duration_listed", "price_usd"]
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            square=True, linewidths=0.5, ax=ax)
ax.set_title("Корреляция числовых признаков с ценой")
plt.tight_layout()
st.pyplot(fig)
plt.close()

# 3. Scatter: пробег vs цена (с окраской по году)
st.header("3. Пробег vs Цена (с разбивкой по году выпуска)")
fig, ax = plt.subplots(figsize=(10, 5))
sample = df.sample(min(3000, len(df)), random_state=42)
sc = ax.scatter(
    sample["odometer_value"], sample["price_usd"],
    c=sample["year_produced"], cmap="viridis", alpha=0.5, s=15
)
plt.colorbar(sc, ax=ax, label="Год выпуска")
ax.set_xlabel("Пробег, км")
ax.set_ylabel("Цена, USD")
ax.set_title("Зависимость цены от пробега")
plt.tight_layout()
st.pyplot(fig)
plt.close()

# 4. Boxplot: цена по типу трансмиссии
st.header("4. Цена по типу трансмиссии")
fig, ax = plt.subplots(figsize=(7, 5))
trans_map = {0: "Механика", 1: "Автомат"}
df_box = df.copy()
df_box["Трансмиссия"] = df_box["transmission"].map(trans_map)
sns.boxplot(data=df_box, x="Трансмиссия", y="price_usd",
            palette="Set2", showfliers=False, ax=ax)
ax.set_ylabel("Цена, USD")
ax.set_title("Распределение цены по типу трансмиссии (без выбросов)")
plt.tight_layout()
st.pyplot(fig)
plt.close()

# 5. Barplot: средняя цена по типу топлива
st.header("5. Средняя цена по типу топлива")
fuel_cols = [c for c in df.columns if c.startswith("fuel_")]
fuel_means = {}
for col in fuel_cols:
    name = col.replace("fuel_", "").replace("-", " ").capitalize()
    fuel_means[name] = df[df[col] == 1]["price_usd"].mean()

fuel_df = pd.DataFrame(list(fuel_means.items()), columns=["Топливо", "Средняя цена"]).sort_values("Средняя цена", ascending=False)
fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.barh(fuel_df["Топливо"], fuel_df["Средняя цена"], color=sns.color_palette("muted", len(fuel_df)))
ax.set_xlabel("Средняя цена, USD")
ax.set_title("Средняя цена автомобиля по типу топлива")
for bar, val in zip(bars, fuel_df["Средняя цена"]):
    ax.text(val + 50, bar.get_y() + bar.get_height() / 2,
            f"${val:,.0f}", va="center", fontsize=9)
plt.tight_layout()
st.pyplot(fig)
plt.close()
