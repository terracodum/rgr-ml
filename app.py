import streamlit as st

import os
import urllib.request
import requests
import streamlit as st
# Создаем директорию под модели на сервере Hugging Face
os.makedirs("models", exist_ok=True)

# --- 1. СЮДА ВСТАВЬТЕ ВАШИ ССЫЛКИ ИЗ БРАУЗЕРА (ЯНДЕКС.ДИСК) ---
YANDEX_SHARING_URLS = {
    "models/elasticnet.pkl": "https://disk.yandex.ru/d/OZyxo8ki1rYZbg",
    "models/xgboost.json": "https://disk.yandex.ru/i/VCUOOfuu0ZPkLQ",
    "models/catboost.cbm": "https://disk.yandex.ru/d/umwdY3RNkM9ZKQ",
    "models/randomforest.pkl": "https://disk.yandex.ru/d/mDrWSGIkAnmUZw",
    "models/stacking.pkl": "https://disk.yandex.ru/d/WxQJ5jBi7lOQ2w",
    "models/mlp.pkl": "https://disk.yandex.ru/d/rX1ZnWH4TKuFSg",
    "models/scaler.pkl": "https://disk.yandex.ru/d/OvJvRmrTaqy1Ug"
}

# --- 2. Функция генерации прямой ссылки на скачивание (Direct Download Link) ---
def get_yandex_direct_url(sharing_url):
    api_url = f"https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key={sharing_url}"
    try:
        response = requests.get(api_url).json()
        return response.get("href") # Возвращает прямую ссылку на downloader.disk.yandex.ru
    except Exception:
        return None

# --- 3. Автоматический хук загрузки файлов при старте приложения ---
for path, sharing_url in YANDEX_SHARING_URLS.items():
    if not os.path.exists(path):
        with st.spinner(f"Загрузка модели {path} из облачного хранилища..."):
            direct_url = get_yandex_direct_url(sharing_url)
            if direct_url:
                try:
                    urllib.request.urlretrieve(direct_url, path)
                except Exception as e:
                    st.error(f"Ошибка сохранения файла {path}: {e}")
            else:
                st.error(f"Не удалось получить доступ к {path}. Проверьте права ссылки!")

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
