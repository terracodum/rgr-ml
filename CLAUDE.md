# РГР — ML Dashboard (Streamlit)

## Задача
Разработать Streamlit-дашборд для инференса ML-моделей регрессии.
Датасет в файле `data/dataset.csv` (целевая переменная уточняется по данным).

## Структура проекта
```
rgr-ml/
  data/
    dataset.csv
  models/              # сюда сохраняются обученные модели
  pages/
    1_about.py         # о разработчике
    2_dataset.py       # описание датасета и EDA
    3_visualizations.py
    4_inference.py     # инференс моделей
  notebooks/
    train_and_save.ipynb  # обучение и сериализация всех моделей
  app.py               # точка входа
  requirements.txt
  README.md
```

## Модели (6 штук, регрессия, метрика Test R²)

| Слот | Тип | Модель | Целевой R² |
|------|-----|--------|------------|
| ML1 | Классика | ElasticNet | ~0.85 |
| ML2 | Бустинг | XGBoost + Optuna | ~0.91 |
| ML3 | CatBoost | CatBoost + Optuna | ~0.91 |
| ML4 | Бэггинг | RandomForest + RandomizedSearchCV | ~0.91 |
| ML5 | Стэкинг | StackingRegressor | ~0.90 |
| ML6 | Нейросеть | MLPRegressor + Hyperopt | ~0.88 |

## Что нужно сделать

### 1. Ноутбук `notebooks/train_and_save.ipynb`
Обучить все 6 моделей и сохранить в `models/`.

**Порядок:**
1. Загрузить `data/dataset.csv`, сделать train/test split (80/20, random_state=42)
2. Предобработка: заполнить пропуски, масштабировать (StandardScaler где нужно)
3. Для каждой модели — подбор гиперпараметров (Optuna или RandomizedSearchCV), затем финальное обучение
4. Сериализация:
   - ElasticNet, RandomForest, StackingRegressor, MLPRegressor → `pickle`
   - XGBoost → `model.save_model("models/xgboost.json")`
   - CatBoost → `model.save_model("models/catboost.cbm")`
5. Вывести итоговую таблицу Test R² для всех 6 моделей

**Детали моделей:**

ML1 — ElasticNet:
```python
from sklearn.linear_model import ElasticNet
# Подбор alpha и l1_ratio через Optuna или GridSearchCV
```

ML2 — XGBoost + Optuna:
```python
import xgboost as xgb
import optuna
# Подбор: n_estimators, max_depth, learning_rate, subsample, colsample_bytree
```

ML3 — CatBoost + Optuna:
```python
from catboost import CatBoostRegressor
# Подбор: iterations, depth, learning_rate, l2_leaf_reg
```

ML4 — RandomForest + RandomizedSearchCV:
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
# Подбор: n_estimators, max_depth, min_samples_split, min_samples_leaf
```

ML5 — StackingRegressor:
```python
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

StackingRegressor(
    estimators=[
        ('rf',  RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
        ('svr', Pipeline([('sc', StandardScaler()), ('m', LinearSVR(max_iter=3000, random_state=42))])),
        ('knn', Pipeline([('sc', StandardScaler()), ('m', KNeighborsRegressor(n_neighbors=10, n_jobs=-1))])),
    ],
    final_estimator=Ridge(),
    cv=5, n_jobs=-1,
)
```

ML6 — MLPRegressor + Hyperopt:
```python
from sklearn.neural_network import MLPRegressor
from hyperopt import fmin, tpe, hp, Trials
# Подбор: hidden_layer_sizes, activation, alpha, learning_rate_init
```

### 2. Streamlit-приложение

**app.py** — навигация через `st.navigation` или `st.sidebar` по 4 страницам.

**pages/1_about.py:**
- ФИО, группа, цветное фото, тема РГР

**pages/2_dataset.py:**
- Описание датасета: предметная область, признаки, типы данных
- Предобработка: пропуски, кодирование, масштабирование
- EDA: форма данных, describe(), распределение целевой переменной

**pages/3_visualizations.py:**
- Минимум 4 разных вида визуализаций через Matplotlib/Seaborn:
  1. Гистограмма целевой переменной
  2. Корреляционная тепловая карта (heatmap)
  3. Scatter plot — предсказание vs реальные значения (лучшая модель)
  4. Boxplot или barplot по одному из признаков
- Все графики через `st.pyplot(fig)`

**pages/4_inference.py:**
- Селектор модели (`st.selectbox`) — выбор одной из 6
- Два режима ввода:
  - Загрузка CSV (`st.file_uploader`)
  - Ручной ввод признаков (`st.number_input` / `st.slider`)
- Валидация входных данных
- Вывод предсказания в понятном формате

### 3. requirements.txt
```
streamlit
scikit-learn
xgboost
catboost
lightgbm
hyperopt
optuna
pandas
numpy
matplotlib
seaborn
tensorflow  # если используется Keras
```

### 4. Деплой
- Код на GitHub: репозиторий `rgr-ml`
- Деплой на Streamlit Cloud: https://streamlit.io/cloud
- В README указать обе ссылки

## Важно
- Все модели загружаются один раз через `@st.cache_resource`
- Препроцессинг (scaler) сохранять вместе с моделями в `models/`
- Стэкинг и нейросеть требуют StandardScaler — сохранить scaler отдельно как `models/scaler.pkl`