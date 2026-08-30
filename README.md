# Анализ данных о пассажирах авиакомпаний с использованием технологий визуализации

Курсовая работа по дисциплине «Методы машинного обучения».
Московский политехнический университет, группа 231-329.

## Задача

Бинарная классификация: предсказать удовлетворённость пассажира полётом
(`satisfied` / `neutral or dissatisfied`) по анкетным данным и оценкам сервиса,
а также определить, какие факторы сильнее всего влияют на удовлетворённость.

Датасет — [Airline Passenger Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)
(Kaggle): 129 880 записей, 22 признака + бинарная целевая переменная.

## Структура проекта

```
ml-coursework/
├── airline_passenger_satisfaction.ipynb   # основной ноутбук (весь анализ)
├── data/
│   ├── train.csv                          # исходные данные (Kaggle, часть train)
│   └── test.csv                           # исходные данные (Kaggle, часть test)
├── figures/                               # все графики в PNG (генерируются ноутбуком)
├── report/                                # отчёт по ГОСТ 7.32-2017 (.docx)
└── README.md
```

## Как запустить

Требуется Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn matplotlib seaborn scipy jupyter
jupyter notebook airline_passenger_satisfaction.ipynb
```

Либо выполнить весь ноутбук из командной строки:

```bash
jupyter nbconvert --to notebook --execute --inplace airline_passenger_satisfaction.ipynb
```

Данные лежат в `data/`; если их нет, скачайте датасет со страницы Kaggle
(файлы `train.csv` и `test.csv`) и положите в папку `data/`.

Ноутбук выполняется с нуля до конца без ошибок (~5–10 минут: обучение случайного
леса и подбор гиперпараметров по сетке — самые долгие шаги). Все графики
автоматически сохраняются в `figures/`.

## Использованные модели

| Модель | Зачем |
|---|---|
| Логистическая регрессия | линейный baseline |
| K ближайших соседей | метрический метод |
| Дерево решений | нелинейные правила, интерпретируемость |
| Случайный лес | ансамбль, лучшая модель работы |

Улучшение: стандартизация признаков, отбор признаков, подбор гиперпараметров
(GridSearchCV). Метрики: accuracy, precision, recall, F1, ROC-AUC.
