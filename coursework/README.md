# Курсовая работа: анализ данных о пассажирах авиакомпаний с использованием технологий визуализации

«Методы машинного обучения», Московский Политех, группа 231-329, Арутюнян Ф. Р.

Бинарная классификация удовлетворённости пассажира полётом по анкете и оценкам сервиса.
Пять моделей (логистическая регрессия, kNN, дерево, случайный лес, градиентный бустинг),
улучшение масштабированием, отбором признаков и подбором гиперпараметров, 17 графиков.

Лучшая модель: градиентный бустинг после подбора, accuracy 0,9651, ROC-AUC 0,9957 на тесте из 32 470 анкет.

## Данные

[Airline Passenger Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction), Kaggle.
Файлы `train.csv` и `test.csv` (15 МБ) лежат в `data/`, скачивать ничего не нужно.

## Запуск

Из корня репозитория `mmo/`:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
cd coursework
../.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebook.ipynb
```

Полный прогон около двух минут на M4 Pro. Графики пересохраняются в `figures/`.

## Файлы

- `notebook.ipynb` весь анализ с выводом ячеек
- `figures/` 17 графиков в png
- `report.md` исходник отчёта, `report.docx` отчёт по ГОСТ 7.32-2017 (`node ../tools/gost_report.js report.md report.docx`)
- `defense.md` шпаргалка к защите
