# ПР5: тональность русскоязычных рецензий, предобученные эмбеддинги

«Методы машинного обучения», Московский Политех, группа 231-329, Арутюнян Ф. Р.
Парная к ЛР2: другой корпус и другой метод.

Корпус рецензий Кинопоиска, эмбеддинги Navec против TF-IDF.

Результаты: TF-IDF 0,9221, Navec 0,8906. Для сравнения ЛР2 на IMDB: TF-IDF 0,8736, LSTM 0,7396.

## Данные

Скачиваются в `data/` при первом запуске (около 200 МБ):
- `kinopoisk.jsonl` — 36 591 рецензия, Hugging Face `blinoff/kinopoisk`
- `navec_hudlit.tar` — эмбеддинги Navec, 500 002 слова

## Запуск

```bash
../.venv/bin/pip install navec
../.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebook.ipynb
```

Прогон около двух минут после загрузки данных.

## Файлы

- `notebook.ipynb` весь разбор, 8 графиков
- `figures/` графики в png
- `report.md` → `report.docx` (`node ../tools/gost_report.js report.md report.docx`)
- `defense.md` шпаргалка к защите
