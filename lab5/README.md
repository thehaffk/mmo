# ЛР5: создание датасета из открытого API и его анализ

«Методы машинного обучения», Московский Политех, группа 231-329, Арутюнян Ф. Р.

По мотивам примера `Анализ-активности` из материалов к работе: сырые данные собираются скриптом
из открытого API Hacker News, приводятся к таблице, очищаются и анализируются.

Результат: `data/hn_stories.csv`, 8000 историй, 16 столбцов.

## Данные

Сбор идёт в ноутбуке через API поиска Hacker News (`hn.algolia.com`), без авторизации.
Сырой ответ сохранён в `data/raw_hn_stories.json`; если файл есть, ноутбук в сеть не ходит.
Чтобы собрать свежие данные, удалите этот файл.

## Запуск

```bash
../.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebook.ipynb
```

## Файлы

- `notebook.ipynb` сбор, разбор, анализ, модель, 6 графиков
- `data/raw_hn_stories.json` сырой ответ API
- `data/hn_stories.csv` итоговый датасет
- `report.md` → `report.docx` (`node ../tools/gost_report.js report.md report.docx`)
- `defense.md` шпаргалка к защите
