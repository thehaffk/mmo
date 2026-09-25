# Лабораторная работа № 4: обработка многоцветных изображений

«Методы машинного обучения», Московский Политех, группа 231-329, Арутюнян Ф. Р.
Вариант 1: скорпион.

Свёрточная сеть различает пять классов членистоногих: скорпион, паук, многоножка, краб, жук.
Сравниваются своя сеть, обученная с нуля, и дообученная ResNet18.

## Данные

Открытый API [iNaturalist](https://api.inaturalist.org/v1/observations): наблюдения со статусом
research и лицензиями CC, по 500 фотографий на класс. В репозитории лежит только скрипт загрузки
и папка `data_sample/` с несколькими примерами, сами изображения занимают сотни мегабайт.

```bash
cd lab4
python download_data.py 500     # создаёт data/<класс>/*.jpg
```

## Запуск

```bash
../.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebook.ipynb
```

## Файлы

- `notebook.ipynb` обучение и оценка
- `download_data.py` загрузка набора
- `figures/` графики в png
- `report.md` → `report.docx` (`node ../tools/gost_report.js report.md report.docx`)
- `defense.md` шпаргалка к защите
