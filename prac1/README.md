# ПР1: введение в Google Colab, работа с нейронными сетями

«Методы машинного обучения», Московский Политех, группа 231-329, Арутюнян Ф. Р.

Colab, подключение Диска, импорт библиотек, MNIST с разбиением по варианту (random_state = 1),
первая сеть: точность 0,9680 на тесте.

## Скриншоты

`screenshots/01…07` сняты в Google Colab: создание блокнота, переименование, подключение Диска,
выбор GPU, импорт библиотек. Список с описаниями в `defense.md`.

## Запуск

В Colab: загрузить `notebook.ipynb` и выполнить все ячейки.

Локально нужен TensorFlow. Он стоит в отдельном окружении ПР3:

```bash
../prac3/.venv-tf/bin/jupyter nbconvert --to notebook --execute --inplace notebook.ipynb
```

## Файлы

- `notebook.ipynb` весь код, 4 графика
- `screenshots/` скриншоты работы в Colab
- `report.md` → `report.docx` (`node ../tools/gost_report.js report.md report.docx`)
- `defense.md` шпаргалка и список скриншотов
