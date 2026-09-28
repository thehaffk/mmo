# ЛР1: полносвязная сеть, распознавание образов

«Методы машинного обучения», Московский Политех, группа 231-329, Арутюнян Ф. Р.

Fashion-MNIST, две архитектуры полносвязной сети, влияние числа эпох и слоёв,
отдельный этап верификации на 1000 отложенных изображений.

Результат: лучшая сеть (три скрытых слоя) accuracy 0,8879 на тесте и 0,8870 на верификации.

## Данные

Fashion-MNIST скачивается через torchvision в `data/` при первом запуске.

## Запуск

```bash
../.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebook.ipynb
```

Прогон около трёх минут на MPS.

## Файлы

- `notebook.ipynb` весь разбор, 9 графиков
- `figures/` графики в png
- `report.md` → `report.docx` (`node ../tools/gost_report.js report.md report.docx`)
- `defense.md` шпаргалка к защите
