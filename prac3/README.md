# ПР3: распознавание аномалий автокодировщиком

«Методы машинного обучения», Московский Политех, группа 231-329, Арутюнян Ф. Р.

Библиотека `lab02_lib.py` из материалов к работе перенесена на актуальные версии (изменения в начале файла).
Двумерный пример из методички: AE1 и AE2, ошибка реконструкции, характеристики EDCA, тест с аномалиями.
Затем реальные наборы из архива: WBC, cardio, letter.

Результат: AE2 нашёл 7 аномалий из 7 на синтетике; на реальных данных с порогом p99 около 38 % аномалий
WBC и cardio при 1 % ложных тревог, на letter метод не работает.

## Окружение

TensorFlow ставится в отдельное окружение, чтобы не трогать общее:

```bash
uv venv --python 3.12 .venv-tf
uv pip install --python .venv-tf/bin/python tensorflow h5py pandas scikit-learn matplotlib seaborn jupyter nbconvert ipykernel
```

## Запуск

```bash
.venv-tf/bin/jupyter nbconvert --to notebook --execute --inplace notebook.ipynb
```

Около пяти минут на CPU. Библиотека пишет свои графики и модели в `out/`, графики для отчёта копируются в `figures/`.

## Файлы

- `lab02_lib.py` перенесённая библиотека преподавателя
- `data/` наборы WBC, cardio, letter из архива
- `notebook.ipynb` весь разбор, 11 графиков
- `report.md` → `report.docx` (`node ../tools/gost_report.js report.md report.docx`)
- `defense.md` шпаргалка к защите
