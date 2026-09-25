"""Скачивает изображения пяти классов членистоногих с iNaturalist.

Источник: открытый API iNaturalist (https://api.inaturalist.org/v1/observations),
берутся только наблюдения со статусом research и лицензиями CC.
Порядок наблюдений фиксирован (order_by=id), поэтому выборка повторяется.

Запуск: python download_data.py [сколько картинок на класс]
Результат: data/<класс>/<observation_id>.jpg
"""

import os
import sys
import time
import urllib.request

API = 'https://api.inaturalist.org/v1/observations'
HEADERS = {'User-Agent': 'mospolytech-ml-coursework/1.0'}

# taxon_id в iNaturalist, проверены через /v1/taxa
CLASSES = {
    'scorpion': 48894,    # Scorpiones, скорпионы
    'spider': 47118,      # Araneae, пауки
    'centipede': 49556,   # Chilopoda, губоногие многоножки
    'crab': 121639,       # Brachyura, крабы
    'beetle': 47208,      # Coleoptera, жуки
}

PER_PAGE = 200


def fetch_page(taxon_id, page):
    url = (f'{API}?taxon_id={taxon_id}&photos=true&quality_grade=research'
           f'&license=cc0,cc-by,cc-by-nc&order_by=id&order=asc'
           f'&per_page={PER_PAGE}&page={page}')
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as r:
        import json
        return json.load(r)['results']


def download(name, taxon_id, want):
    out = os.path.join('data', name)
    os.makedirs(out, exist_ok=True)
    have = len(os.listdir(out))
    page = 1
    while have < want and page <= 50:
        try:
            results = fetch_page(taxon_id, page)
        except Exception as e:
            print(f'{name}: страница {page} не загрузилась ({e}), пробую дальше')
            page += 1
            time.sleep(2)
            continue
        if not results:
            break
        for obs in results:
            if have >= want:
                break
            photos = obs.get('photos') or []
            if not photos:
                continue
            # square -> medium: то же фото со стороной около 500 px
            url = photos[0]['url'].replace('square', 'medium')
            path = os.path.join(out, f"{obs['id']}.jpg")
            if os.path.exists(path):
                have += 1
                continue
            try:
                req = urllib.request.Request(url, headers=HEADERS)
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = r.read()
                if len(data) < 5000:      # заглушки и битые файлы пропускаем
                    continue
                with open(path, 'wb') as f:
                    f.write(data)
                have += 1
            except Exception:
                continue
        print(f'{name}: {have} из {want}')
        page += 1
        time.sleep(1)
    return have


if __name__ == '__main__':
    want = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    for name, taxon in CLASSES.items():
        n = download(name, taxon, want)
        print(f'готово {name}: {n} изображений')
