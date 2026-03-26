# Data_for_ML

Система для полуавтоматической сборки и доразметки датасетов (в первую очередь изображений).  
Она строит единый pipeline из 4 агентов:

1. `dataset-agent`
   Собирает данные по пользовательскому запросу и раскладывает их.

2. `quality-agent`
   Очищает собранный датасет, в первую очередь удаляет дубликаты и near-duplicates. В случае с табличными данными пытается заполнить пропуски и применяет разные стратегии для этого

3. `annotation-agent`
   Делает авторазметку bbox, готовит `labels.csv` и экспорт в Label Studio.

4. `al-agent`
   Запускает active learning поверх bbox-разметки, экспортирует uncertain batch в Label Studio, ждёт human feedback и продолжает цикл в том же процессе.

## Что делает система

Основной сценарий такой:

1. Пользователь даёт запрос на датасет.
2. `dataset-agent` собирает изображения.
3. `quality-agent` чистит датасет.
4. `annotation-agent` строит стартовую bbox-разметку.
5. `al-agent` выбирает самые полезные примеры для ручной доразметки.
6. Пользователь размечает batch в Label Studio.
7. `al-agent` подхватывает human export, мержит его и переобучает модель.

## Структура агентов

- [dataset-agent](/home/agar1us/Data_for_ML/dataset-agent)
- [data_quality_tools_agent](/home/agar1us/Data_for_ML/data_quality_tools_agent)
- [annotation_agent](/home/agar1us/Data_for_ML/annotation_agent)
- [al_agent](/home/agar1us/Data_for_ML/al_agent)

Главный orchestration entrypoint:
- [testing_pipeline.py](/home/agar1us/Data_for_ML/testing_pipeline.py)

## Артефакты pipeline

По умолчанию все stage-артефакты пишутся в:
- [data/current_run](/home/agar1us/Data_for_ML/data/current_run)

Основные каталоги:
- `data/current_run/collection`
- `data/current_run/quality`
- `data/current_run/annotation`
- `data/current_run/al`
- `data/current_run/logs`

## Зависимости

В репозитории используется единый файл зависимостей:
- [requirements.txt](/home/agar1us/Data_for_ML/requirements.txt)

Установка:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Переменные окружения

Минимально нужен `.env` в корне проекта.

Обычно достаточно:

```env
OPENAI_API_KEY=...
```

Опционально, если используешь Kaggle:

```env
KAGGLE_USERNAME=...
KAGGLE_KEY=...
```

## Как запустить весь pipeline одной командой

Полный запуск всех 4 стадий:

```bash
.venv/bin/python testing_pipeline.py "Создай датасет для object detection лебедей. Интересует 3 вида: кликун, шипун, малый. Ищи в Яндекс Картинки. Хватит 50 картинок на каждый класс"
```

Эта команда:
- запускает `dataset-agent`
- затем `quality-agent`
- затем `annotation-agent`
- затем `al-agent`

## Частичный запуск

Можно запускать только часть pipeline:

```bash
.venv/bin/python testing_pipeline.py "Создай датасет для object detection лебедей. Интересует 3 вида: кликун, шипун, малый. Ищи в Яндекс Картинки. Хватит 50 картинок на каждый класс" --stages dataset quality annotation
```

Или только `al` поверх уже готовой разметки:

```bash
.venv/bin/python testing_pipeline.py "Создай датасет для object detection лебедей. Интересует 3 вида: кликун, шипун, малый. Ищи в Яндекс Картинки. Хватит 50 картинок на каждый класс" --stages al --al-n-iterations 1 --al-strategy confidence --al-wait-for-human-feedback
```

## Label Studio

`annotation-agent` и `al-agent` генерируют:
- `labelstudio_import.json`
- `labelstudio_config.xml`

Для локального запуска Label Studio в Docker обычно используется:

```bash
docker run -it -p 8080:8080 \
  --name label-studio \
  -v labelstudio_data:/label-studio/data \
  -v /home/agar1us/Data_for_ML:/label-studio/files \
  -e LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \
  -e LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/label-studio/files \
  heartexlabs/label-studio:latest
```

## Полезные CLI-параметры

У [testing_pipeline.py](/home/agar1us/Data_for_ML/testing_pipeline.py) есть, в частности:

- `--stages`
- `--annotation-object-prompt`
- `--annotation-confidence-threshold`
- `--al-batch-size`
- `--al-n-iterations`
- `--al-test-size`
- `--al-strategy`
- `--al-wait-for-human-feedback`
- `--al-human-feedback-dir`
