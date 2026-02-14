# Документация scripts/data_download

## Назначение
`bash scripts/data_download.sh` скачивает выбранные ASR-датасеты в `data/raw`, распаковывает архивы и считает статистику:
- количество аудиофайлов по каждому набору;
- оценка часов по каждому набору;
- общий итог по файлам и часам.

Скрипт `scripts/data_download.sh` является оболочкой для `scripts/data_download.py`.

## Источники данных
Встроенные источники:
- Golos Opus
- SOVA (RuDevices, публичная ссылка Yandex)
- OpenSTT `asr_public_phone_calls_2` (архив + манифест)
- OpenSTT `public_youtube1120` (архив + манифест)

Опциональный источник (только при передаче ссылки пользователем):
- Mozilla/Common Voice архив через `--mozilla-url`

## Расположение данных
Данные сохраняются в `data/raw`:
- `data/raw/golos_opus/`
- `data/raw/sova_rudevices/`
- `data/raw/open_stt/asr_public_phone_calls_2/`
- `data/raw/open_stt/public_youtube1120/`
- `data/raw/mozilla_voice_custom/` (только если указан `--mozilla-url`)

## CLI
```bash
bash scripts/data_download.sh [options]
```

Аргументы:
- `--raw-dir PATH`
  Изменить целевую директорию (по умолчанию: `data/raw`).
- `--datasets {golos,sova,openstt_phone,openstt_youtube,all} ...`
  Выбор встроенных наборов данных. По умолчанию: `all`.
- `--mozilla-url URL`
  Опциональная пользовательская ссылка на Mozilla/Common Voice архив.
  Если параметр задан, Mozilla будет скачан, даже если его нет в `--datasets`.
- `--sova-url URL`
  Переопределить ссылку на SOVA.
- `--no-extract`
  Только скачать архивы, без распаковки.

## Примеры
Скачать все встроенные наборы:
```bash
bash scripts/data_download.sh
```

Скачать только OpenSTT:
```bash
bash scripts/data_download.sh --datasets openstt_phone openstt_youtube
```

Скачать Golos и SOVA без распаковки:
```bash
bash scripts/data_download.sh --datasets golos sova --no-extract
```

Скачать Mozilla по пользовательской ссылке дополнительно:
```bash
bash scripts/data_download.sh --datasets golos --mozilla-url "https://example.com/mozilla_ru.tar.gz"
```

Изменить директорию назначения:
```bash
bash scripts/data_download.sh --raw-dir data/custom_raw --datasets all
```

## Подсчет длительности
Длительность считается в таком порядке:
1. Колонка длительности из CSV-манифеста (если есть).
2. `soundfile`.
3. WAV-заголовок через стандартный модуль `wave`.
4. `ffprobe` как fallback.

Если длительность конкретного файла определить нельзя, файл учитывается как 0 секунд.

## Безопасность
- Разрешены только URL-схемы `http`/`https`.
- Для сетевых запросов установлен timeout.
- В распаковке архивов есть защита от path traversal.
- Конвертация аудиоформатов не выполняется.

## Зависимости и окружение
- Python 3.12+
- Опционально: `soundfile`, `ffprobe` для более точного подсчета длительности

## Типовые проблемы
- `Unsupported URL scheme`: используйте только `http` или `https`.
- SOVA не скачивается: проверьте доступность публичной ссылки Yandex.
- Мало часов в отчете: проверьте, что архивы распакованы и аудио-файлы присутствуют.
