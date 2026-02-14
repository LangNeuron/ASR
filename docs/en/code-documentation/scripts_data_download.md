# scripts/data_download documentation

## Purpose
`bash scripts/data_download.sh` downloads selected ASR datasets into `data/raw`, extracts archives, and computes dataset statistics:
- audio file count per dataset;
- estimated hours per dataset;
- total file count and hours.

The shell script is a wrapper for `scripts/data_download.py`.

## Data sources
Built-in sources:
- Golos Opus
- SOVA (RuDevices, Yandex public link)
- OpenSTT subset `asr_public_phone_calls_2` (archive + manifest)
- OpenSTT subset `public_youtube1120` (archive + manifest)

Optional source (only if user provides URL):
- Mozilla/Common Voice archive via `--mozilla-url`

## Output layout
Data is stored under `data/raw`:
- `data/raw/golos_opus/`
- `data/raw/sova_rudevices/`
- `data/raw/open_stt/asr_public_phone_calls_2/`
- `data/raw/open_stt/public_youtube1120/`
- `data/raw/mozilla_voice_custom/` (only when `--mozilla-url` is set)

## CLI
```bash
bash scripts/data_download.sh [options]
```

Arguments:
- `--raw-dir PATH`
  Change target root directory (default: `data/raw`).
- `--datasets {golos,sova,openstt_phone,openstt_youtube,all} ...`
  Select built-in datasets. Default: `all`.
- `--mozilla-url URL`
  Optional custom Mozilla/Common Voice archive URL.
  If present, Mozilla is downloaded even though it is not part of `--datasets`.
- `--sova-url URL`
  Override the default SOVA Yandex public URL.
- `--no-extract`
  Download archives only, skip extraction.

## Examples
Download all built-in datasets:
```bash
bash scripts/data_download.sh
```

Download only OpenSTT subsets:
```bash
bash scripts/data_download.sh --datasets openstt_phone openstt_youtube
```

Download Golos and SOVA without extraction:
```bash
bash scripts/data_download.sh --datasets golos sova --no-extract
```

Download Mozilla from custom URL in addition to selected datasets:
```bash
bash scripts/data_download.sh --datasets golos --mozilla-url "https://example.com/mozilla_ru.tar.gz"
```

Change target directory:
```bash
bash scripts/data_download.sh --raw-dir data/custom_raw --datasets all
```

## Duration calculation
Duration is estimated in this order:
1. Manifest duration column (if available in CSV).
2. `soundfile` metadata reader.
3. WAV header parsing via stdlib `wave`.
4. `ffprobe` fallback.

If a file duration cannot be read, it is skipped (treated as 0 seconds).

## Safety notes
- Only `http`/`https` URL schemes are allowed.
- Requests use explicit timeout.
- Archive extraction has path traversal checks.
- Archive contents are not re-encoded and audio format is not converted.

## Dependencies and environment
- Python 3.12+
- Optional: `soundfile`, `ffprobe` for better duration detection

## Troubleshooting
- `Unsupported URL scheme`: use an `http` or `https` URL.
- SOVA download failed: verify Yandex public link accessibility.
- Low hour counts: ensure extraction finished and audio files are present.
