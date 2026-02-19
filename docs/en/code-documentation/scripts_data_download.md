# scripts/data_download documentation

## Purpose
`bash scripts/data_download.sh` downloads selected ASR datasets into `data/raw`, extracts archives, and computes dataset statistics:
- audio file count per dataset;
- estimated hours per dataset;
- total file count and hours.

The shell script is a wrapper for `scripts/data_download.py`.

## Data sources
Built-in downloadable sources:
- Golos Opus
- OpenSTT subset `asr_public_phone_calls_2` (archive + manifest)
- OpenSTT subset `public_youtube1120` (archive + manifest)

Optional sources:
- Mozilla/Common Voice archive via `--mozilla-url` (URL download)
- SOVA/RuDevices via `--sova-archive` (local archive path)

For SOVA archive, download it manually from:
- `https://disk.yandex.ru/d/jz3k7pnzTpnTgw`

## Output layout
Data is stored under `data/raw`:
- `data/raw/golos_opus/`
- `data/raw/open_stt/asr_public_phone_calls_2/`
- `data/raw/open_stt/public_youtube1120/`
- `data/raw/mozilla_voice_custom/` (only when `--mozilla-url` is set)
- `data/raw/sova_rudevices/` (only when `--sova-archive` is set)

## CLI
```bash
bash scripts/data_download.sh [command] [options]
```

Commands:
- `download` (default)
  Downloads missing data, verifies existing archives by size, and resumes partial downloads.
- `reinstall-all`
  Forces re-download of built-in downloadable datasets and re-installs optional local/archive inputs.

Arguments:
- `--raw-dir PATH`
  Change target root directory (default: `data/raw`).
- `--datasets {golos,openstt_phone,openstt_youtube,all} ...`
  Select built-in downloadable datasets. Default: `all`.
- `--mozilla-url URL`
  Optional custom Mozilla/Common Voice archive URL.
  If present, Mozilla is downloaded even though it is not part of `--datasets`.
- `--sova-archive PATH`
  Path to already downloaded local SOVA archive (`RuDevices.tar`).
- `--no-extract`
  Download/copy archives only, skip extraction.

## Retry policy
The downloader uses 3 retry levels:
- Request-level retries: `5` attempts per file transfer operation.
- Dataset-level retries: `3` attempts per dataset pipeline.
- Global retries: `5` attempts for the full selected download pipeline.

## Download behavior
- If archive already exists and local size matches remote size, download is skipped.
- If archive is partially downloaded, script tries to resume via HTTP Range requests.
- If server does not support resume, script restarts that file from zero.
- If extraction detects corrupted archive, script re-downloads archive and retries extraction.
- After successful `.tar` / `.tar.gz` extraction, archive file is removed to save disk space.
- Installed/extracted datasets are tracked and auto-detected, so reruns do not re-download them.

## Examples
Download all built-in downloadable datasets:
```bash
bash scripts/data_download.sh
```

Install SOVA from local archive path:
```bash
bash scripts/data_download.sh --sova-archive /path/to/RuDevices.tar
```

Use SOVA local archive + built-in downloads:
```bash
bash scripts/data_download.sh --datasets all --sova-archive /path/to/RuDevices.tar
```

Resume previously interrupted download:
```bash
bash scripts/data_download.sh --datasets golos
```

Force full re-download/reinstall:
```bash
bash scripts/data_download.sh reinstall-all --sova-archive /path/to/RuDevices.tar
```

## Duration calculation
Duration is estimated in this order:
1. Manifest duration column (if available in CSV).
2. `soundfile` metadata reader.
3. WAV header parsing via stdlib `wave`.
4. `ffprobe` fallback.

If a file duration cannot be read, it is skipped (treated as 0 seconds).

## Safety notes
- Only `http`/`https` URL schemes are allowed for network downloads.
- Requests use explicit timeout.
- Archive extraction has path traversal checks.
- Archive contents are not re-encoded and audio format is not converted.

## Dependencies and environment
- Python 3.12+
- Optional: `soundfile`, `ffprobe` for better duration detection

## Troubleshooting
- `Unsupported URL scheme`: use an `http` or `https` URL.
- `Local archive not found`: verify `--sova-archive` path exists.
- Low hour counts: ensure extraction finished and audio files are present.
