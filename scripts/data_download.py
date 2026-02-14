"""CLI downloader for ASR datasets with extraction and duration statistics."""

from __future__ import annotations

import argparse
import csv
import gzip
import importlib
import json
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.parse
import urllib.request
import wave
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, cast


class SoundFileInfo(Protocol):
    """Typed subset of `soundfile.info()` return object."""

    samplerate: int
    frames: int


class SoundFileModule(Protocol):
    """Typed subset of `soundfile` module used by this script."""

    def info(self, file: str) -> SoundFileInfo:  # pyright: ignore
        """Return metadata for an audio file."""


SOUND_FILE: SoundFileModule | None
try:
    SOUND_FILE = cast(SoundFileModule, importlib.import_module("soundfile"))
except ImportError:
    SOUND_FILE = None

GOLOS_URL = "https://cdn.chatwm.opensmodel.sberdevices.ru/golos/golos_opus.tar"
SOVA_PUBLIC_URL = "https://disk.yandex.ru/d/jz3k7pnzTpnTgw"

OPENSTT_PHONE_ARCHIVE = (
    "https://azureopendatastorage.blob.core.windows.net/openstt/"
    "ru_open_stt_opus/archives/asr_public_phone_calls_2.tar.gz"
)
OPENSTT_PHONE_MANIFEST = (
    "https://azureopendatastorage.blob.core.windows.net/openstt/"
    "ru_open_stt_opus/manifests/asr_public_phone_calls_2.csv"
)
OPENSTT_YT_ARCHIVE = (
    "https://azureopendatastorage.blob.core.windows.net/openstt/"
    "ru_open_stt_opus/archives/public_youtube1120.tar.gz"
)
OPENSTT_YT_MANIFEST = (
    "https://azureopendatastorage.blob.core.windows.net/openstt/"
    "ru_open_stt_opus/manifests/public_youtube1120.csv"
)

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".opus", ".ogg", ".m4a", ".aac", ".wma", ".mp4"}
FIFTY_MB = 50 * 1024 * 1024
URL_TIMEOUT_SECONDS = 60
ALLOWED_URL_SCHEMES = {"http", "https"}
DatasetStats = tuple[str, int, float]
DatasetTargets = tuple[str, Path, Path | None]


def log(msg: str) -> None:
    """Print timestamped progress messages."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def format_size(num: int) -> str:
    """Format byte size into human-readable units."""
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num} B"


def validate_download_url(url: str) -> str:
    """Validate URL scheme before external download requests."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ALLOWED_URL_SCHEMES:
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme!r} for URL {url!r}")
    return url


def download_file(url: str, destination: Path) -> Path:
    """Download URL content to destination with progress logs."""
    safe_url = validate_download_url(url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    request_headers = {"User-Agent": "ASRDownloader/1.0"}
    req = urllib.request.Request(safe_url, headers=request_headers)  # noqa: S310
    with urllib.request.urlopen(req, timeout=URL_TIMEOUT_SECONDS) as response:  # noqa: S310
        total_str = response.headers.get("Content-Length")
        total = int(total_str) if total_str and total_str.isdigit() else None
        downloaded = 0
        chunk_size = 1024 * 1024
        next_report = 0
        with destination.open("wb") as out:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                if total:
                    percent = int(downloaded * 100 / total)
                    if percent >= next_report:
                        log(
                            f"Downloading {destination.name}: {percent}% "
                            f"({format_size(downloaded)} / {format_size(total)})"
                        )
                        next_report += 5
                elif downloaded // FIFTY_MB > (downloaded - len(chunk)) // FIFTY_MB:
                    log(f"Downloading {destination.name}: {format_size(downloaded)}")
    log(f"Saved: {destination}")
    return destination


def get_yandex_direct_url(public_url: str) -> str:
    """Resolve a Yandex public link into a direct download URL."""
    validate_download_url(public_url)
    api = (
        "https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key="
        f"{urllib.parse.quote(public_url, safe='')}"
    )
    request_headers = {"User-Agent": "ASRDownloader/1.0"}
    req = urllib.request.Request(api, headers=request_headers)  # noqa: S310
    with urllib.request.urlopen(req, timeout=URL_TIMEOUT_SECONDS) as response:  # noqa: S310
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Yandex Disk API returned invalid JSON payload.")
    href = payload.get("href")
    if not isinstance(href, str) or not href:
        raise RuntimeError(f"Yandex Disk API did not return 'href': {payload}")
    return href


def _resolve_safe_target(base: Path, candidate: Path) -> Path:
    """Validate extraction target path against path traversal."""
    base_resolved = base.resolve()
    candidate_resolved = candidate.resolve()
    if candidate_resolved != base_resolved and base_resolved not in candidate_resolved.parents:
        raise RuntimeError(f"Unsafe archive member path: {candidate}")
    return candidate_resolved


def _safe_extract_tar(tf: tarfile.TarFile, target_dir: Path) -> None:
    """Safely extract tar archive content into target directory."""
    for member in tf.getmembers():
        member_path = target_dir / member.name
        _resolve_safe_target(target_dir, member_path)
    tf.extractall(target_dir)  # noqa: S202


def _safe_extract_zip(zf: zipfile.ZipFile, target_dir: Path) -> None:
    """Safely extract zip archive content into target directory."""
    for member in zf.infolist():
        member_path = target_dir / member.filename
        _resolve_safe_target(target_dir, member_path)
    zf.extractall(target_dir)  # noqa: S202


def extract_archive(archive_path: Path, target_dir: Path) -> None:
    """Extract supported archives into the target directory."""
    suffixes = "".join(archive_path.suffixes).lower()
    target_dir.mkdir(parents=True, exist_ok=True)
    log(f"Extracting {archive_path.name} -> {target_dir}")

    if suffixes.endswith(".tar.gz") or suffixes.endswith(".tgz"):
        with tarfile.open(archive_path, "r:gz") as tf:
            _safe_extract_tar(tf, target_dir)
        return

    if suffixes.endswith(".tar"):
        with tarfile.open(archive_path, "r:") as tf:
            _safe_extract_tar(tf, target_dir)
        return

    if suffixes.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            _safe_extract_zip(zf, target_dir)
        return

    if suffixes.endswith(".gz"):
        out_file = target_dir / archive_path.with_suffix("").name
        with gzip.open(archive_path, "rb") as src, out_file.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        return

    log(f"Skip extraction (unsupported extension): {archive_path.name}")


def ffprobe_duration_seconds(audio_path: Path) -> float | None:
    """Get duration with ffprobe if available."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)  # noqa: S603
        value = result.stdout.strip()
        return float(value) if value else None
    except (subprocess.SubprocessError, OSError, ValueError):
        return None


def wav_duration_seconds(audio_path: Path) -> float | None:
    """Get WAV duration using the standard library."""
    if audio_path.suffix.lower() != ".wav":
        return None
    try:
        with wave.open(str(audio_path), "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            return frames / float(rate) if rate else None
    except (wave.Error, OSError, EOFError, ValueError):
        return None


def soundfile_duration_seconds(audio_path: Path) -> float | None:
    """Get audio duration via soundfile backend."""
    if SOUND_FILE is None:
        return None
    try:
        info = SOUND_FILE.info(str(audio_path))
        if info.samplerate and info.frames:
            return float(info.frames) / float(info.samplerate)
    except (OSError, RuntimeError, ValueError, TypeError):
        return None
    return None


def audio_duration_seconds(audio_path: Path) -> float:
    """Get audio duration with cascading backends."""
    for getter in (soundfile_duration_seconds, wav_duration_seconds, ffprobe_duration_seconds):
        value = getter(audio_path)
        if value is not None and value > 0:
            return value
    return 0.0


def collect_audio_files(root: Path) -> list[Path]:
    """Collect supported audio files recursively."""
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTS:
            files.append(path)
    return files


def load_manifest_duration_seconds(csv_path: Path) -> float | None:
    """Parse known duration columns from OpenSTT-like CSV manifests."""
    if not csv_path.exists():
        return None
    duration_candidates = ["duration", "duration_sec", "duration_s", "audio_duration", "length"]
    total = 0.0
    parsed_any = False
    try:
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return None
            fields = {name.strip().lower(): name for name in reader.fieldnames}
            duration_col = None
            for candidate in duration_candidates:
                if candidate in fields:
                    duration_col = fields[candidate]
                    break
            if duration_col is None:
                return None
            for row in reader:
                raw = (row.get(duration_col) or "").strip().replace(",", ".")
                if not raw:
                    continue
                try:
                    total += float(raw)
                    parsed_any = True
                except ValueError:
                    continue
    except (OSError, UnicodeDecodeError, csv.Error):
        return None
    return total if parsed_any else None


def summarize_dataset(name: str, root: Path, manifest_csv: Path | None = None) -> tuple[int, float]:
    """Compute file count and duration in hours for a dataset directory."""
    log(f"Collecting stats for dataset: {name}")
    audio_files = collect_audio_files(root)
    file_count = len(audio_files)

    duration = 0.0
    manifest_duration = load_manifest_duration_seconds(manifest_csv) if manifest_csv else None
    if manifest_duration is not None:
        duration = manifest_duration
        log(f"Using manifest duration for {name}: {duration / 3600:.2f}h")
    else:
        for index, audio in enumerate(audio_files, start=1):
            duration += audio_duration_seconds(audio)
            if index % 500 == 0:
                log(f"{name}: scanned {index}/{file_count} audio files")

    hours = duration / 3600.0
    log(f"Dataset summary [{name}]: files={file_count}, hours={hours:.2f}")
    return file_count, hours


def maybe_extract(archive_path: Path, target_dir: Path, extract_enabled: bool) -> None:
    """Conditionally extract archive based on CLI flag."""
    if extract_enabled:
        extract_archive(archive_path, target_dir)
    else:
        log(f"Extraction disabled, archive kept: {archive_path}")


def download_mozilla(raw_dir: Path, extract_enabled: bool, mozilla_url: str) -> Path:
    """Download and optionally extract Mozilla dataset from a custom URL."""
    dataset_dir = raw_dir / "mozilla_voice_custom"
    downloads = dataset_dir / "downloads"
    archive = downloads / Path(urllib.parse.urlparse(mozilla_url).path).name
    log("Start download: Mozilla Voice RU")
    download_file(mozilla_url, archive)
    maybe_extract(archive, dataset_dir, extract_enabled)
    return dataset_dir


def download_golos(raw_dir: Path, extract_enabled: bool) -> Path:
    """Download and optionally extract Golos Opus dataset."""
    dataset_dir = raw_dir / "golos_opus"
    downloads = dataset_dir / "downloads"
    archive = downloads / "golos_opus.tar"
    log("Start download: Golos Opus")
    download_file(GOLOS_URL, archive)
    maybe_extract(archive, dataset_dir, extract_enabled)
    return dataset_dir


def download_sova(raw_dir: Path, extract_enabled: bool, sova_public_url: str) -> Path:
    """Download and optionally extract SOVA dataset from Yandex public link."""
    dataset_dir = raw_dir / "sova_rudevices"
    downloads = dataset_dir / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    log("Start download: SOVA (RuDevices) via Yandex Disk public link")
    direct_url = get_yandex_direct_url(sova_public_url)
    archive_name = Path(urllib.parse.urlparse(direct_url).path).name or "sova_dataset.bin"
    archive = downloads / archive_name
    download_file(direct_url, archive)
    maybe_extract(archive, dataset_dir, extract_enabled)
    return dataset_dir


def download_openstt_subset(
    raw_dir: Path,
    subset_name: str,
    archive_url: str,
    manifest_url: str,
    extract_enabled: bool,
) -> tuple[Path, Path]:
    """Download OpenSTT subset archive and manifest; optionally extract archive."""
    dataset_dir = raw_dir / "open_stt" / subset_name
    downloads = dataset_dir / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)

    archive = downloads / Path(urllib.parse.urlparse(archive_url).path).name
    manifest = dataset_dir / Path(urllib.parse.urlparse(manifest_url).path).name

    log(f"Start download: OpenSTT {subset_name} archive")
    download_file(archive_url, archive)
    log(f"Start download: OpenSTT {subset_name} manifest")
    download_file(manifest_url, manifest)

    maybe_extract(archive, dataset_dir, extract_enabled)
    return dataset_dir, manifest


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Download ASR datasets to data/raw and calculate per-dataset and total stats "
            "(audio files + hours)."
        )
    )
    parser.add_argument("--raw-dir", default="data/raw", help="Target raw data directory")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=[
            "golos",
            "sova",
            "openstt_phone",
            "openstt_youtube",
            "all",
        ],
        default=["all"],
        help="Datasets to download",
    )
    parser.add_argument(
        "--mozilla-url",
        default="",
        help="Custom Mozilla archive URL (downloaded only if provided)",
    )
    parser.add_argument("--sova-url", default=SOVA_PUBLIC_URL, help="SOVA public URL (Yandex Disk)")
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Download archives only, do not extract",
    )
    return parser.parse_args(list(argv))


def normalize_selected(datasets: list[str]) -> set[str]:
    """Normalize dataset selection handling the 'all' alias."""
    if "all" in datasets:
        return {"golos", "sova", "openstt_phone", "openstt_youtube"}
    return set(datasets)


def resolve_targets(
    selected: set[str], args: argparse.Namespace, raw_dir: Path, extract_enabled: bool
) -> list[DatasetTargets]:
    """Download selected datasets and return roots/manifests for summarization."""
    targets: list[DatasetTargets] = []
    if args.mozilla_url:
        targets.append(
            (
                "mozilla_voice_custom",
                download_mozilla(raw_dir, extract_enabled, args.mozilla_url),
                None,
            )
        )
    if "golos" in selected:
        targets.append(("golos_opus", download_golos(raw_dir, extract_enabled), None))
    if "sova" in selected:
        targets.append(
            ("sova_rudevices", download_sova(raw_dir, extract_enabled, args.sova_url), None)
        )
    if "openstt_phone" in selected:
        openstt_phone = download_openstt_subset(
            raw_dir,
            "asr_public_phone_calls_2",
            OPENSTT_PHONE_ARCHIVE,
            OPENSTT_PHONE_MANIFEST,
            extract_enabled,
        )
        targets.append(("openstt_phone_calls_2", openstt_phone[0], openstt_phone[1]))
    if "openstt_youtube" in selected:
        openstt_youtube = download_openstt_subset(
            raw_dir,
            "public_youtube1120",
            OPENSTT_YT_ARCHIVE,
            OPENSTT_YT_MANIFEST,
            extract_enabled,
        )
        targets.append(("openstt_public_youtube1120", openstt_youtube[0], openstt_youtube[1]))
    return targets


def summarize_targets(targets: list[DatasetTargets]) -> list[DatasetStats]:
    """Run statistics collection for downloaded dataset targets."""
    stats: list[DatasetStats] = []
    for dataset_name, root, manifest in targets:
        count, hours = summarize_dataset(dataset_name, root, manifest)
        stats.append((dataset_name, count, hours))
    return stats


def print_summary(stats: list[DatasetStats]) -> None:
    """Print per-dataset and total statistics."""
    total_files = sum(item[1] for item in stats)
    total_hours = sum(item[2] for item in stats)
    print("\n" + "=" * 72)
    print("Download and statistics summary")
    print("=" * 72)
    for name, file_count, hours in stats:
        print(f"- {name:<30} files={file_count:<8} hours={hours:.2f}")
    print("-" * 72)
    print(f"TOTAL: files={total_files}, hours={total_hours:.2f}")
    print("=" * 72)


def main(argv: Iterable[str]) -> int:
    """Entry point for dataset downloader CLI."""
    args = parse_args(argv)
    selected = normalize_selected(args.datasets)
    raw_dir = Path(args.raw_dir)
    extract_enabled = not args.no_extract

    raw_dir.mkdir(parents=True, exist_ok=True)
    log(f"Target raw directory: {raw_dir.resolve()}")
    log(f"Selected datasets: {', '.join(sorted(selected))}")

    targets = resolve_targets(selected, args, raw_dir, extract_enabled)
    stats = summarize_targets(targets)

    if not stats:
        log("No datasets selected. Nothing to do.")
        return 0

    print_summary(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
