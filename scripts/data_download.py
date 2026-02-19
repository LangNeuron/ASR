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
import urllib.error
import urllib.parse
import urllib.request
import wave
import zipfile
from collections.abc import Callable, Iterable
from contextlib import suppress
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
REQUEST_RETRIES = 5
DATASET_RETRIES = 3
GLOBAL_RETRIES = 5
EXTRACT_REPAIR_RETRIES = 2
RETRY_SLEEP_SECONDS = 3
DatasetStats = tuple[str, int, float]
DatasetTargets = tuple[str, Path, Path | None]
BUILTIN_DATASETS = {"golos", "openstt_phone", "openstt_youtube"}
INSTALL_MARKER_NAME = ".asr_dataset_installed.json"


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


def _parse_int_header(value: str | None) -> int | None:
    """Parse integer HTTP header value."""
    if value is None or not value.isdigit():
        return None
    return int(value)


def _parse_content_range_total(value: str | None) -> int | None:
    """Extract total size from Content-Range header."""
    if value is None or "/" not in value:
        return None
    total_part = value.rsplit("/", maxsplit=1)[-1].strip()
    if total_part == "*" or not total_part.isdigit():
        return None
    return int(total_part)


def get_remote_size(url: str) -> int | None:
    """Try to get remote file size through HTTP HEAD request."""
    safe_url = validate_download_url(url)
    request_headers = {"User-Agent": "ASRDownloader/1.0"}
    req = urllib.request.Request(safe_url, headers=request_headers, method="HEAD")  # noqa: S310
    try:
        with urllib.request.urlopen(req, timeout=URL_TIMEOUT_SECONDS) as response:  # noqa: S310
            return _parse_int_header(response.headers.get("Content-Length"))
    except (OSError, TimeoutError, urllib.error.URLError, urllib.error.HTTPError):
        return None


def download_file(url: str, destination: Path, force_redownload: bool = False) -> Path:
    """Download URL content to destination with resume support and progress logs."""
    safe_url = validate_download_url(url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    remote_size = get_remote_size(safe_url)

    if force_redownload:
        _safe_remove_file(destination)

    if destination.exists():
        local_size = destination.stat().st_size
        if remote_size is not None and local_size == remote_size:
            log(
                f"Already downloaded {destination.name}: "
                f"{format_size(local_size)} (size verified)"
            )
            return destination
        if remote_size is not None and local_size > remote_size:
            log(
                f"Local file is larger than remote for {destination.name}. "
                "Re-downloading from scratch."
            )
            _safe_remove_file(destination)

    start_offset = destination.stat().st_size if destination.exists() else 0
    use_range = start_offset > 0

    for _ in range(2):
        request_headers = {"User-Agent": "ASRDownloader/1.0"}
        if use_range:
            request_headers["Range"] = f"bytes={start_offset}-"

        req = urllib.request.Request(safe_url, headers=request_headers)  # noqa: S310
        with urllib.request.urlopen(req, timeout=URL_TIMEOUT_SECONDS) as response:  # noqa: S310
            status_code = int(getattr(response, "status", 200))
            if use_range and status_code != 206:
                log(
                    f"Server did not honor resume for {destination.name}; "
                    "restarting from 0."
                )
                _safe_remove_file(destination)
                start_offset = 0
                use_range = False
                continue

            content_length = _parse_int_header(response.headers.get("Content-Length"))
            content_range = response.headers.get("Content-Range")

            if status_code == 206:
                total = _parse_content_range_total(content_range)
                if total is None and content_length is not None:
                    total = start_offset + content_length
                mode = "ab"
                downloaded = start_offset
            else:
                total = remote_size if remote_size is not None else content_length
                mode = "wb"
                downloaded = 0

            chunk_size = 1024 * 1024
            next_report = int(downloaded * 100 / total) if total else 0
            with destination.open(mode) as out:
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

            if total is not None and downloaded < total:
                raise RuntimeError(
                    f"Incomplete download for {destination.name}: "
                    f"{format_size(downloaded)} of {format_size(total)}"
                )

            log(f"Saved: {destination}")
            return destination

    raise RuntimeError(f"Failed to download {destination.name}: unable to complete transfer")


def _safe_remove_file(path: Path) -> None:
    """Remove file if it exists."""
    with suppress(OSError):
        path.unlink(missing_ok=True)


def _install_marker_path(dataset_dir: Path) -> Path:
    """Return marker file path used to track extracted datasets."""
    return dataset_dir / INSTALL_MARKER_NAME


def _write_install_marker(dataset_dir: Path, source_url: str) -> None:
    """Write dataset installation marker after successful extraction."""
    marker = _install_marker_path(dataset_dir)
    payload = {
        "installed": True,
        "source_url": source_url,
        "created_at_unix": int(time.time()),
    }
    marker.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _has_extracted_content(dataset_dir: Path) -> bool:
    """Check whether dataset directory already contains extracted content."""
    if not dataset_dir.exists():
        return False
    for entry in dataset_dir.iterdir():
        if entry.name in {"downloads", INSTALL_MARKER_NAME}:
            continue
        return True
    return False


def is_dataset_installed(dataset_dir: Path) -> bool:
    """Return True when dataset appears extracted and ready to use."""
    marker = _install_marker_path(dataset_dir)
    if marker.exists() and _has_extracted_content(dataset_dir):
        return True

    # Backward-compatible auto-detection for datasets extracted before markers were introduced.
    if _has_extracted_content(dataset_dir):
        _write_install_marker(dataset_dir, source_url="auto-detected")
        return True

    return False


def cleanup_dataset_for_reinstall(dataset_dir: Path) -> None:
    """Delete dataset directory to enforce clean reinstall."""
    if dataset_dir.exists():
        log(f"Reinstall mode: removing existing dataset directory {dataset_dir}")
        shutil.rmtree(dataset_dir, ignore_errors=True)


def _should_remove_archive_after_extract(archive_path: Path) -> bool:
    """Return True if extracted archive should be removed to save disk space."""
    suffixes = "".join(archive_path.suffixes).lower()
    return suffixes.endswith(".tar") or suffixes.endswith(".tar.gz") or suffixes.endswith(".tgz")


def download_file_with_retries(
    url: str,
    destination: Path,
    max_attempts: int = REQUEST_RETRIES,
    force_redownload: bool = False,
) -> Path:
    """Download with request-level retries."""
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return download_file(url, destination, force_redownload=force_redownload)
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code in {401, 403, 413, 416}:
                _safe_remove_file(destination)
            if attempt < max_attempts:
                log(
                    f"Download failed ({attempt}/{max_attempts}) for {destination.name}: {error}. "
                    f"Retrying in {RETRY_SLEEP_SECONDS}s..."
                )
                time.sleep(RETRY_SLEEP_SECONDS)
            else:
                break
        except (RuntimeError, TimeoutError, OSError, urllib.error.URLError) as error:
            last_error = error
            if attempt < max_attempts:
                log(
                    f"Download failed ({attempt}/{max_attempts}) for {destination.name}: {error}. "
                    f"Retrying in {RETRY_SLEEP_SECONDS}s..."
                )
                time.sleep(RETRY_SLEEP_SECONDS)
            else:
                break

    raise RuntimeError(
        f"Failed to download {destination.name} after {max_attempts} attempts: "
        f"{last_error}"
    )


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


def maybe_extract_with_repair(
    archive_url: str,
    archive_path: Path,
    target_dir: Path,
    extract_enabled: bool,
    max_attempts: int = EXTRACT_REPAIR_RETRIES,
) -> None:
    """Extract archive and repair by re-downloading if archive is corrupted."""
    if not extract_enabled:
        log(f"Extraction disabled, archive kept: {archive_path}")
        return

    for attempt in range(1, max_attempts + 1):
        try:
            extract_archive(archive_path, target_dir)
            _write_install_marker(target_dir, source_url=archive_url)
            if _should_remove_archive_after_extract(archive_path):
                _safe_remove_file(archive_path)
                log(f"Removed archive after extraction: {archive_path}")
            return
        except (tarfile.ReadError, zipfile.BadZipFile, EOFError, OSError) as error:
            if attempt >= max_attempts:
                raise RuntimeError(
                    f"Failed to extract archive {archive_path} after {max_attempts} attempts: "
                    f"{error}"
                ) from error
            log(
                f"Archive appears corrupted ({archive_path.name}): {error}. "
                "Re-downloading archive and retrying extraction..."
            )
            _safe_remove_file(archive_path)
            download_file_with_retries(archive_url, archive_path, force_redownload=True)


def run_dataset_with_retries(
    dataset_name: str,
    operation: Callable[[], DatasetTargets],
    max_attempts: int = DATASET_RETRIES,
) -> DatasetTargets:
    """Run dataset download/extract with dataset-level retries."""
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except RuntimeError as error:
            last_error = error
            if attempt < max_attempts:
                log(
                    f"Dataset {dataset_name} failed ({attempt}/{max_attempts}): {error}. "
                    f"Retrying dataset in {RETRY_SLEEP_SECONDS}s..."
                )
                time.sleep(RETRY_SLEEP_SECONDS)
            else:
                break
    raise RuntimeError(
        f"Dataset {dataset_name} failed after {max_attempts} attempts: {last_error}"
    )


def download_mozilla(
    raw_dir: Path,
    extract_enabled: bool,
    mozilla_url: str,
    force_redownload: bool = False,
) -> Path:
    """Download and optionally extract Mozilla dataset from a custom URL."""
    dataset_dir = raw_dir / "mozilla_voice_custom"
    if force_redownload:
        cleanup_dataset_for_reinstall(dataset_dir)
    if extract_enabled and not force_redownload and is_dataset_installed(dataset_dir):
        log(f"Dataset already extracted and ready: {dataset_dir}. Skipping download.")
        return dataset_dir

    downloads = dataset_dir / "downloads"
    archive = downloads / Path(urllib.parse.urlparse(mozilla_url).path).name
    log("Start download: Mozilla Voice RU")
    download_file_with_retries(mozilla_url, archive, force_redownload=force_redownload)
    maybe_extract_with_repair(mozilla_url, archive, dataset_dir, extract_enabled)
    return dataset_dir


def download_golos(raw_dir: Path, extract_enabled: bool, force_redownload: bool = False) -> Path:
    """Download and optionally extract Golos Opus dataset."""
    dataset_dir = raw_dir / "golos_opus"
    if force_redownload:
        cleanup_dataset_for_reinstall(dataset_dir)
    if extract_enabled and not force_redownload and is_dataset_installed(dataset_dir):
        log(f"Dataset already extracted and ready: {dataset_dir}. Skipping download.")
        return dataset_dir

    downloads = dataset_dir / "downloads"
    archive = downloads / "golos_opus.tar"
    log("Start download: Golos Opus")
    download_file_with_retries(GOLOS_URL, archive, force_redownload=force_redownload)
    maybe_extract_with_repair(GOLOS_URL, archive, dataset_dir, extract_enabled)
    return dataset_dir


def extract_local_archive_for_dataset(
    raw_dir: Path,
    dataset_name: str,
    archive_path_str: str,
    extract_enabled: bool,
    force_redownload: bool = False,
) -> Path:
    """Use local archive for dataset extraction and tracking."""
    source_archive = Path(archive_path_str).expanduser().resolve()
    if not source_archive.exists() or not source_archive.is_file():
        raise RuntimeError(f"Local archive not found: {source_archive}")

    dataset_dir = raw_dir / dataset_name
    if force_redownload:
        cleanup_dataset_for_reinstall(dataset_dir)
    if extract_enabled and not force_redownload and is_dataset_installed(dataset_dir):
        log(f"Dataset already extracted and ready: {dataset_dir}. Skipping download.")
        return dataset_dir

    downloads = dataset_dir / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)

    archive = downloads / source_archive.name
    source_size = source_archive.stat().st_size
    if not archive.exists() or archive.stat().st_size != source_size:
        log(f"Copy local archive for {dataset_name}: {source_archive} -> {archive}")
        shutil.copy2(source_archive, archive)
    else:
        log(f"Local archive already prepared for {dataset_name}: {archive}")

    if extract_enabled:
        extract_archive(archive, dataset_dir)
        _write_install_marker(dataset_dir, source_url=f"local-archive:{source_archive}")
        if _should_remove_archive_after_extract(archive):
            _safe_remove_file(archive)
            log(f"Removed archive after extraction: {archive}")
    else:
        log(f"Extraction disabled, archive kept: {archive}")

    return dataset_dir


def download_openstt_subset(
    raw_dir: Path,
    subset_name: str,
    archive_url: str,
    manifest_url: str,
    extract_enabled: bool,
    force_redownload: bool = False,
) -> tuple[Path, Path]:
    """Download OpenSTT subset archive and manifest; optionally extract archive."""
    dataset_dir = raw_dir / "open_stt" / subset_name
    if force_redownload:
        cleanup_dataset_for_reinstall(dataset_dir)

    downloads = dataset_dir / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)

    archive = downloads / Path(urllib.parse.urlparse(archive_url).path).name
    manifest = dataset_dir / Path(urllib.parse.urlparse(manifest_url).path).name

    if extract_enabled and not force_redownload and is_dataset_installed(dataset_dir):
        log(f"Dataset already extracted and ready: {dataset_dir}. Skipping download.")
        if not manifest.exists():
            download_file_with_retries(manifest_url, manifest)
        return dataset_dir, manifest

    log(f"Start download: OpenSTT {subset_name} archive")
    download_file_with_retries(archive_url, archive, force_redownload=force_redownload)
    log(f"Start download: OpenSTT {subset_name} manifest")
    download_file_with_retries(manifest_url, manifest, force_redownload=force_redownload)

    maybe_extract_with_repair(archive_url, archive, dataset_dir, extract_enabled)
    return dataset_dir, manifest


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Download ASR datasets to data/raw and calculate per-dataset and total stats "
            "(audio files + hours)."
        )
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="download",
        choices=["download", "reinstall-all"],
        help="`download` keeps existing valid archives; `reinstall-all` forces full re-download.",
    )
    parser.add_argument("--raw-dir", default="data/raw", help="Target raw data directory")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=[
            "golos",
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
    parser.add_argument(
        "--sova-archive",
        default="",
        help="Path to already downloaded SOVA archive (RuDevices.tar)",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Download archives only, do not extract",
    )
    return parser.parse_args(list(argv))


def normalize_selected(datasets: list[str], reinstall_all: bool) -> set[str]:
    """Normalize dataset selection handling the 'all' alias and reinstall mode."""
    if reinstall_all or "all" in datasets:
        return set(BUILTIN_DATASETS)
    return set(datasets)


def resolve_targets(
    selected: set[str],
    args: argparse.Namespace,
    raw_dir: Path,
    extract_enabled: bool,
    force_redownload: bool,
) -> list[DatasetTargets]:
    """Download selected datasets and return roots/manifests for summarization."""
    targets: list[DatasetTargets] = []
    if args.mozilla_url:
        targets.append(
            run_dataset_with_retries(
                "mozilla_voice_custom",
                lambda: (
                    "mozilla_voice_custom",
                    download_mozilla(
                        raw_dir,
                        extract_enabled,
                        args.mozilla_url,
                        force_redownload=force_redownload,
                    ),
                    None,
                ),
            )
        )
    if "golos" in selected:
        targets.append(
            run_dataset_with_retries(
                "golos_opus",
                lambda: (
                    "golos_opus",
                    download_golos(
                        raw_dir, extract_enabled, force_redownload=force_redownload
                    ),
                    None,
                ),
            )
        )
    if args.sova_archive:
        targets.append(
            run_dataset_with_retries(
                "sova_rudevices",
                lambda: (
                    "sova_rudevices",
                    extract_local_archive_for_dataset(
                        raw_dir,
                        "sova_rudevices",
                        args.sova_archive,
                        extract_enabled,
                        force_redownload=force_redownload,
                    ),
                    None,
                ),
            )
        )
    if "openstt_phone" in selected:
        openstt_phone = run_dataset_with_retries(
            "openstt_phone_calls_2",
            lambda: (
                "openstt_phone_calls_2",
                *download_openstt_subset(
                    raw_dir,
                    "asr_public_phone_calls_2",
                    OPENSTT_PHONE_ARCHIVE,
                    OPENSTT_PHONE_MANIFEST,
                    extract_enabled,
                    force_redownload=force_redownload,
                ),
            ),
        )
        targets.append(openstt_phone)
    if "openstt_youtube" in selected:
        openstt_youtube = run_dataset_with_retries(
            "openstt_public_youtube1120",
            lambda: (
                "openstt_public_youtube1120",
                *download_openstt_subset(
                    raw_dir,
                    "public_youtube1120",
                    OPENSTT_YT_ARCHIVE,
                    OPENSTT_YT_MANIFEST,
                    extract_enabled,
                    force_redownload=force_redownload,
                ),
            ),
        )
        targets.append(openstt_youtube)
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
    force_redownload = args.command == "reinstall-all"
    selected = normalize_selected(args.datasets, force_redownload)
    raw_dir = Path(args.raw_dir)
    extract_enabled = not args.no_extract

    raw_dir.mkdir(parents=True, exist_ok=True)
    log(f"Target raw directory: {raw_dir.resolve()}")
    log(f"Selected datasets: {', '.join(sorted(selected))}")
    if force_redownload:
        log("Mode: reinstall-all (forcing re-download of selected datasets)")

    stats: list[DatasetStats] = []
    for attempt in range(1, GLOBAL_RETRIES + 1):
        try:
            targets = resolve_targets(
                selected,
                args,
                raw_dir,
                extract_enabled,
                force_redownload=force_redownload,
            )
            stats = summarize_targets(targets)
            break
        except RuntimeError as error:
            if attempt < GLOBAL_RETRIES:
                log(
                    f"Global download failed ({attempt}/{GLOBAL_RETRIES}): {error}. "
                    f"Retrying full pipeline in {RETRY_SLEEP_SECONDS}s..."
                )
                time.sleep(RETRY_SLEEP_SECONDS)
            else:
                raise RuntimeError(
                    f"Download pipeline failed after {GLOBAL_RETRIES} global attempts: {error}"
                ) from error

    if not stats:
        log("No datasets selected. Nothing to do.")
        return 0

    print_summary(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
