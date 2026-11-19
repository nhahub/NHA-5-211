"""Download SMART IDS v3 model artifacts from a GitHub Release URL (Git LFS-free fallback)."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import http.client
import json
from pathlib import Path
import shutil
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_app_config
from src.utils.logging import get_logger
from src.utils.model_registry import ModelIntegrityError, compute_file_sha256, is_git_lfs_pointer

logger = get_logger("smart_ids.scripts.download_models")

DownloaderFn = Callable[[str, Path, int], None]


def _default_fetch(url: str, dest: Path, timeout: int) -> None:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported URL scheme '{parsed.scheme}'; expected http or https.")
    req = Request(url, headers={"User-Agent": "smart-ids-v3-model-downloader/1.0"})
    tmp_path = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(req, timeout=timeout) as resp, tmp_path.open("wb") as out_fh:  # noqa: S310
        if getattr(resp, "status", 200) >= 400:
            raise http.client.HTTPException(f"HTTP {resp.status} downloading {url}")
        shutil.copyfileobj(resp, out_fh)
    tmp_path.replace(dest)


def _validate_asset_name(asset_name: str) -> str:
    clean = asset_name.strip()
    if not clean or "/" in clean or "\\" in clean or clean in {".", ".."}:
        raise ValueError(f"Unsafe asset filename '{asset_name}'.")
    return clean


def download_model_assets(
    base_url: str,
    output_dir: Path,
    required_assets: list[str],
    optional_assets: list[str] | None = None,
    overwrite: bool = False,
    verify_hashes: bool = True,
    dry_run: bool = False,
    timeout: int = 60,
    fetcher: DownloaderFn | None = None,
) -> dict[str, Any]:
    """Download required and optional model artifacts from ``base_url`` into ``output_dir``."""
    base = base_url.strip().rstrip("/")
    parsed = urlsplit(base)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            "A valid release base URL (http:// or https://) is required. Configure "
            "`models_download.release_base_url` / `models_download.release_repo` in configs/config.yaml "
            "or pass --base-url / --repo."
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fetch = fetcher or _default_fetch

    opt_list = [_validate_asset_name(a) for a in (optional_assets or [])]
    req_list = [_validate_asset_name(a) for a in required_assets]
    # Fetch MANIFEST.json first so downloaded binaries can be SHA-256 verified automatically.
    ordered_assets: list[tuple[str, bool]] = []
    if "MANIFEST.json" in opt_list:
        ordered_assets.append(("MANIFEST.json", True))
    for asset in req_list:
        if asset != "MANIFEST.json":
            ordered_assets.append((asset, False))
    for asset in opt_list:
        if asset != "MANIFEST.json" and asset not in req_list:
            ordered_assets.append((asset, True))

    summary: dict[str, Any] = {
        "base_url": base,
        "output_dir": str(out_dir),
        "downloaded": [],
        "skipped_existing": [],
        "optional_missing": [],
        "verified": [],
        "failed_required": [],
    }

    for asset, is_optional in ordered_assets:
        dest = out_dir / asset
        already_binary = dest.exists() and not is_git_lfs_pointer(dest)
        if already_binary and not overwrite:
            summary["skipped_existing"].append(asset)
            continue

        asset_url = f"{base}/{asset}"
        if dry_run:
            logger.info("[DRY RUN] Would download %s -> %s", asset_url, dest)
            summary["downloaded"].append(asset)
            continue

        try:
            logger.info("Downloading %s -> %s", asset_url, dest)
            fetch(asset_url, dest, timeout)
            if is_git_lfs_pointer(dest):
                dest.unlink(missing_ok=True)
                raise ValueError(f"Downloaded file '{asset}' is a Git LFS pointer stub, not a binary asset.")
            summary["downloaded"].append(asset)
        except (HTTPError, URLError, http.client.HTTPException, OSError, ValueError) as exc:
            status = getattr(exc, "code", None)
            if is_optional and (status == 404 or isinstance(exc, FileNotFoundError)):
                logger.info("Optional asset not present at release URL (%s): %s", exc, asset)
                summary["optional_missing"].append(asset)
            elif is_optional:
                logger.warning("Skipping optional asset '%s' due to download error: %s", asset, exc)
                summary["optional_missing"].append(asset)
            else:
                logger.error("Failed downloading required asset '%s': %s", asset, exc)
                summary["failed_required"].append({"asset": asset, "error": str(exc)})

    manifest_path = out_dir / "MANIFEST.json"
    if verify_hashes and not dry_run and manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as fh:
            manifest_data = json.load(fh)
        expected_map = dict(manifest_data.get("artifacts", {}) or {})
        for asset, expected_entry in expected_map.items():
            target = out_dir / asset
            if not target.exists() or is_git_lfs_pointer(target):
                continue
            expected_sha = str(expected_entry.get("sha256", "")).strip().lower()
            actual_sha = compute_file_sha256(target).lower()
            if expected_sha and actual_sha != expected_sha:
                target.unlink(missing_ok=True)
                raise ModelIntegrityError(
                    f"Downloaded asset '{asset}' failed SHA-256 verification "
                    f"(expected {expected_sha}, got {actual_sha}). Corrupt file removed."
                )
            summary["verified"].append(asset)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download SMART IDS v3 model artifacts from a GitHub Release URL (Git LFS fallback)."
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Direct release asset base URL (overrides models_download.release_base_url).",
    )
    parser.add_argument(
        "--repo",
        default=None,
        help="GitHub repository slug 'owner/repo' to construct release download URL.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="GitHub release tag when using --repo (default: models_download.release_tag).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Destination directory for model files (default: paths.models_dir).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing local binaries instead of skipping them.",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip SHA-256 verification against MANIFEST.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned downloads without performing network requests.",
    )
    parser.add_argument("--timeout", type=int, default=60, help="HTTP request timeout in seconds.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app_cfg = get_app_config()
    dl_cfg = app_cfg.models_download

    if args.base_url:
        base_url = args.base_url.strip()
    elif args.repo:
        tag = (args.tag or dl_cfg.release_tag or "models-v3").strip()
        base_url = f"https://github.com/{args.repo.strip().strip('/')}/releases/download/{tag}"
    else:
        base_url = dl_cfg.resolved_base_url()

    if not base_url:
        logger.error(
            "No release URL configured. Set `models_download.release_base_url` or "
            "`models_download.release_repo` in configs/config.yaml, or pass --base-url / --repo."
        )
        return 2

    out_dir = Path(args.output_dir) if args.output_dir else app_cfg.paths.resolve("models_dir")
    try:
        summary = download_model_assets(
            base_url=base_url,
            output_dir=out_dir,
            required_assets=list(dl_cfg.required_assets),
            optional_assets=list(dl_cfg.optional_assets),
            overwrite=args.overwrite,
            verify_hashes=not args.skip_verify,
            dry_run=args.dry_run,
            timeout=args.timeout,
        )
    except (ValueError, ModelIntegrityError) as exc:
        logger.error("Model download aborted: %s", exc)
        return 1

    print(json.dumps(summary, indent=2))
    return 1 if summary["failed_required"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
