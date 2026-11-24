from __future__ import annotations

import argparse
import os
from pathlib import Path
from urllib.parse import quote

import requests


FILES = [
    "dataset/patches_512/features/features_UNI.tar.gz",
    "dataset/patches_512/labels.tar.gz",
    "dataset/patches_512/coords.tar.gz",
    "dataset/patches_512/patch_labels.tar.gz",
    "dataset/splits.csv",
]

REPO_ID = "torchmil/Camelyon16_MIL"
BASE_URL = "https://huggingface.co/datasets/torchmil/Camelyon16_MIL/resolve/main/"
RANGE_SIZE = 8 * 1024 * 1024


def download_resumable(filename: str, target: Path) -> Path:
    """Download one public HF file in fixed byte ranges with resume support."""
    output = target / filename
    partial = output.with_name(output.name + ".part")
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists():
        print(f"already_present={output} bytes={output.stat().st_size}", flush=True)
        return output

    existing = partial.stat().st_size if partial.exists() else 0
    url = BASE_URL + quote(filename, safe="/") + "?download=true"
    print(f"requesting={url}", flush=True)
    if existing:
        print(f"resuming_from={existing}", flush=True)

    written = existing
    total = None
    with partial.open("ab" if existing else "wb") as handle:
        while total is None or written < total:
            end = written + RANGE_SIZE - 1
            response = requests.get(
                url,
                headers={"Range": f"bytes={written}-{end}"},
                stream=True,
                allow_redirects=True,
                timeout=(60, 300),
            )
            try:
                if response.status_code == 416:
                    content_range = response.headers.get("Content-Range", "")
                    if content_range.startswith("bytes */"):
                        total = int(content_range.rsplit("/", 1)[1])
                        if written == total:
                            break
                    response.raise_for_status()
                if response.status_code != 206:
                    raise RuntimeError(
                        f"expected HTTP 206 for byte range, got {response.status_code}"
                    )
                content_range = response.headers.get("Content-Range", "")
                if "/" not in content_range:
                    raise RuntimeError("missing Content-Range in ranged response")
                range_spec, total_text = content_range.split("/", 1)
                total = int(total_text)
                start_text, end_text = range_spec.removeprefix("bytes ").split("-", 1)
                response_start = int(start_text)
                response_end = int(end_text)
                if response_start != written:
                    raise RuntimeError(
                        f"server returned range {response_start}-{response_end}, "
                        f"requested {written}-{end}"
                    )
                for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                    if chunk:
                        handle.write(chunk)
                        written += len(chunk)
                handle.flush()
                print(
                    f"progress={written}/{total} bytes "
                    f"({written / total * 100:.2f}%)",
                    flush=True,
                )
            finally:
                response.close()

    partial.replace(output)
    print(f"downloaded={output} bytes={written}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--proxy", default=None)
    args = parser.parse_args()
    if args.proxy:
        os.environ["HTTP_PROXY"] = args.proxy
        os.environ["HTTPS_PROXY"] = args.proxy
        os.environ["http_proxy"] = args.proxy
        os.environ["https_proxy"] = args.proxy
    target = args.project_root / "data" / "downloads" / "camelyon16_hf"
    target.mkdir(parents=True, exist_ok=True)
    print(f"repo={REPO_ID}")
    print(f"download_dir={target.resolve()}")
    print(f"proxy={args.proxy or os.environ.get('HTTPS_PROXY', 'unset')}")
    for filename in FILES:
        print(f"downloading={filename}", flush=True)
        download_resumable(filename, target)


if __name__ == "__main__":
    main()

