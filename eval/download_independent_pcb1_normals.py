"""Acquire a small, rights-cleared external HC-SR04 normal-image audit set.

These Wikimedia Commons photographs were not sourced from Amazon VisA. They
support an independent acquisition/normal-board audit only; they do not provide
labelled defective boards and therefore cannot measure defect escape rate.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "data" / "real_tests" / "pcb1_commons"
API = "https://commons.wikimedia.org/w/api.php"
TITLES = (
    "File:HC SR04 Ultrasonic Module.jpg",
    "File:HC SR04 Ultrasonic sensor 1480319 20 21 HDR Enhancer.jpg",
    "File:HC SR04 Ultrasonic sensor 1480322 3 4 HDR Enhancer.jpg",
    "File:HC-SR04 ultrasonic distance sensor module pcb.jpg",
    "File:HC04 Ultrasonic module 01.jpg",
    "File:SparkFun HC-SR04 Ultrasonic-Sensor 13959-01a.jpg",
)
ALLOWED_LICENSES = {"CC BY 2.0", "CC BY-SA 3.0", "CC BY-SA 4.0"}
COHORT_BY_TITLE = {
    TITLES[0]: "primary_front_blue_hc_sr04",
    TITLES[1]: "primary_front_blue_hc_sr04",
    TITLES[2]: "primary_front_blue_hc_sr04",
    TITLES[3]: "supplemental_back_view",
    TITLES[4]: "primary_front_blue_hc_sr04",
    TITLES[5]: "supplemental_red_product_variant",
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def fetch(url: str, attempts: int = 5) -> bytes:
    last_error: Exception | None = None
    for attempt in range(attempts):
        request = urllib.request.Request(url, headers={"User-Agent": "SureInspect-evaluation/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"Could not download {url}: {last_error}") from last_error


def query_pages() -> list[dict[str, Any]]:
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "titles": "|".join(TITLES),
            "prop": "info|imageinfo",
            "inprop": "url",
            "iiprop": "url|sha1|size|timestamp|extmetadata",
            "format": "json",
            "formatversion": "2",
        }
    )
    payload = json.loads(fetch(f"{API}?{params}"))
    pages = payload.get("query", {}).get("pages", [])
    by_title = {page["title"]: page for page in pages if "missing" not in page}
    missing = sorted(set(TITLES) - set(by_title))
    if missing:
        raise RuntimeError(f"Wikimedia Commons files are missing: {missing}")
    return [by_title[title] for title in TITLES]


def metadata_value(image_info: dict[str, Any], key: str) -> str:
    return str(image_info.get("extmetadata", {}).get(key, {}).get("value", ""))


def run() -> int:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for index, page in enumerate(query_pages(), 1):
        image_info = page["imageinfo"][0]
        license_name = metadata_value(image_info, "LicenseShortName")
        if license_name not in ALLOWED_LICENSES:
            raise RuntimeError(f"Unsupported license for {page['title']}: {license_name}")
        destination = DESTINATION / f"normal_{index:02d}.jpg"
        payload = destination.read_bytes() if destination.is_file() else fetch(image_info["url"])
        if not payload.startswith(b"\xff\xd8"):
            raise RuntimeError(f"Expected JPEG data for {page['title']}")
        if not destination.is_file():
            temporary = destination.with_suffix(".jpg.part")
            temporary.write_bytes(payload)
            os.replace(temporary, destination)
        records.append(
            {
                "path": destination.relative_to(ROOT).as_posix(),
                "label": "normal",
                "cohort": COHORT_BY_TITLE[page["title"]],
                "title": page["title"],
                "description_url": page["canonicalurl"],
                "download_url": image_info["url"],
                "author": metadata_value(image_info, "Artist"),
                "license": license_name,
                "license_url": metadata_value(image_info, "LicenseUrl"),
                "source_timestamp": image_info.get("timestamp"),
                "width": image_info["width"],
                "height": image_info["height"],
                "bytes": len(payload),
                "wikimedia_sha1": image_info.get("sha1"),
                "sha256": sha256_bytes(payload),
            }
        )
        print(f"Downloaded {index}/{len(TITLES)}: {page['title']}", flush=True)
        time.sleep(1)

    manifest = {
        "schema_version": "sureinspect.external-pcb1-normal-acquisition.v1",
        "source": "Wikimedia Commons",
        "source_category": "https://commons.wikimedia.org/wiki/Category:Ultrasonic_range_finding",
        "scope": "external_normal_only_acquisition_audit",
        "limitations": (
            "Visible-normal labels only; electrical function is unverified and no labelled defects are present. "
            "This set cannot estimate defect escape rate or complete PCB1 task accuracy."
        ),
        "records": records,
    }
    manifest_path = DESTINATION / "acquisition_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"External PCB1 normal set ready: {len(records)} images")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
