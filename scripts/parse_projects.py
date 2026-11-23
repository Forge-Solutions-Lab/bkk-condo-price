"""Parse saved Livinginsider listing pages (one per condo project) into one row per project.

Each listing page embeds the project's data: location, completion year, number of units,
floors and buildings, facilities, and nearby places with walking distance. The listing's
own text is not kept.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path

LOG = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "projects"
OUTPUT_CSV = PROJECT_ROOT / "data" / "interim" / "projects.csv"

FLIGHT_PATTERN = re.compile(r'self\.__next_f\.push\(\[1,"(.*?)"\]\)', re.S)
SAVED_FROM_PATTERN = re.compile(r"saved from url=\(\d+\)(\S+)")
DISTRICT_PATTERN = re.compile(r"เขต\s*([ก-๙]+)")
FACILITY_FLAGS = {"has_pool": "สระว่ายน้ำ", "has_gym": "ฟิตเนส", "has_parking": "ที่จอดรถ"}
COLUMNS = [
    "project_id", "project_name", "project_slug", "latitude", "longitude", "district_th", "complete_year",
    "total_unit", "total_floor", "total_building", "facility_count", *FACILITY_FLAGS,
    "nearest_rail_km", "nearest_rail_line", "nearest_mall_km", "nearest_hospital_km", "nearest_school_km",
    "source_url", "source_file",
]


def page_data(html: str) -> str:
    # The page ships its data as escaped JSON strings; decoding them gives plain JSON text.
    parts = []
    for chunk in FLIGHT_PATTERN.findall(html):
        try:
            parts.append(json.loads('"' + chunk + '"'))
        except json.JSONDecodeError:
            parts.append(chunk)
    return "".join(parts)


def json_object_at(text: str, start: int) -> dict | None:
    """Read the JSON object that begins at text[start] by matching braces."""
    depth, in_string, escaped = 0, False, False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            escaped = char == "\\" and not escaped
            if char == '"' and not escaped:
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:index + 1])
                except json.JSONDecodeError:
                    return None
    return None


def find_object(text: str, key: str) -> dict | None:
    position = text.find(f'"{key}":{{')
    return json_object_at(text, position + len(key) + 3) if position >= 0 else None


def nearest(items: list[dict], *categories: str) -> dict | None:
    allowed = set(categories)
    candidates = [item for item in items if item.get("category") in allowed and item.get("distance") is not None]
    return min(candidates, key=lambda item: item["distance"]) if candidates else None


def rail_line(icon: str) -> str:
    # icons look like "icon_tran_mrt_pink.svg" or "icon_tran_bts.svg"
    match = re.search(r"icon_tran_([a-z_]+)\.svg", icon or "")
    return match.group(1) if match else ""


def parse_file(path: Path) -> dict | None:
    html = path.read_text(encoding="utf-8", errors="ignore")
    data = page_data(html)
    identity = find_object(data, "identity")
    overview = find_object(data, "overview") or {}
    location = find_object(data, "location") or {}
    amenities = find_object(data, "amenities") or {}
    if identity is None:
        LOG.warning("no project data in %s", path.name)
        return None
    # the project's own "location" (with the address text) follows its "overview" block;
    # the first "location" in the page is the listing's map pin with the coordinates
    overview_at = data.find('"overview":')
    project_address = find_object(data[overview_at:], "location") if overview_at >= 0 else None
    district = DISTRICT_PATTERN.search(json.dumps(project_address or {}, ensure_ascii=False))
    facilities = [item.get("name", "") for item in amenities.get("facilities", {}).get("items", [])]
    nearby = amenities.get("nearby", {}).get("items", [])
    rail = nearest(nearby, "transit")
    saved_from = SAVED_FROM_PATTERN.search(html[:5000])
    return {
        "project_id": identity.get("id"),
        "project_name": identity.get("name"),
        "project_slug": identity.get("slug"),
        "latitude": location.get("latitude"),
        "longitude": location.get("longitude"),
        "district_th": district.group(1) if district else "",
        "complete_year": overview.get("complete_year"),
        "total_unit": overview.get("total_unit"),
        "total_floor": overview.get("total_floor"),
        "total_building": overview.get("total_building"),
        "facility_count": len(facilities),
        **{flag: int(any(word in name for name in facilities)) for flag, word in FACILITY_FLAGS.items()},
        "nearest_rail_km": rail["distance"] if rail else None,
        "nearest_rail_line": rail_line(rail["icon"]) if rail else "",
        "nearest_mall_km": (nearest(nearby, "mall") or {}).get("distance"),
        "nearest_hospital_km": (nearest(nearby, "hospital") or {}).get("distance"),
        "nearest_school_km": (nearest(nearby, "school", "academy") or {}).get("distance"),
        "source_url": saved_from.group(1) if saved_from else "",
        "source_file": path.name,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rows = [row for path in sorted(RAW_DIR.glob("*.html")) if (row := parse_file(path))]
    unique = {row["project_id"]: row for row in rows}
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(unique.values())
    LOG.info("pages=%s projects=%s -> %s", len(rows), len(unique), OUTPUT_CSV)


if __name__ == "__main__":
    main()
