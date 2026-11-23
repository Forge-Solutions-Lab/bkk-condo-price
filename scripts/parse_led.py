"""Parse saved Legal Execution Department (LED) auction search pages for Bangkok condo units.

The size column header says square wa, but for condo units ("ห้องชุด") the values
(about 20-40) match unit sizes in square metres, so they are treated as square metres.
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path

from bs4 import BeautifulSoup

LOG = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "led"
OUTPUT_CSV = PROJECT_ROOT / "data" / "interim" / "led_raw.csv"
COLUMNS = ["case_no", "asset_type", "area_sqm", "appraisal_thb", "subdistrict_th", "district_th", "province_th", "source_file"]
EXPECTED_CELLS = 11


def normalise_district(text: str) -> str:
    # older records add the former district in brackets, for example "จตุจักร (บางซื่อ)"
    return re.sub(r"\s*\(.*\)\s*$", "", text).strip()


def parse_file(path: Path) -> list[dict[str, str]]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    rows = []
    for row in soup.select("table tr"):
        cells = [cell.get_text(" ", strip=True) for cell in row.find_all("td")]
        if len(cells) != EXPECTED_CELLS:
            continue
        _, _, case_no, asset_type, _, _, size, appraisal, subdistrict, district, province = cells
        rows.append({
            "case_no": case_no,
            "asset_type": asset_type,
            "area_sqm": size.replace(",", ""),
            "appraisal_thb": appraisal.replace(",", ""),
            "subdistrict_th": subdistrict,
            "district_th": normalise_district(district),
            "province_th": province,
            "source_file": path.name,
        })
    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    records = []
    for path in sorted(RAW_DIR.glob("led_p*.html")):
        parsed = parse_file(path)
        LOG.info("%s: %s rows", path.name, len(parsed))
        records.extend(parsed)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(records)
    LOG.info("LED rows=%s -> %s", len(records), OUTPUT_CSV)


if __name__ == "__main__":
    main()
