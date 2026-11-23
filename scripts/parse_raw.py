"""Parse saved Livinginsider zone pages (HTML) into one CSV of condo-for-sale listings.

Only listing facts are kept. Listing titles are dropped on purpose because sellers
often put phone numbers or LINE IDs in them.
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from bs4 import BeautifulSoup, Tag

LOG = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "livinginsider"
OUTPUT_CSV = PROJECT_ROOT / "data" / "interim" / "listings_raw.csv"
ZONE_LABELS_CSV = PROJECT_ROOT / "data" / "labels" / "zone_to_district.csv"

FILE_NAME_PATTERN = re.compile(r"^(?P<zone_no>\d+)_(?P<zone_slug>.+)_p(?P<page>\d+)\.html$")
SAVED_FROM_PATTERN = re.compile(
    r"saved from url=\(\d+\)https://www\.livinginsider\.com/zone/(?P<slug>[a-z0-9-]+?)-condo-buysell(?:/(?P<page>\d+))?"
)
LISTING_ID_PATTERN = re.compile(r"-(\d+)$")
NUMBER_PATTERN = re.compile(r"[\d,.]+")

# The site shows bedrooms, bathrooms, floor and area as icon + number, without text labels.
# Each icon is an inline SVG, so the start of its path data identifies the field.
ICON_SIGNATURES = {
    "M20.4995 12.4561": "bedrooms",
    "M18.75 13.3725": "bathrooms",
    "M6.2749 18.8751": "floor",
    "M14.3257 11.5": "area",
}


@dataclass
class Listing:
    listing_id: str
    zone_no: int
    zone_slug: str
    page: int
    project_name: str
    bedrooms_raw: str
    bathrooms_raw: str
    floor_raw: str
    area_raw: str
    price_raw: str
    badges: str
    listing_url: str
    source_file: str


def is_listing_card(div: Tag) -> bool:
    classes = div.get("class") or []
    return (
        "rounded-2xl" in classes
        and "border-light-gray" in classes
        and div.find("a", href=re.compile(r"/detail/")) is not None
    )


def read_icon_fields(card: Tag) -> dict[str, str]:
    values: dict[str, str] = {}
    for span in card.select("span.leading-\\[24px\\]"):
        path = span.find("path")
        if path is None:
            continue
        for signature, field_name in ICON_SIGNATURES.items():
            if path.get("d", "").startswith(signature):
                values[field_name] = span.get_text(" ", strip=True)
    return values


def parse_card(card: Tag, zone_no: int, zone_slug: str, page: int, source_file: str) -> Listing | None:
    link = card.find("a", href=re.compile(r"/detail/"))
    url = link["href"].split("?")[0]
    listing_id_match = LISTING_ID_PATTERN.search(url)
    if listing_id_match is None:
        LOG.warning("skip card without listing id: %s", url)
        return None
    project = card.select_one("span.line-clamp-1")
    price = card.select_one("p.h-7")
    badges = [badge.get_text(strip=True) for badge in card.select("span.inline-flex.rounded")]
    icon_fields = read_icon_fields(card)
    return Listing(
        listing_id=listing_id_match.group(1),
        zone_no=zone_no,
        zone_slug=zone_slug,
        page=page,
        project_name=project.get_text(strip=True) if project else "",
        bedrooms_raw=icon_fields.get("bedrooms", ""),
        bathrooms_raw=icon_fields.get("bathrooms", ""),
        floor_raw=icon_fields.get("floor", ""),
        area_raw=icon_fields.get("area", ""),
        price_raw=price.get_text(strip=True) if price else "",
        badges="|".join(badges),
        listing_url=url,
        source_file=source_file,
    )


def load_zone_numbers() -> dict[str, int]:
    with ZONE_LABELS_CSV.open(encoding="utf-8") as handle:
        return {row["zone_slug"]: int(row["zone_no"]) for row in csv.DictReader(handle)}


def page_identity(path: Path, text: str, zone_numbers: dict[str, int]) -> tuple[int, str, int] | None:
    # Chrome writes the page URL into a "saved from url" comment, so pages can be saved
    # with any file name; the file-name pattern is only a fallback.
    saved_from = SAVED_FROM_PATTERN.search(text[:5000])
    if saved_from and saved_from["slug"] in zone_numbers:
        return zone_numbers[saved_from["slug"]], saved_from["slug"], int(saved_from["page"] or 1)
    match = FILE_NAME_PATTERN.match(path.name)
    if match:
        return int(match["zone_no"]), match["zone_slug"], int(match["page"])
    return None


def parse_file(path: Path, zone_numbers: dict[str, int]) -> list[Listing]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    identity = page_identity(path, text, zone_numbers)
    if identity is None:
        LOG.warning("skip file that is not a known Livinginsider zone page: %s", path.name)
        return []
    zone_no, zone_slug, page = identity
    soup = BeautifulSoup(text, "html.parser")
    cards = [div for div in soup.find_all("div") if is_listing_card(div)]
    listings = [parse_card(card, zone_no, zone_slug, page, path.name) for card in cards]
    return [listing for listing in listings if listing is not None]


def write_csv(listings: list[Listing], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[field.name for field in fields(Listing)])
        writer.writeheader()
        writer.writerows(asdict(listing) for listing in listings)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    files = sorted(RAW_DIR.glob("*.html"))
    zone_numbers = load_zone_numbers()
    listings: list[Listing] = []
    for path in files:
        parsed = parse_file(path, zone_numbers)
        LOG.info("%s: %s listings", path.name, len(parsed))
        listings.extend(parsed)
    write_csv(listings, OUTPUT_CSV)
    LOG.info("files=%s listings=%s -> %s", len(files), len(listings), OUTPUT_CSV)


if __name__ == "__main__":
    main()
