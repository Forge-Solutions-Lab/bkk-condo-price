"""Automated data quality checks for Bangkok condo listings and project building data.

Reads data/processed/listings_clean.csv and data/interim/projects.csv, runs structural and
domain consistency checks, prints 30 randomly sampled listing links (seed=42) for manual
verification against Livinginsider, and writes a Markdown report to reports/data_quality.md.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LISTINGS_CSV = PROJECT_ROOT / "data" / "processed" / "listings_clean.csv"
RAW_LISTINGS_CSV = PROJECT_ROOT / "data" / "interim" / "listings_raw.csv"
PROJECTS_CSV = PROJECT_ROOT / "data" / "interim" / "projects.csv"
REPORT_MD = PROJECT_ROOT / "reports" / "data_quality.md"

REQUIRED_COLUMNS = ["price_thb", "area_sqm", "bedrooms", "bathrooms"]
PRICE_PER_SQM_TOL = 1.0
MIN_COMPLETE_YEAR = 1980
MAX_COMPLETE_YEAR = 2030
BKK_LAT_BOUNDS = (13.5, 14.0)
BKK_LON_BOUNDS = (100.3, 100.95)
SAMPLE_SIZE = 30
SAMPLE_SEED = 42


@dataclass
class CheckResult:
    name: str
    status: str  # "PASS", "FAIL", "WARN", or "INFO"
    failed_rows: int
    details: str
    issues_table: pd.DataFrame = field(default_factory=pd.DataFrame)


def project_key(names: pd.Series) -> pd.Series:
    return names.fillna("").astype(str).str.strip().str.lower()


def check_no_duplicate_ids(listings: pd.DataFrame) -> CheckResult:
    dups = listings[listings.duplicated("listing_id", keep=False)]
    count = int(listings["listing_id"].duplicated().sum())
    return CheckResult(
        name="1. ไม่มีรหัสประกาศซ้ำ (No duplicate listing_id)",
        status="PASS" if count == 0 else "FAIL",
        failed_rows=count,
        details=f"พบ listing_id ซ้ำ {count} แถว จากทั้งหมด {len(listings):,} แถว",
        issues_table=dups[["listing_id", "zone_name_th", "project_name", "price_thb", "area_sqm"]].head(20),
    )


def check_required_fields(listings: pd.DataFrame) -> CheckResult:
    missing_mask = listings[REQUIRED_COLUMNS].isna().any(axis=1)
    non_positive_mask = (listings["price_thb"] <= 0) | (listings["area_sqm"] <= 0)
    bad_mask = missing_mask | non_positive_mask
    count = int(bad_mask.sum())
    return CheckResult(
        name="2. ทุกแถวมีราคา พื้นที่ ห้องนอน ห้องน้ำ (Required fields present)",
        status="PASS" if count == 0 else "FAIL",
        failed_rows=count,
        details=f"พบแถวที่ขาดค่าหรือราคา/พื้นที่ไม่เป็นบวก {count} แถว",
        issues_table=listings.loc[bad_mask, ["listing_id", "project_name", *REQUIRED_COLUMNS]].head(20),
    )


def check_price_per_sqm(listings: pd.DataFrame) -> CheckResult:
    expected = listings["price_thb"] / listings["area_sqm"]
    diff = (listings["price_per_sqm"] - expected).abs()
    bad_mask = diff.isna() | (diff > PRICE_PER_SQM_TOL)
    count = int(bad_mask.sum())
    max_diff = float(diff.max()) if len(diff) else 0.0
    return CheckResult(
        name="3. ราคาต่อ ตร.ม. ตรงกับ ราคา ÷ พื้นที่ (price_per_sqm consistency)",
        status="PASS" if count == 0 else "FAIL",
        failed_rows=count,
        details=f"พบแถวที่คลาดเคลื่อนเกิน {PRICE_PER_SQM_TOL:g} บาท/ตร.ม. จำนวน {count} แถว (ส่วนต่างสูงสุด {max_diff:.6f})",
        issues_table=listings.loc[bad_mask, ["listing_id", "price_thb", "area_sqm", "price_per_sqm"]].head(20),
    )


def check_bedrooms_area_consistency(listings: pd.DataFrame) -> CheckResult:
    # 3+ bedrooms in under 40 sqm is almost certainly a parsing or seller typing error
    warn_mask = (listings["bedrooms"] >= 3) & (listings["area_sqm"] < 40.0)
    count = int(warn_mask.sum())
    compact_2br = int(((listings["bedrooms"] == 2) & (listings["area_sqm"] < 30.0)).sum())
    detail = f"พบห้อง 3+ ห้องนอนที่พื้นที่ < 40 ตร.ม. จำนวน {count} แถว"
    if compact_2br > 0:
        detail += f" (หมายเหตุ: พบ 2 ห้องนอนพื้นที่ < 30 ตร.ม. {compact_2br} แถว)"
    return CheckResult(
        name="4. ห้องนอนกับพื้นที่สอดคล้องกัน (Bedrooms vs. area consistency)",
        status="PASS" if count == 0 else "WARN",
        failed_rows=count,
        details=detail,
        issues_table=listings.loc[warn_mask, ["listing_id", "project_name", "bedrooms", "area_sqm", "price_thb"]].head(20),
    )


def check_projects_validity(projects: pd.DataFrame) -> CheckResult:
    year = pd.to_numeric(projects["complete_year"], errors="coerce")
    lat = pd.to_numeric(projects["latitude"], errors="coerce")
    lon = pd.to_numeric(projects["longitude"], errors="coerce")

    bad_year = year.isna() | ~year.between(MIN_COMPLETE_YEAR, MAX_COMPLETE_YEAR)
    bad_coord = (
        lat.isna()
        | lon.isna()
        | ~lat.between(*BKK_LAT_BOUNDS)
        | ~lon.between(*BKK_LON_BOUNDS)
    )
    bad_mask = bad_year | bad_coord

    issues = projects.loc[bad_mask, ["project_id", "project_name", "complete_year", "latitude", "longitude"]].copy()
    if not issues.empty:
        reasons = []
        for idx in issues.index:
            parts = []
            if bad_year.loc[idx]:
                parts.append(f"complete_year นอกช่วง {MIN_COMPLETE_YEAR}-{MAX_COMPLETE_YEAR}")
            if bad_coord.loc[idx]:
                parts.append(
                    f"พิกัดนอกกรุงเทพฯ (lat {BKK_LAT_BOUNDS[0]}-{BKK_LAT_BOUNDS[1]}, "
                    f"lon {BKK_LON_BOUNDS[0]}-{BKK_LON_BOUNDS[1]})"
                )
            reasons.append("; ".join(parts))
        issues["reason"] = reasons

    count = int(bad_mask.sum())
    year_bad_count = int(bad_year.sum())
    coord_bad_count = int(bad_coord.sum())
    details = (
        f"ตรวจ {len(projects):,} โครงการ: ปีที่สร้างนอกช่วง {MIN_COMPLETE_YEAR}-{MAX_COMPLETE_YEAR} "
        f"จำนวน {year_bad_count} โครงการ, พิกัดนอกกรุงเทพฯ จำนวน {coord_bad_count} โครงการ"
    )
    return CheckResult(
        name="5. ข้อมูลตึก: ปีที่สร้าง (1980-2030) และพิกัดในกรุงเทพฯ",
        status="PASS" if count == 0 else "FAIL",
        failed_rows=count,
        details=details,
        issues_table=issues,
    )


def compute_zone_coverage(listings: pd.DataFrame, projects: pd.DataFrame) -> tuple[CheckResult, pd.DataFrame]:
    known_projects = set(project_key(projects["project_name"]))
    matched = project_key(listings["project_name"]).isin(known_projects)
    frame = listings.assign(has_project=matched)

    by_zone = (
        frame.groupby(["zone_no", "zone_slug", "zone_name_th"], as_index=False)
        .agg(total_listings=("listing_id", "count"), matched_listings=("has_project", "sum"))
        .sort_values("zone_no")
    )
    by_zone["coverage_pct"] = (by_zone["matched_listings"] / by_zone["total_listings"]) * 100.0

    overall_pct = float(matched.mean() * 100.0) if len(frame) else 0.0
    min_pct = float(by_zone["coverage_pct"].min()) if len(by_zone) else 0.0
    max_pct = float(by_zone["coverage_pct"].max()) if len(by_zone) else 0.0
    details = (
        f"ครอบคลุมรวม {int(matched.sum()):,}/{len(frame):,} ประกาศ ({overall_pct:.1f}%) "
        f"ใน {len(by_zone)} โซน (ต่ำสุด {min_pct:.1f}%, สูงสุด {max_pct:.1f}%)"
    )
    result = CheckResult(
        name="6. ข้อมูลตึกครอบคลุมกี่ % ในแต่ละโซน (Project coverage per zone)",
        status="PASS",
        failed_rows=0,
        details=details,
    )
    return result, by_zone


def _normalise_url(url: object) -> str:
    text = "" if pd.isna(url) else str(url).strip()
    if text.startswith("/"):
        return f"https://www.livinginsider.com{text}"
    return text


def sample_listings_for_manual_check(
    listings: pd.DataFrame,
    raw_listings: pd.DataFrame | None,
    n: int = SAMPLE_SIZE,
    seed: int = SAMPLE_SEED,
) -> pd.DataFrame:
    sample = listings.sample(n=min(n, len(listings)), random_state=seed).copy()
    if raw_listings is not None and "listing_url" in raw_listings.columns:
        url_lookup = raw_listings.drop_duplicates("listing_id").set_index("listing_id")["listing_url"]
        sample["listing_url"] = sample["listing_id"].map(url_lookup).map(_normalise_url)
    else:
        sample["listing_url"] = ""
    return sample.reset_index(drop=True)


def _fmt_number(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        if float(value).is_integer():
            return f"{int(value):,}"
        return f"{float(value):,.2f}"
    return str(value)


def _escape_md(text: object) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def build_report_markdown(
    listings: pd.DataFrame,
    projects: pd.DataFrame,
    checks: list[CheckResult],
    zone_coverage: pd.DataFrame,
    sample: pd.DataFrame,
) -> str:
    passed = sum(1 for c in checks if c.status == "PASS")
    failed_checks = sum(1 for c in checks if c.status == "FAIL")
    warned_checks = sum(1 for c in checks if c.status == "WARN")
    total_bad_rows = sum(c.failed_rows for c in checks)

    lines: list[str] = [
        "# Data Quality Report",
        "",
        f"- **Listings dataset**: `data/processed/listings_clean.csv` ({len(listings):,} แถว)",
        f"- **Projects dataset**: `data/interim/projects.csv` ({len(projects):,} โครงการ)",
        f"- **สรุปผลตรวจ**: ผ่าน **{passed}/{len(checks)}** ข้อ | ไม่ผ่าน **{failed_checks}** ข้อ | แจ้งเตือน **{warned_checks}** ข้อ (แถวที่ไม่ผ่าน/แจ้งเตือนรวม **{total_bad_rows}** แถว)",
        "",
        "## 1. สรุปผลการตรวจคุณภาพข้อมูล (Checks 1–6)",
        "",
        "| ข้อตรวจ | สถานะ | จำนวนแถวที่ไม่ผ่าน/แจ้งเตือน | รายละเอียด |",
        "| :--- | :---: | ---: | :--- |",
    ]

    for check in checks:
        lines.append(
            f"| {_escape_md(check.name)} | **{check.status}** | {check.failed_rows} | {_escape_md(check.details)} |"
        )

    flagged = [c for c in checks if not c.issues_table.empty]
    if flagged:
        lines.extend(["", "## 2. รายละเอียดแถวที่ไม่ผ่านหรือแจ้งเตือน", ""])
        for check in flagged:
            lines.append(f"### {check.name}")
            lines.append("")
            cols = list(check.issues_table.columns)
            lines.append("| " + " | ".join(cols) + " |")
            lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
            for _, row in check.issues_table.iterrows():
                vals = [_escape_md(row[col]) for col in cols]
                lines.append("| " + " | ".join(vals) + " |")
            lines.append("")

    lines.extend([
        "## 3. ความครอบคลุมของข้อมูลตึกรายโซน (Project Coverage by Zone)",
        "",
        "| โซนที่ | โซน | ประกาศที่มีข้อมูลตึก | ประกาศทั้งหมด | ครอบคลุม (%) |",
        "| :---: | :--- | ---: | ---: | ---: |",
    ])
    for _, row in zone_coverage.iterrows():
        lines.append(
            f"| {int(row['zone_no'])} | {_escape_md(row['zone_name_th'])} | "
            f"{int(row['matched_listings']):,} | {int(row['total_listings']):,} | {row['coverage_pct']:.1f}% |"
        )
    total_matched = int(zone_coverage["matched_listings"].sum())
    total_all = int(zone_coverage["total_listings"].sum())
    overall_pct = (total_matched / total_all * 100.0) if total_all else 0.0
    lines.append(f"| **รวม** | **ทั้งหมด ({len(zone_coverage)} โซน)** | **{total_matched:,}** | **{total_all:,}** | **{overall_pct:.1f}%** |")

    lines.extend([
        "",
        f"## 4. สุ่มตรวจเทียบกับหน้าเว็บจริงด้วยมือ ({len(sample)} แถว, seed={SAMPLE_SEED})",
        "",
        "| # | listing_id | โซน | โครงการ | ราคา (บาท) | พื้นที่ (ตร.ม.) | ห้องนอน | ห้องน้ำ | ลิงก์ประกาศ | ตรงกับเว็บ: ใช่/ไม่ใช่ |",
        "| :---: | :---: | :--- | :--- | ---: | ---: | :---: | :---: | :--- | :---: |",
    ])
    for idx, row in sample.iterrows():
        url = str(row.get("listing_url", "")).strip()
        link_cell = f"[เปิดประกาศ]({url})" if url else "-"
        lines.append(
            f"| {idx + 1} | {row['listing_id']} | {_escape_md(row['zone_name_th'])} | "
            f"{_escape_md(row['project_name'])} | {_fmt_number(row['price_thb'])} | "
            f"{_fmt_number(row['area_sqm'])} | {_fmt_number(row['bedrooms'])} | "
            f"{_fmt_number(row['bathrooms'])} | {link_cell} |  |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    listings = pd.read_csv(LISTINGS_CSV)
    projects = pd.read_csv(PROJECTS_CSV)
    raw_listings = pd.read_csv(RAW_LISTINGS_CSV) if RAW_LISTINGS_CSV.exists() else None

    coverage_check, zone_coverage = compute_zone_coverage(listings, projects)
    checks = [
        check_no_duplicate_ids(listings),
        check_required_fields(listings),
        check_price_per_sqm(listings),
        check_bedrooms_area_consistency(listings),
        check_projects_validity(projects),
        coverage_check,
    ]
    sample = sample_listings_for_manual_check(listings, raw_listings)

    report = build_report_markdown(listings, projects, checks, zone_coverage, sample)
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text(report, encoding="utf-8")

    for check in checks:
        print(f"[{check.status}] {check.name}: {check.details}")
    print(f"\nReport written to {REPORT_MD}")
    print(f"\nRandom sample ({len(sample)} listings, seed={SAMPLE_SEED}) for manual web verification:")
    for idx, row in sample.iterrows():
        print(
            f"{idx + 1:2d}. id={row['listing_id']} | {row['project_name']} | "
            f"{_fmt_number(row['price_thb'])} THB | {_fmt_number(row['area_sqm'])} sqm | {row['listing_url']}"
        )


if __name__ == "__main__":
    main()
