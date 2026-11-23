"""Unit tests for listing parsers, deduplication, and cleaning rules in src/clean/clean.py."""

from __future__ import annotations

import math

import pytest

from src.clean.clean import parse_area, parse_bedrooms, parse_floor, parse_price


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("฿5,250,000", 5_250_000.0),
        ("3.5M", 3_500_000.0),
        ("฿ 4.25 m", 4_250_000.0),
        ("850K", 850_000.0),
        ("1200000", 1_200_000.0),
    ],
)
def test_parse_price_valid_formats(raw: str, expected: float) -> None:
    assert parse_price(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["", "สอบถามราคา", "฿N/A", "5.2B"])
def test_parse_price_invalid_returns_nan(raw: str) -> None:
    assert math.isnan(parse_price(raw))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("35 ตร.ม.", 35.0),
        ("1,250.5 sqm", 1250.5),
        ("42.75", 42.75),
    ],
)
def test_parse_area_extracts_number(raw: str, expected: float) -> None:
    assert parse_area(raw) == pytest.approx(expected)


def test_parse_area_invalid_returns_nan() -> None:
    assert math.isnan(parse_area("ไม่ระบุ"))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("สตูดิโอ", 0.0),
        (" 1 ", 1.0),
        ("3", 3.0),
    ],
)
def test_parse_bedrooms_handles_studio_and_digits(raw: str, expected: float) -> None:
    assert parse_bedrooms(raw) == expected


@pytest.mark.parametrize("raw", ["", "2 ห้อง", "studio"])
def test_parse_bedrooms_invalid_returns_nan(raw: str) -> None:
    assert math.isnan(parse_bedrooms(raw))


def test_parse_floor_exact_and_range() -> None:
    assert parse_floor("ชั้น 15") == (15.0, False)
    assert parse_floor("11-20") == (15.5, True)
    val, is_range = parse_floor("ไม่ระบุ")
    assert math.isnan(val) and is_range is False
