"""Tests that check the models honestly: no leakage, sensible answers, and accuracy gates.

The gates read reports/benchmark.json (python -m src.evaluate.benchmark). They are targets the
current models do not all meet yet; a failing gate is the honest state, not something to relax.
"""

import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.clean.clean import unlink_area_names
from src.evaluate import benchmark as bm
from src.features.build import BrandPremium, KnownBuildingPrice, ProjectPriceLevels, drop_building_outliers, project_brand

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "reports" / "benchmark.json"

# accuracy gates (grouped CV on the development rows, mean over seeds)
UNSEEN_MAX_MAPE = 18.0
UNSEEN_MIN_WITHIN_20 = 67.0
LARGE_UNSEEN_MIN_WITHIN_20 = 62.0
MAX_ZONE_BIAS = 15.0


def listing(project, area, floor, price, zone=1, bedrooms=1, listing_id=None, is_new=False):
    return {"project_name": project, "area_sqm": area, "floor": floor, "price_thb": price, "zone_no": zone,
            "price_per_sqm": price / area, "bedrooms": bedrooms, "is_new_unit": is_new,
            "listing_id": listing_id or f"{project}-{area}-{floor}-{price}"}


# ---------- leakage ----------

def test_twins_are_grouped_and_different_units_are_not():
    rows = pd.DataFrame([
        listing("A", 30.0, 5, 3_000_000), listing("A", 30.5, 5, 3_050_000),  # same unit posted twice
        listing("A", 30.0, 6, 3_000_000),  # other floor: a different unit
        listing("A", 45.0, 5, 3_000_000),  # other size
        listing("B", 30.0, 5, 3_000_000),  # other building
        listing("A", 30.0, 5, 3_000_000, bedrooms=0),  # other layout
        listing("", 30.0, 5, 3_000_000), listing("", 30.0, 5, 3_000_000, listing_id="x"),  # no building
    ])
    groups = bm.twin_groups(rows)
    assert groups[0] == groups[1]
    assert len({groups[0], groups[2], groups[3], groups[4], groups[5], groups[6], groups[7]}) == 7


def test_group_folds_never_split_a_group():
    groups = np.repeat(np.arange(40), 3)
    folds = bm.group_folds(groups, 5, seed=1)
    for g in np.unique(groups):
        assert len(set(folds[groups == g])) == 1
    assert set(folds) == set(range(5))


@pytest.fixture(scope="module")
def listings():
    return bm.load_listings()


def test_holdout_is_disjoint_from_development(listings):
    dev, unseen, known = bm.split_holdout(listings)
    assert not set(dev["building"]) & set(unseen["building"])
    assert not set(dev["twin_group"]) & set(known["twin_group"])
    assert not set(unseen["twin_group"]) & set(known["twin_group"])
    assert len(dev) + len(unseen) + len(known) == len(listings)


def test_project_level_never_sees_its_own_price():
    rows = pd.DataFrame([listing("A", 30, 5, 3_000_000), listing("A", 30, 6, 6_000_000)])
    levels = ProjectPriceLevels.fit(rows).transform(rows, leave_one_out=True)
    # each row only sees the other row's price per sqm
    assert np.allclose(np.exp(levels["project_log_price_per_sqm"]), [200_000, 100_000])


def test_holdout_does_not_move_when_rows_are_removed(listings):
    _, unseen, known = bm.split_holdout(listings)
    _, unseen_less, known_less = bm.split_holdout(listings.iloc[::2].reset_index(drop=True))
    assert set(unseen_less["listing_id"]) <= set(unseen["listing_id"])
    assert set(known_less["listing_id"]) <= set(known["listing_id"]) | set(unseen["listing_id"])


def test_extra_pages_never_add_history_to_a_tested_unseen_building(listings):
    _, unseen, _ = bm.split_holdout(listings)
    extra = bm.usable_extra(bm.load_extra_rows(), unseen)
    assert not set(extra["building"]) & set(unseen["building"])


def test_area_name_is_not_a_building():
    rows = pd.DataFrame([listing("มีนบุรี-ร่มเกล้า", 30, 5, 2_000_000), listing("Real Tower", 30, 5, 2_000_000)])
    rows["zone_name_th"] = "มีนบุรี ร่มเกล้า"
    cleaned = unlink_area_names(rows, [])
    assert cleaned["project_name"].tolist() == ["", "Real Tower"]
    assert ProjectPriceLevels.fit(cleaned).counts.index.tolist() == ["real tower"]


# ---------- known-building model ----------

def known_rows():
    rows = [listing("A", 30, f, 3_000_000 + 10_000 * f, listing_id=f"a{f}") for f in range(1, 9)]
    rows += [listing("B", 30, 5, 6_000_000, listing_id="b1")]
    rows += [listing("A", 30, 9, 9_000_000, listing_id="typo")]  # 3x the others: a typing error
    return pd.DataFrame(rows)


def test_a_typing_error_does_not_enter_the_building_history():
    history = drop_building_outliers(known_rows())
    assert "typo" not in set(history["listing_id"])
    assert len(history) == 9


def test_known_model_never_sees_its_own_listing():
    rows = known_rows()
    model = KnownBuildingPrice.fit(rows, np.log(rows["price_per_sqm"].to_numpy()))
    stats = model.building_stats(rows, leave_one_out=True)
    # B has one listing: without its own row it has no history
    assert stats.loc[rows["listing_id"] == "b1", "n"].item() == 0
    assert stats.loc[rows["listing_id"] == "a1", "n"].item() == 7


def test_few_listings_lean_on_the_prior_more():
    rows = known_rows()
    model = KnownBuildingPrice.fit(rows, np.full(len(rows), np.log(100_000)))
    model.W_adjust = np.zeros_like(model.W_adjust)  # look at the shrunk level only
    prior = np.log(50_000.0)
    unit = pd.DataFrame([listing("A", 30, 5, 1, listing_id="new1"), listing("B", 30, 5, 1, listing_id="new2")])
    price_per_sqm = model.predict(unit, np.full(2, prior)) / 30
    a_level, b_level = 101_500, 200_000
    # A (8 listings) stays near its own level; B (1 listing) moves a third of the way to the prior
    assert abs(price_per_sqm[0] - a_level) / a_level < 0.1
    assert price_per_sqm[1] < b_level * 0.85


def test_unknown_building_gets_no_known_price():
    rows = known_rows()
    model = KnownBuildingPrice.fit(rows, np.log(rows["price_per_sqm"].to_numpy()))
    unit = pd.DataFrame([listing("C", 30, 5, 1), listing("", 30, 5, 1)])
    assert np.isnan(model.predict(unit, np.zeros(2))).all()


def test_brand_is_the_developer_series():
    names = pd.Series(["ไอดีโอ คิว สุขุมวิท 36", "เดอะ ไลน์ สุขุมวิท 101", "ลุมพินี เพลส พระราม 8", "", None])
    assert project_brand(names).tolist() == ["ไอดีโอ", "เดอะ ไลน์", "ลุมพินี", "", ""]


def test_brand_premium_never_uses_the_buildings_own_price():
    rows = pd.DataFrame([listing("ไอดีโอ A", 30, 5, 6_000_000), listing("ไอดีโอ B", 30, 5, 3_000_000),
                         listing("ลุมพินี C", 30, 5, 3_000_000)])
    neighbourhood = np.log(np.full(3, 100_000.0))
    premium = BrandPremium.fit(rows, neighbourhood, prior_buildings=0.0).transform(rows)
    # A sees only B (100k/sqm, premium 0); B sees only A (200k/sqm, premium log 2); C has no other building
    assert np.allclose(premium, [0.0, np.log(2.0), 0.0])
    unseen = pd.DataFrame([listing("ไอดีโอ D", 30, 5, 1)])
    assert np.isclose(BrandPremium.fit(rows, neighbourhood, prior_buildings=0.0).transform(unseen)[0], np.log(2.0) / 2)


def test_unseen_building_has_no_project_history(listings):
    dev, unseen, _ = bm.split_holdout(listings)
    levels = ProjectPriceLevels.fit(dev).transform(unseen, leave_one_out=False)
    assert levels["has_project_history"].sum() == 0


# ---------- sensible answers from the web app ----------

@pytest.fixture(scope="module")
def app():
    spec = importlib.util.spec_from_file_location("app", ROOT / "app" / "streamlit_app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bundle():
    with (ROOT / "models" / "model_bundle.pkl").open("rb") as handle:
        return pickle.load(handle)


def price(app, bundle, area=35.0, bedrooms=1, bathrooms=1, project=None, zone=1, is_new=False):
    room = app.build_input(bundle, zone, area, bedrooms, bathrooms, 10, is_new, project, 2016, 0.5)
    return app.predict(bundle, room)["price"]


def known_project(bundle):
    counts = bundle["reference_rows"]["project_name"].value_counts()
    return counts.index[0]


@pytest.mark.parametrize("project", [None, "known"])
def test_bigger_unit_costs_more(app, bundle, project):
    name = known_project(bundle) if project == "known" else None
    prices = [price(app, bundle, area=a, bedrooms=b, project=name) for a, b in ((28, 1), (45, 1), (70, 2), (120, 3))]
    assert all(later > earlier for earlier, later in zip(prices, prices[1:]))


@pytest.mark.parametrize("area,bedrooms,bathrooms", [(20, 0, 1), (35, 1, 1), (500, 5, 5), (40, 2, 4)])
def test_edge_inputs_give_a_plausible_price_per_sqm(app, bundle, area, bedrooms, bathrooms):
    for zone in bundle["zone_table"]["zone_no"]:
        value = price(app, bundle, area=area, bedrooms=bedrooms, bathrooms=bathrooms, zone=int(zone))
        assert np.isfinite(value)
        # listings in the data run from about 20,000 to 400,000 baht per sqm
        assert 15_000 <= value / area <= 450_000


# ---------- accuracy gates ----------

@pytest.fixture(scope="module")
def cv():
    if not BENCHMARK.exists():
        pytest.fail("reports/benchmark.json missing: run python -m src.evaluate.benchmark")
    return json.loads(BENCHMARK.read_text(encoding="utf-8"))["cv"]


def mean(values, *path):
    for key in path:
        values = values[key]
    return values["mean"]


def test_gate_unseen_building_mape(cv):
    assert mean(cv, "unseen_building", "model", "mape_pct") <= UNSEEN_MAX_MAPE


def test_gate_unseen_building_within_20(cv):
    assert mean(cv, "unseen_building", "model", "within_20pct") >= UNSEEN_MIN_WITHIN_20


def test_gate_large_units_on_unseen_buildings(cv):
    assert mean(cv, "unseen_building", "large_units", "within_20pct") >= LARGE_UNSEEN_MIN_WITHIN_20


@pytest.mark.parametrize("case", ["unseen_building", "known_building"])
def test_gate_model_beats_its_baseline(cv, case):
    assert mean(cv, case, "model", "mape_pct") < mean(cv, case, "baseline", "mape_pct")


@pytest.mark.parametrize("case", ["unseen_building", "known_building"])
def test_gate_no_zone_is_priced_far_off_on_average(cv, case):
    assert mean(cv, case, "worst_zone_bias_pct") <= MAX_ZONE_BIAS
