# Data card: Bangkok condo asking prices (collected 23 Nov 2026)

## Summary
| Item | Value |
|---|---|
| What one row is | One condo unit listed for sale |
| Main source | Livinginsider.com zone pages "condo for sale" (คอนโด ขาย) |
| Second source | Legal Execution Department auction search (asset.led.go.th), Bangkok condo units, used only for comparison |
| Collection date | 23 November 2026 (2569), one snapshot |
| Collected by | The project team, by opening each page in Chrome and saving it by hand ("Webpage, Complete") |
| Raw pages | 115 Livinginsider pages (38 zones) + 5 LED result pages |
| Raw listings | 5,624 rows (5,468 unique listing IDs) |
| Clean listings | 4,918 rows |
| Used for modelling | 3,271 rows (at most 2 pages per zone), 38 zones, 971 projects |
| Target | Asking price in THB (not the final sale price) |

## Why the data was collected by hand
- No ready-made or pre-labelled dataset was used.
- Livinginsider answers scripted requests with an AWS WAF challenge, and the LED search requires a verification code. We did not bypass either. Every page was opened and saved by a person in a normal browser.
- Only listing facts are kept. Listing titles are dropped because sellers often put phone numbers or LINE IDs in them. Raw HTML is kept locally and is not published.

## Collection protocol
1. **Zones:** 38 Bangkok "condo for sale" zone pages were picked from the site's sitemap. Zones outside Bangkok were excluded (Nonthaburi, Pathum Thani, Samut Prakan).
2. **Pages per zone:** pages 1 and 2 were saved for each zone, which are the newest listings. Some zones have only one page, because the site has fewer listings there:
   - Yaowarat: 9 listings
   - Dao Khanong / Bang Bon: 44 listings
3. **Zone 01:** Sukhumvit–Asok–Thonglor was saved to 38 pages during a first pilot run. For a fair comparison between zones, the models use only pages 1–2 of every zone. The extra pages are kept for analysis.

## Fields
| Field | How it was obtained |
|---|---|
| listing_id | Number at the end of the listing URL |
| zone_no, zone_slug | URL saved inside each HTML file |
| project_name | Project name shown on the listing card |
| bedrooms, bathrooms, floor, area | Read from the four icon fields of each card. Each icon is identified by its SVG shape, because the site shows no text labels |
| price_thb | Card price. Formats such as "฿ 8M" or "฿ 19.5M" are converted to baht |
| badges | Listing tags: ขาย, ขายและเช่า, ขายดาวน์, มือ 1 |

## Labels we created
| Label | Rule |
|---|---|
| district_th | Hand-made table `data/labels/zone_to_district.csv` maps each zone to its main district (เขต). Each label carries a confidence level: 16 high, 18 medium, 4 low. Zones that span several districts are noted |
| room_type | studio (0 bedrooms, "สตูดิโอ"), 1br, 2br, 3br_plus |
| is_new_unit | Card has the "มือ 1" tag |
| is_down_payment | Card has the "ขายดาวน์" tag. These rows are removed, because the price is a down-payment transfer, not a full price |
| floor | Exact floor if shown. For a floor band such as "11-20", the midpoint is used; 55.6% of rows have a band |
| location tier | K-Means cluster of the zone, computed from training data only (see method) |

## Cleaning steps (from `reports/cleaning_log.json`)
| Step | Rows removed | Rows left |
|---|---|---|
| Start | 0 | 5,624 |
| Same listing ID on several pages | 156 | 5,468 |
| Same unit re-posted (same project, area, price, floor, bedrooms) | 319 | 5,149 |
| Down-payment listing (price is not the full price) | 93 | 5,056 |
| Missing a required field | 15 | 5,041 |
| Area outside 15–500 sqm | 14 | 5,027 |
| Price below 500,000 THB | 5 | 5,022 |
| Price per sqm outlier (1.5 × IQR on log scale, within each zone) | 104 | 4,918 |

The outlier fence is computed inside each zone, because price per sqm differs several times between central and suburban zones.

## Dataset profile (modelling set, 3,271 rows)
- **Room types:**
  - 1br: 2,017
  - 2br: 850
  - studio: 244
  - 3br_plus: 160
- **Prices:**
  - median asking price: 3.35 million THB
  - median price per sqm: 86,711 THB
- **New units:** 8.9% of listings are "มือ 1".

## LED auction data (comparison only)
- 150 Bangkok condo units, from 112 court cases across 20 districts.
- The price is the court appraisal (ราคาประเมิน).
- The size column is headed "ตร.วา", but for condo units the values (median 30) match square metres, so they are treated as square metres. This is our assumption.
- In the 10 districts with at least 3 units in both sources, the median appraisal per sqm is about **73%** of the median asking price per sqm. The ratio ranges from 0.23 to 1.05.

## Known limitations and biases
- **Asking price, not sale price:** sellers usually ask more than they finally get.
- **One snapshot:** all data is from 23 Nov 2026, so there is no time trend.
- **Newest listings only:** pages 1–2 show the newest listings, which may differ from older ones.
- **Agent listings:** agents post many units, and some identical units with different prices remain after de-duplication. In the random split, 36.6% of test units have a unit with the same project, bedrooms and rounded area in training. This is why we also report a split by project, where test buildings are never seen in training.
- **District labels are coarse:** they are per zone, not per building, and 4 zones have low confidence.
- **Missing features:** distance to BTS/MRT, building age and number of units are not on the listing cards, so they are not in the data.
