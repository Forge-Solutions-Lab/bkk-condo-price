#!/usr/bin/env bash
# Opens one batch of Livinginsider zone pages (zones 02-38, pages 1 and 2) in Google Chrome,
# so the person collecting data can save each tab by hand. Usage: bash scripts/open_zone_tabs.sh <batch 1-8>
set -euo pipefail

batch="${1:?give a batch number from 1 to 8}"
batch_size=10
labels="$(dirname "$0")/../data/labels/zone_to_district.csv"

urls=()
while IFS=, read -r zone_no zone_slug _; do
  [[ "$zone_no" == "zone_no" || "$zone_no" == "1" ]] && continue
  urls+=("https://www.livinginsider.com/zone/${zone_slug}-condo-buysell")
  urls+=("https://www.livinginsider.com/zone/${zone_slug}-condo-buysell/2")
done < "$labels"

start=$(( (batch - 1) * batch_size ))
if (( start >= ${#urls[@]} )); then
  echo "batch $batch is empty: there are ${#urls[@]} pages, $(( (${#urls[@]} + batch_size - 1) / batch_size )) batches"
  exit 1
fi
selection=("${urls[@]:start:batch_size}")
echo "opening batch $batch: pages $(( start + 1 ))-$(( start + ${#selection[@]} )) of ${#urls[@]}"
open -a "Google Chrome" "${selection[@]}"
