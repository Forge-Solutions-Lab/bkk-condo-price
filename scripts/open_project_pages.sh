#!/usr/bin/env bash
# Opens one batch of listing pages (one listing per condo project) in Google Chrome, so a
# person can save each tab by hand. Each listing page carries the project's location,
# completion year, number of units and nearby rail stations.
# Usage: bash scripts/open_project_pages.sh <person 1-5> <batch>
set -euo pipefail

person="${1:?give a person number from 1 to 5}"
batch="${2:?give a batch number starting at 1}"
batch_size=10
list="$(dirname "$0")/../docs/project_pages/person_${person}.txt"

urls=()
while IFS= read -r url; do
  [[ -n "$url" ]] && urls+=("$url")
done < "$list"

start=$(( (batch - 1) * batch_size ))
total_batches=$(( (${#urls[@]} + batch_size - 1) / batch_size ))
if (( start >= ${#urls[@]} )); then
  echo "person $person has ${#urls[@]} pages in $total_batches batches; batch $batch is empty"
  exit 1
fi
selection=("${urls[@]:start:batch_size}")
echo "person $person, batch $batch of $total_batches: pages $(( start + 1 ))-$(( start + ${#selection[@]} )) of ${#urls[@]}"
open -a "Google Chrome" "${selection[@]}"
