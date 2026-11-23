(() => {
  "use strict";

  const state = {
    meta: null,
    projects: [],
    selectedProject: null,
    activeOptionIndex: -1,
    // last prediction from the server; the year slider re-reads its forecast series without a new request
    lastPrediction: null,
  };

  const els = {
    globalError: document.getElementById("global-error"),
    form: document.getElementById("estimate-form"),
    zoneSelect: document.getElementById("zone-select"),
    areaInput: document.getElementById("area-input"),
    bedroomsSelect: document.getElementById("bedrooms-select"),
    bathroomsSelect: document.getElementById("bathrooms-select"),
    floorInput: document.getElementById("floor-input"),
    projectSearch: document.getElementById("project-search"),
    projectClear: document.getElementById("project-clear"),
    projectListbox: document.getElementById("project-listbox"),
    projectHint: document.getElementById("project-hint"),
    manualFields: document.getElementById("manual-building-fields"),
    completeYearInput: document.getElementById("complete-year-input"),
    railKmInput: document.getElementById("rail-km-input"),
    isNewCheckbox: document.getElementById("is-new-checkbox"),
    yearSliderWrapper: document.getElementById("year-slider-wrapper"),
    yearSlider: document.getElementById("year-slider"),
    yearValue: document.getElementById("year-value"),
    yearMinLabel: document.getElementById("year-min-label"),
    yearMaxLabel: document.getElementById("year-max-label"),
    submitBtn: document.getElementById("submit-btn"),
    resetBtn: document.getElementById("reset-btn"),
    resultsSection: document.getElementById("results-section"),
    copySummaryBtn: document.getElementById("copy-summary-btn"),
    inputWarnings: document.getElementById("input-warnings"),
    mainModelLabel: document.getElementById("main-model-label"),
    mainPrice: document.getElementById("main-price"),
    mainRange: document.getElementById("main-range"),
    mainBasis: document.getElementById("main-basis"),
    knnPrice: document.getElementById("knn-price"),
    knnSetting: document.getElementById("knn-setting"),
    tierRow: document.getElementById("tier-row"),
    reliabilityLines: document.getElementById("reliability-lines"),
    reliabilityWarnings: document.getElementById("reliability-warnings"),
    buildingInfo: document.getElementById("building-info"),
    modelBreakdownBody: document.getElementById("model-breakdown-body"),
    neighboursBody: document.getElementById("neighbours-body"),
    futureSection: document.getElementById("future-section"),
    staleNote: document.getElementById("stale-note"),
    futureLabel: document.getElementById("future-label"),
    futurePrice: document.getElementById("future-price"),
    futureNote: document.getElementById("future-note"),
    forecastChart: document.getElementById("forecast-chart"),
    qualityBody: document.getElementById("quality-body"),
  };

  function formatNumber(val, decimals = 0) {
    if (val === null || val === undefined || Number.isNaN(Number(val))) return "-";
    return Number(val).toLocaleString("en-US", {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    });
  }

  function formatBaht(val) {
    return `${formatNumber(Math.round(val), 0)} บาท`;
  }

  function showGlobalError(message) {
    if (!message) {
      els.globalError.classList.add("hidden");
      els.globalError.textContent = "";
      return;
    }
    els.globalError.textContent = message;
    els.globalError.classList.remove("hidden");
  }

  function renderWarnings(container, warnings) {
    container.innerHTML = "";
    (warnings || []).forEach((msg) => {
      const div = document.createElement("div");
      div.className = "alert alert-warn";
      div.textContent = msg;
      container.appendChild(div);
    });
  }

  function markStale() {
    if (state.lastPrediction) els.staleNote.classList.remove("hidden");
  }

  function setSelectedProject(name) {
    if ((name || null) !== state.selectedProject) markStale();
    state.selectedProject = name || null;
    if (state.selectedProject) {
      els.projectSearch.value = state.selectedProject;
      els.projectClear.classList.remove("hidden");
      els.manualFields.classList.add("muted-fields");
      els.projectHint.textContent = `เลือกโครงการ: ${state.selectedProject} (ระบบจะใช้ข้อมูลตึกของโครงการนี้อัตโนมัติ)`;
    } else {
      els.projectSearch.value = "";
      els.projectClear.classList.add("hidden");
      els.manualFields.classList.remove("muted-fields");
      els.projectHint.textContent = "ไม่ได้เลือกโครงการ — ใช้ปีสร้างเสร็จและระยะรถไฟฟ้าที่กรอกด้านล่าง";
    }
    closeProjectList();
  }

  function closeProjectList() {
    els.projectListbox.classList.add("hidden");
    els.projectSearch.setAttribute("aria-expanded", "false");
    state.activeOptionIndex = -1;
  }

  function renderProjectOptions(query) {
    const q = (query || "").trim().toLowerCase();
    const matches = [];
    if (!q) {
      matches.push({ label: state.meta?.no_project_label || "ไม่ระบุโครงการ", value: null });
    }
    const maxItems = 60;
    for (let i = 0; i < state.projects.length && matches.length < maxItems; i++) {
      const p = state.projects[i];
      if (!q || p.toLowerCase().includes(q)) {
        matches.push({ label: p, value: p });
      }
    }

    els.projectListbox.innerHTML = "";
    if (!matches.length) {
      const li = document.createElement("li");
      li.className = "combobox-empty";
      li.textContent = "ไม่พบชื่อโครงการที่ตรงกัน";
      els.projectListbox.appendChild(li);
    } else {
      matches.forEach((item) => {
        const li = document.createElement("li");
        li.className = "combobox-option";
        li.role = "option";
        li.textContent = item.label;
        li.addEventListener("mousedown", (e) => {
          e.preventDefault();
          setSelectedProject(item.value);
        });
        els.projectListbox.appendChild(li);
      });
    }
    els.projectListbox.classList.remove("hidden");
    els.projectSearch.setAttribute("aria-expanded", "true");
    state.activeOptionIndex = -1;
  }

  function setupCombobox() {
    els.projectSearch.addEventListener("focus", () => {
      renderProjectOptions(state.selectedProject ? "" : els.projectSearch.value);
    });

    els.projectSearch.addEventListener("input", () => {
      const val = els.projectSearch.value.trim();
      if (!val) {
        state.selectedProject = null;
        els.projectClear.classList.add("hidden");
        els.manualFields.classList.remove("muted-fields");
        els.projectHint.textContent = "ไม่ได้เลือกโครงการ — ใช้ปีสร้างเสร็จและระยะรถไฟฟ้าที่กรอกด้านล่าง";
      } else if (state.selectedProject && val !== state.selectedProject) {
        state.selectedProject = null;
        els.manualFields.classList.remove("muted-fields");
      }
      renderProjectOptions(val);
    });

    els.projectSearch.addEventListener("keydown", (e) => {
      const options = els.projectListbox.querySelectorAll(".combobox-option");
      if (!options.length) return;
      if (e.key === "ArrowDown") {
        e.preventDefault();
        state.activeOptionIndex = (state.activeOptionIndex + 1) % options.length;
        updateActiveOption(options);
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        state.activeOptionIndex = (state.activeOptionIndex - 1 + options.length) % options.length;
        updateActiveOption(options);
      } else if (e.key === "Enter" && !els.projectListbox.classList.contains("hidden")) {
        e.preventDefault();
        const target = options[state.activeOptionIndex >= 0 ? state.activeOptionIndex : 0];
        if (target) target.dispatchEvent(new MouseEvent("mousedown"));
      } else if (e.key === "Escape") {
        closeProjectList();
      }
    });

    els.projectSearch.addEventListener("blur", () => {
      setTimeout(() => {
        const typed = els.projectSearch.value.trim();
        if (!state.selectedProject && typed) {
          const exact = state.projects.find((p) => p.toLowerCase() === typed.toLowerCase());
          if (exact) {
            setSelectedProject(exact);
          } else {
            setSelectedProject(null);
          }
        }
        closeProjectList();
      }, 120);
    });

    els.projectClear.addEventListener("click", () => {
      setSelectedProject(null);
      els.projectSearch.focus();
    });
  }

  function updateActiveOption(options) {
    options.forEach((el, idx) => {
      el.classList.toggle("active", idx === state.activeOptionIndex);
      if (idx === state.activeOptionIndex) {
        el.scrollIntoView({ block: "nearest" });
      }
    });
  }

  function renderQualityTable(rows) {
    els.qualityBody.innerHTML = "";
    (rows || []).forEach((row) => {
      const tr = document.createElement("tr");
      const cells = [
        { text: row.model, cls: "" },
        { text: formatNumber(row.mape_random_pct, 1), cls: "num" },
        { text: formatNumber(row.mape_unseen_pct, 1), cls: "num" },
        { text: formatNumber(row.mae_million_thb, 2), cls: "num" },
        { text: formatNumber(row.r2, 3), cls: "num" },
        { text: formatNumber(row.within_10pct, 1), cls: "num" },
      ];
      cells.forEach(({ text, cls }) => {
        const td = document.createElement("td");
        if (cls) td.className = cls;
        td.textContent = text;
        tr.appendChild(td);
      });
      els.qualityBody.appendChild(tr);
    });
  }

  function renderForecastChart(series) {
    els.forecastChart.innerHTML = "";
    if (!series || series.length < 2) return;

    const width = 820;
    const height = 280;
    const pad = { top: 20, right: 24, bottom: 38, left: 78 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;

    const allVals = series.flatMap((d) => [d.low, d.price, d.high]);
    const rawMin = Math.min(...allVals);
    const rawMax = Math.max(...allVals);
    const span = Math.max(rawMax - rawMin, rawMax * 0.08, 1);
    const minY = Math.max(0, rawMin - span * 0.12);
    const maxY = rawMax + span * 0.12;

    const xOf = (idx) => pad.left + (idx / (series.length - 1)) * plotW;
    const yOf = (val) => pad.top + plotH - ((val - minY) / (maxY - minY)) * plotH;

    const svgNS = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(svgNS, "svg");
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", "กราฟคาดการณ์ราคาคอนโดตามแนวโน้มตลาด");

    const gridSteps = 4;
    for (let i = 0; i <= gridSteps; i++) {
      const val = minY + ((maxY - minY) * i) / gridSteps;
      const y = yOf(val);
      const line = document.createElementNS(svgNS, "line");
      line.setAttribute("x1", pad.left);
      line.setAttribute("x2", width - pad.right);
      line.setAttribute("y1", y.toFixed(1));
      line.setAttribute("y2", y.toFixed(1));
      line.setAttribute("stroke", "#2B2B2B");
      line.setAttribute("stroke-width", "1");
      svg.appendChild(line);

      const label = document.createElementNS(svgNS, "text");
      label.setAttribute("x", pad.left - 10);
      label.setAttribute("y", (y + 4).toFixed(1));
      label.setAttribute("text-anchor", "end");
      label.setAttribute("fill", "#7D7D7D");
      label.setAttribute("font-size", "11");
      label.textContent = `${(val / 1e6).toFixed(2)}M`;
      svg.appendChild(label);
    }

    series.forEach((d, idx) => {
      const x = xOf(idx);
      const label = document.createElementNS(svgNS, "text");
      label.setAttribute("x", x.toFixed(1));
      label.setAttribute("y", height - 12);
      label.setAttribute("text-anchor", "middle");
      label.setAttribute("fill", "#7D7D7D");
      label.setAttribute("font-size", "11");
      label.textContent = String(d.year);
      svg.appendChild(label);
    });

    const lines = [
      { key: "low", color: "#7D7D7D", width: 2 },
      { key: "high", color: "#FFFFFF", width: 2 },
      { key: "price", color: "#FFC000", width: 3 },
    ];

    lines.forEach(({ key, color, width: strokeWidth }) => {
      const points = series.map((d, idx) => `${xOf(idx).toFixed(1)},${yOf(d[key]).toFixed(1)}`).join(" ");
      const poly = document.createElementNS(svgNS, "polyline");
      poly.setAttribute("fill", "none");
      poly.setAttribute("stroke", color);
      poly.setAttribute("stroke-width", String(strokeWidth));
      poly.setAttribute("points", points);
      svg.appendChild(poly);
    });

    series.forEach((d, idx) => {
      const circle = document.createElementNS(svgNS, "circle");
      circle.setAttribute("cx", xOf(idx).toFixed(1));
      circle.setAttribute("cy", yOf(d.price).toFixed(1));
      circle.setAttribute("r", "4");
      circle.setAttribute("fill", "#FFC000");
      const title = document.createElementNS(svgNS, "title");
      title.textContent = `ปี ${d.year}: ${formatBaht(d.price)} (ช่วง ${formatBaht(d.low)} - ${formatBaht(d.high)})`;
      circle.appendChild(title);
      svg.appendChild(circle);
    });

    els.forecastChart.appendChild(svg);
  }

  function baseYear(data) {
    return data.forecast_series ? data.forecast_series.base_year : 2026;
  }

  // Price for the year on the slider, from the forecast series the server already sent.
  function renderFuture(year) {
    const data = state.lastPrediction;
    const series = data && data.forecast_series;
    if (!series || year <= series.base_year) {
      els.futureSection.classList.add("hidden");
      return;
    }
    const row = series.years.find((item) => item.year === year);
    if (!row) {
      els.futureSection.classList.add("hidden");
      return;
    }
    const horizon = year - series.base_year;
    els.futureLabel.textContent = `คาดการณ์ราคาปี ${year} (แนวโน้มตลาด จากราคาวันนี้ ${formatBaht(data.price)})`;
    els.futurePrice.textContent = formatBaht(row.price);
    const errText = row.backtest_mape_pct !== null && row.backtest_mape_pct !== undefined
      ? ` | ทดสอบย้อนหลังทายล่วงหน้า ${horizon} ปี คลาดเคลื่อนเฉลี่ย ${Number(row.backtest_mape_pct).toFixed(1)}%`
      : "";
    els.futureNote.textContent =
      `ช่วงที่น่าจะเป็น ${formatBaht(row.low)} - ${formatBaht(row.high)} ` +
      `| แนวโน้ม ${Number(series.annual_growth_pct).toFixed(1)}% ต่อปี จากดัชนีราคาที่อยู่อาศัยกรุงเทพฯ 1991-2026${errText}`;
    renderForecastChart(series.years.filter((item) => item.year <= year));
    els.futureSection.classList.remove("hidden");
  }

  function renderPrediction(data) {
    renderWarnings(els.inputWarnings, data.input_warnings);

    els.mainPrice.textContent = formatBaht(data.price);
    els.mainRange.textContent =
      `ถ้าทายพลาดไม่เกิน 20% ราคาจริงอยู่ระหว่าง ${formatBaht(data.range_low)} - ${formatBaht(data.range_high)} ` +
      `| ${formatNumber(Math.round(data.price_per_sqm), 0)} บาท/ตร.ม.`;
    els.mainBasis.textContent = data.main_basis;

    els.knnPrice.textContent = formatBaht(data.knn);
    els.knnSetting.textContent = `ระยะ ${data.knn_setting}`;
    els.tierRow.innerHTML = "";
    const badge = document.createElement("span");
    badge.className = "tier-badge";
    badge.textContent = `ระดับทำเล ${data.tier} / ${data.n_clusters}`;
    els.tierRow.appendChild(badge);
    els.tierRow.appendChild(document.createTextNode("จาก K-Means (1 = ราคาต่อตร.ม. ต่ำสุด)"));

    els.reliabilityLines.innerHTML = "";
    (data.reliability_lines || []).forEach((line) => {
      const div = document.createElement("div");
      div.className = "value-note";
      div.textContent = line;
      els.reliabilityLines.appendChild(div);
    });
    renderWarnings(els.reliabilityWarnings, data.reliability_warnings);

    const b = data.building || {};
    const yearText = b.complete_year !== null && b.complete_year !== undefined ? `${b.complete_year}` : "ไม่ทราบ";
    const railText = b.nearest_rail_km !== null && b.nearest_rail_km !== undefined
      ? `${Number(b.nearest_rail_km).toFixed(1)} กม.`
      : "เกิน 3 กม.";
    let buildingLine = `ข้อมูลตึกที่ใช้: สร้างเสร็จปี ${yearText} | ระยะถึงรถไฟฟ้า ${railText}`;
    if (b.total_unit && b.total_floor) {
      buildingLine += ` | ${b.total_unit} ยูนิต ${b.total_floor} ชั้น`;
    }
    els.buildingInfo.textContent = buildingLine;

    const areaSqm = data.price_per_sqm > 0 ? data.price / data.price_per_sqm : Number(els.areaInput.value) || 1;
    els.modelBreakdownBody.innerHTML = "";
    (data.model_breakdown || []).forEach((row) => {
      const tr = document.createElement("tr");
      const tdModel = document.createElement("td");
      tdModel.textContent = row.model;
      const tdPrice = document.createElement("td");
      tdPrice.className = "num";
      tdPrice.textContent = formatNumber(Math.round(row.price), 0);
      const tdSqm = document.createElement("td");
      tdSqm.className = "num";
      tdSqm.textContent = formatNumber(Math.round(row.price / areaSqm), 0);
      tr.appendChild(tdModel);
      tr.appendChild(tdPrice);
      tr.appendChild(tdSqm);
      els.modelBreakdownBody.appendChild(tr);
    });

    els.neighboursBody.innerHTML = "";
    (data.neighbours || []).forEach((row) => {
      const tr = document.createElement("tr");
      const cells = [
        { text: row.project_name || "-", cls: "" },
        { text: row.zone_name_th || "-", cls: "" },
        { text: row.district_th || "-", cls: "" },
        { text: row.room_type || "-", cls: "" },
        { text: formatNumber(row.bedrooms, 0), cls: "num" },
        { text: formatNumber(row.bathrooms, 0), cls: "num" },
        { text: formatNumber(row.floor, 0), cls: "num" },
        { text: formatNumber(row.area_sqm, 1), cls: "num" },
        { text: formatNumber(Math.round(row.price_thb), 0), cls: "num" },
      ];
      cells.forEach(({ text, cls }) => {
        const td = document.createElement("td");
        if (cls) td.className = cls;
        td.textContent = text;
        tr.appendChild(td);
      });
      els.neighboursBody.appendChild(tr);
    });

    state.lastPrediction = data;
    els.staleNote.classList.add("hidden");
    els.mainModelLabel.textContent = `ราคาวันนี้ (ปี ${baseYear(data)}) · ${data.main_title}`;
    renderFuture(Number(els.yearSlider.value || baseYear(data)));
    els.resultsSection.classList.remove("hidden");
  }

  async function loadMetadata() {
    try {
      const res = await fetch("/api/meta");
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      const meta = await res.json();
      state.meta = meta;
      state.projects = meta.projects || [];

      els.zoneSelect.innerHTML = "";
      (meta.zones || []).forEach((z) => {
        const opt = document.createElement("option");
        opt.value = String(z.zone_no);
        opt.textContent = z.label;
        els.zoneSelect.appendChild(opt);
      });
      els.zoneSelect.disabled = false;

      if (meta.forecast && meta.forecast.available) {
        els.yearSlider.min = String(meta.forecast.min_year);
        els.yearSlider.max = String(meta.forecast.max_year);
        els.yearSlider.value = String(meta.forecast.base_year);
        els.yearValue.textContent = String(meta.forecast.base_year);
        els.yearMinLabel.textContent = String(meta.forecast.min_year);
        els.yearMaxLabel.textContent = String(meta.forecast.max_year);
        els.yearSliderWrapper.classList.remove("hidden");
      } else {
        els.yearSliderWrapper.classList.add("hidden");
      }

      renderQualityTable(meta.model_quality);
      els.submitBtn.disabled = false;
      showGlobalError("");
    } catch (err) {
      showGlobalError(`โหลดข้อมูลเริ่มต้นไม่สำเร็จ: ${err.message}`);
    }
  }

  async function handleSubmit(e) {
    e.preventDefault();
    showGlobalError("");
    els.submitBtn.disabled = true;
    els.submitBtn.textContent = "กำลังประเมินราคา...";

    const payload = {
      zone_no: Number(els.zoneSelect.value),
      area_sqm: Number(els.areaInput.value),
      bedrooms: Number(els.bedroomsSelect.value),
      bathrooms: Number(els.bathroomsSelect.value),
      floor: Number(els.floorInput.value),
      project: state.selectedProject,
      complete_year: Number(els.completeYearInput.value),
      rail_km: Number(els.railKmInput.value),
      is_new: Boolean(els.isNewCheckbox.checked),
      year: Number(els.yearSlider.value || 2026),
    };

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data.detail || `HTTP ${res.status}`);
      }
      renderPrediction(data);
    } catch (err) {
      showGlobalError(`ไม่สามารถประเมินราคาได้: ${err.message}`);
    } finally {
      els.submitBtn.disabled = false;
      els.submitBtn.textContent = "ประเมินราคา";
    }
  }

  const PRESETS = {
    studio: { area_sqm: 28, bedrooms: 0, bathrooms: 1, floor: 12, complete_year: 2020, rail_km: 0.3, is_new: false },
    default: { area_sqm: 35, bedrooms: 1, bathrooms: 1, floor: 10, complete_year: 2016, rail_km: 0.5, is_new: false },
    family: { area_sqm: 65, bedrooms: 2, bathrooms: 2, floor: 24, complete_year: 2019, rail_km: 0.4, is_new: false },
  };

  function applyPreset(key) {
    const preset = PRESETS[key];
    if (!preset) return;
    els.areaInput.value = String(preset.area_sqm);
    els.bedroomsSelect.value = String(preset.bedrooms);
    els.bathroomsSelect.value = String(preset.bathrooms);
    els.floorInput.value = String(preset.floor);
    els.completeYearInput.value = String(preset.complete_year);
    els.railKmInput.value = String(preset.rail_km);
    els.isNewCheckbox.checked = Boolean(preset.is_new);
    setSelectedProject(null);
    markStale();
  }

  document.querySelectorAll("[data-preset]").forEach((btn) => {
    btn.addEventListener("click", () => applyPreset(btn.dataset.preset));
  });

  function resetToDefaults() {
    const d = state.meta?.defaults || {
      zone_no: 1, area_sqm: 35, bedrooms: 1, bathrooms: 1, floor: 10,
      complete_year: 2016, rail_km: 0.5, is_new: false, year: 2026,
    };
    if (els.zoneSelect.options.length > 0) {
      els.zoneSelect.value = String(d.zone_no);
    }
    els.areaInput.value = String(d.area_sqm);
    els.bedroomsSelect.value = String(d.bedrooms);
    els.bathroomsSelect.value = String(d.bathrooms);
    els.floorInput.value = String(d.floor);
    els.completeYearInput.value = String(d.complete_year);
    els.railKmInput.value = String(d.rail_km);
    els.isNewCheckbox.checked = Boolean(d.is_new);
    els.yearSlider.value = String(d.year || 2026);
    els.yearValue.textContent = els.yearSlider.value;
    setSelectedProject(null);
    showGlobalError("");
    if (state.lastPrediction) {
      renderFuture(Number(els.yearSlider.value));
      markStale();
    }
  }

  function buildSummaryText() {
    const data = state.lastPrediction;
    if (!data) return "";
    const zoneLabel = els.zoneSelect.options[els.zoneSelect.selectedIndex]?.textContent || "";
    const bedLabel = els.bedroomsSelect.options[els.bedroomsSelect.selectedIndex]?.textContent || "";
    const projPart = state.selectedProject ? `โครงการ: ${state.selectedProject}` : `โซน: ${zoneLabel}`;
    const lines = [
      "BKK Condo Price — สรุปผลการประเมิน",
      `${projPart} | ${els.areaInput.value} ตร.ม. | ${bedLabel} | ${els.bathroomsSelect.value} ห้องน้ำ | ชั้น ${els.floorInput.value}`,
      `ราคาวันนี้ (ปี ${baseYear(data)}): ${formatBaht(data.price)} (${formatNumber(Math.round(data.price_per_sqm), 0)} บาท/ตร.ม.)`,
      `ช่วง ±20%: ${formatBaht(data.range_low)} - ${formatBaht(data.range_high)}`,
      `KNN เปรียบเทียบ: ${formatBaht(data.knn)} | ระดับทำเล ${data.tier}/${data.n_clusters}`,
    ];
    const sliderYear = Number(els.yearSlider.value || baseYear(data));
    if (data.forecast_series && sliderYear > data.forecast_series.base_year) {
      const row = data.forecast_series.years.find((item) => item.year === sliderYear);
      if (row) {
        lines.push(`คาดการณ์ปี ${sliderYear}: ${formatBaht(row.price)} (${formatBaht(row.low)} - ${formatBaht(row.high)})`);
      }
    }
    return lines.join("\n");
  }

  els.copySummaryBtn.addEventListener("click", async () => {
    const text = buildSummaryText();
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      els.copySummaryBtn.textContent = "คัดลอกแล้ว";
      setTimeout(() => {
        els.copySummaryBtn.textContent = "คัดลอกสรุปราคา";
      }, 1800);
    } catch {
      showGlobalError("เบราว์เซอร์ไม่อนุญาตให้คัดลอกอัตโนมัติ");
    }
  });

  els.resetBtn.addEventListener("click", resetToDefaults);

  els.yearSlider.addEventListener("input", () => {
    els.yearValue.textContent = els.yearSlider.value;
    renderFuture(Number(els.yearSlider.value));
  });
  // any change to the room itself makes the shown price out of date (the year slider does not)
  els.form.addEventListener("input", (e) => {
    if (e.target !== els.yearSlider && e.target !== els.projectSearch) markStale();
  });
  els.form.addEventListener("change", (e) => {
    if (e.target !== els.yearSlider && e.target !== els.projectSearch) markStale();
  });
  els.form.addEventListener("submit", handleSubmit);
  setupCombobox();
  loadMetadata();
})();
