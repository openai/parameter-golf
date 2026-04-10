const state = {
  data: null,
  records: [],
  filtered: [],
  selectedId: null,
  charts: {},
};

const els = {};

const metricLabels = {
  val_bpb: "Base BPB",
  cap_val_bpb: "Cap Validation BPB",
  diag_bpb: "Diagnostic BPB",
  sliding_bpb: "Sliding BPB",
  ngram9_bpb: "N-gram 9 BPB",
  delta: "Delta",
};

const metricOrder = ["val_bpb", "cap_val_bpb", "diag_bpb", "sliding_bpb", "ngram9_bpb", "delta"];
const errorKeywords = ["traceback", "oom", "shard size mismatch", "cuda error", "failed", "valueerror", "runtimeerror"];
const warnKeywords = ["promote", "decision", "proxy", "watch", "regression"];

document.addEventListener("DOMContentLoaded", init);

async function init() {
  bindElements();
  bindEvents();
  await loadData();
}

function bindElements() {
  [
    "scopeLabel",
    "generatedAt",
    "ablationCount",
    "statTotal",
    "statOk",
    "statWarn",
    "statError",
    "bestMetricValue",
    "bestMetricLabel",
    "sotaCards",
    "sotaChart",
    "sotaTableBody",
    "bpbRankingBody",
    "baseRankingBody",
    "sizeRankingBody",
    "hypothesisCurrent",
    "hypothesisSupport",
    "hypothesisContradiction",
    "hypothesisNext",
    "statusChart",
    "ablationChart",
    "timelineChart",
    "ablationCards",
    "ablationTableBody",
    "activeQuery",
    "searchInput",
    "categoryFilter",
    "statusFilter",
    "metricSelect",
    "sortDir",
    "resetBtn",
    "visibleCount",
    "visibleExperiments",
    "visibleErrors",
    "resultCount",
    "recordsBody",
    "detailTitle",
    "copyPathBtn",
    "detailStatusBadge",
    "detailPathCode",
    "detailMeta",
    "detailWriteup",
    "detailMetrics",
    "detailNotes",
    "detailSnippet",
  ].forEach((id) => {
    els[id] = document.getElementById(id);
  });
}

function bindEvents() {
  els.searchInput.addEventListener("input", applyFilters);
  els.categoryFilter.addEventListener("change", applyFilters);
  els.statusFilter.addEventListener("change", applyFilters);
  els.metricSelect.addEventListener("change", applyFilters);
  els.sortDir.addEventListener("change", applyFilters);
  els.resetBtn.addEventListener("click", resetFilters);
  els.copyPathBtn.addEventListener("click", copySelectedPath);
  window.addEventListener("resize", resizeCharts);
}

async function loadData() {
  try {
    const response = await fetch("./hub_index.json", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Failed to load index: ${response.status}`);
    }

    state.data = await response.json();
    state.records = Array.isArray(state.data.records) ? state.data.records : [];

    hydrateControls();
    renderHeader();
    renderHypothesis();
    renderSotas();
    renderAblations();
    renderGlobalCharts();
    applyFilters();
  } catch (error) {
    renderLoadFailure(error);
  }
}

function renderLoadFailure(error) {
  els.resultCount.textContent = "Unable to load hub index.";
  els.recordsBody.innerHTML = `<tr><td colspan="5" class="empty-cell">Error: ${escapeHtml(error.message)}</td></tr>`;
  els.detailTitle.textContent = "Load failed";
  els.detailStatusBadge.textContent = "error";
  els.detailStatusBadge.className = "status-pill status-error";
  els.detailPathCode.textContent = "No path available.";
  els.detailWriteup.innerHTML = `<p class="writeup-line">${escapeHtml(String(error.stack || error.message))}</p>`;
  els.detailSnippet.textContent = String(error.stack || error.message);
}

function hydrateControls() {
  const categories = new Set(["all"]);
  const metrics = new Set(metricOrder);

  state.records.forEach((record) => {
    if (record.category) categories.add(record.category);
    Object.keys(record.metrics || {}).forEach((key) => metrics.add(key));
  });

  fillSelect(els.categoryFilter, [...categories], "All categories");
  fillMetricSelect(els.metricSelect, [...metrics]);
  els.sortDir.value = "asc";
}

function fillSelect(select, values, allLabel) {
  const existing = select.value;
  select.innerHTML = "";

  if (allLabel) {
    const option = document.createElement("option");
    option.value = "all";
    option.textContent = allLabel;
    select.appendChild(option);
  }

  values
    .filter((value) => value !== "all")
    .sort((a, b) => a.localeCompare(b))
    .forEach((value) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = value;
      select.appendChild(option);
    });

  if (existing && [...select.options].some((option) => option.value === existing)) {
    select.value = existing;
  }
}

function fillMetricSelect(select, values) {
  const existing = select.value;
  select.innerHTML = "";

  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "Sort by recency / path";
  select.appendChild(placeholder);

  [...values]
    .filter((value) => value && value !== "all")
    .sort((a, b) => {
      const ai = metricOrder.indexOf(a);
      const bi = metricOrder.indexOf(b);
      if (ai === -1 && bi === -1) return a.localeCompare(b);
      if (ai === -1) return 1;
      if (bi === -1) return -1;
      return ai - bi;
    })
    .forEach((value) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = metricLabels[value] || value;
      select.appendChild(option);
    });

  if (existing && [...select.options].some((option) => option.value === existing)) {
    select.value = existing;
    return;
  }

  select.value = metricOrder.find((key) => values.includes(key)) || "";
}

function renderHeader() {
  const counts = state.data?.counts || {};
  els.scopeLabel.textContent = (state.data?.source_roots || []).join(" | ") || "experiments";
  els.generatedAt.textContent = formatTimestamp(state.data?.generated_at);
  els.ablationCount.textContent = counts.ablations ?? (state.data?.ablations || []).length;

  els.statTotal.textContent = counts.total_records ?? state.records.length;
  els.statOk.textContent = counts.by_status?.ok ?? 0;
  els.statWarn.textContent = counts.by_status?.warn ?? 0;
  els.statError.textContent = counts.by_status?.error ?? 0;
}

function renderHypothesis() {
  const hypothesis = state.data?.hypothesis || {};
  els.hypothesisCurrent.textContent = hypothesis.current_hypothesis || "No working hypothesis yet.";
  els.hypothesisSupport.textContent = hypothesis.supporting_signal || "No supporting signal extracted.";
  els.hypothesisContradiction.textContent = hypothesis.contradictory_signal || "No contradiction extracted.";
  els.hypothesisNext.textContent = hypothesis.next_test || "No next test recommendation extracted.";
}

function renderSotas() {
  const sotas = Array.isArray(state.data?.favorite_considerations)
    ? state.data.favorite_considerations
    : Array.isArray(state.data?.personal_sotas)
      ? state.data.personal_sotas
      : [];
  const rankings = state.data?.independent_rankings || {};

  els.sotaCards.innerHTML = sotas
    .map((item) => {
      const status = item.status || "unknown";
      const value = formatConsiderationValue(item);
      const metricUsed = item.metric_used || item.metric_key || "-";
      return `
        <article class="sota-card status-${escapeHtml(status)}">
          <span class="sota-label">${escapeHtml(item.label || item.category || "SOTA")}</span>
          <strong class="sota-value">${escapeHtml(value)}</strong>
          <p class="sota-meta">${escapeHtml(item.run_tag || "untracked")} · ${escapeHtml(item.experiment_group || "n/a")} · ${escapeHtml(metricUsed)}</p>
          <code class="sota-path">${escapeHtml(item.rel_path || "")}</code>
        </article>
      `;
    })
    .join("");

  els.sotaTableBody.innerHTML = sotas
    .map((item) => {
      const status = item.status || "unknown";
      const value = formatConsiderationValue(item);
      return `
        <tr>
          <td>${escapeHtml(item.label || item.category || "-")}</td>
          <td><span class="metric-pill">${escapeHtml(value)}</span></td>
          <td>${escapeHtml(item.metric_used || item.metric_key || "-")}</td>
          <td>${escapeHtml(item.run_tag || "untracked")}</td>
          <td><span class="status-pill status-${escapeHtml(status)}">${escapeHtml(status)}</span></td>
        </tr>
      `;
    })
    .join("");

  renderIndependentRanking(els.bpbRankingBody, rankings.best_bpb || []);
  renderIndependentRanking(els.baseRankingBody, rankings.best_base_model || []);
  renderIndependentRanking(els.sizeRankingBody, rankings.lowest_file_size || []);

  initSotaChart(sotas);
}

function renderIndependentRanking(target, rows) {
  if (!target) return;
  if (!rows.length) {
    target.innerHTML = '<tr><td colspan="4" class="empty-cell">No ranked tests available.</td></tr>';
    return;
  }

  target.innerHTML = rows
    .slice(0, 10)
    .map((row, index) => {
      const value = formatConsiderationValue(row);
      return `
        <tr>
          <td>#${index + 1}</td>
          <td>${escapeHtml(value)}</td>
          <td>${escapeHtml(row.metric_used || row.metric_key || "-")}</td>
          <td>${escapeHtml(row.run_tag || "untracked")}</td>
        </tr>
      `;
    })
    .join("");
}

function initSotaChart(sotas) {
  if (!window.echarts || !els.sotaChart) return;
  const rows = sotas.filter((item) => Number.isFinite(item.value));
  const chart = getChart("sotaChart", els.sotaChart);
  chart.setOption(
    {
      backgroundColor: "transparent",
      tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
      grid: { left: 110, right: 30, top: 20, bottom: 22 },
      xAxis: {
        type: "value",
        axisLabel: { color: "#9fb0ba" },
        splitLine: { lineStyle: { color: "rgba(123,226,217,0.12)" } },
      },
      yAxis: {
        type: "category",
        data: rows.map((item) => item.label || item.category),
        axisLabel: { color: "#d7e2e8" },
      },
      series: [
        {
          type: "bar",
          data: rows.map((item) => item.value),
          itemStyle: {
            color: (params) => (params.dataIndex % 2 === 0 ? "#7fe7d7" : "#f9b06a"),
            borderRadius: [0, 7, 7, 0],
          },
          label: {
            show: true,
            position: "right",
            color: "#f4e6d0",
            formatter: ({ value }) => formatNumber(value),
          },
        },
      ],
    },
    true,
  );
}

function renderAblations() {
  const ablations = Array.isArray(state.data?.ablations) ? state.data.ablations : [];

  els.ablationCards.innerHTML = ablations
    .slice(0, 8)
    .map((item) => {
      const delta = Number.isFinite(item.delta) ? item.delta : null;
      const verdict = item.verdict || "unclassified";
      const className = delta == null ? "unknown" : delta < 0 ? "ok" : "error";
      return `
        <article class="ablation-card verdict-${className}">
          <header>
            <h3>${escapeHtml(item.title || item.group || "Ablation")}</h3>
            <span class="status-pill status-${className}">${escapeHtml(verdict)}</span>
          </header>
          <p class="ablation-summary">${escapeHtml(item.summary || "No summary.")}</p>
          <dl>
            <div><dt>Primary metric</dt><dd>${escapeHtml(metricLabels[item.primary_metric] || item.primary_metric || "-")}</dd></div>
            <div><dt>Baseline</dt><dd>${escapeHtml(item.baseline_label || "-")} (${escapeHtml(formatNumber(item.baseline_value))})</dd></div>
            <div><dt>Candidate</dt><dd>${escapeHtml(item.candidate_label || "-")} (${escapeHtml(formatNumber(item.candidate_value))})</dd></div>
            <div><dt>Delta</dt><dd class="delta ${className}">${delta == null ? "n/a" : escapeHtml(formatNumber(delta))}</dd></div>
          </dl>
        </article>
      `;
    })
    .join("");

  els.ablationTableBody.innerHTML = ablations
    .map((item) => {
      const delta = Number.isFinite(item.delta) ? item.delta : null;
      const className = delta == null ? "unknown" : delta < 0 ? "ok" : "error";
      return `
        <tr>
          <td>${escapeHtml(item.group || item.title || "-")}</td>
          <td>${escapeHtml(item.candidate_label || "-")}</td>
          <td>${escapeHtml(metricLabels[item.primary_metric] || item.primary_metric || "-")}</td>
          <td class="delta ${className}">${delta == null ? "n/a" : escapeHtml(formatNumber(delta))}</td>
          <td><span class="status-pill status-${className}">${escapeHtml(item.verdict || "-")}</span></td>
        </tr>
      `;
    })
    .join("");
}

function renderGlobalCharts() {
  if (!window.echarts) return;
  const charts = state.data?.charts || {};

  initStatusChart(charts);
  initAblationDeltaChart(charts);
  initTimelineChart(charts);
}

function initStatusChart(charts) {
  const statusRows = Array.isArray(charts.status_distribution) ? charts.status_distribution : [];
  if (!els.statusChart || !window.echarts) return;

  const chart = getChart("statusChart", els.statusChart);
  chart.setOption(
    {
      backgroundColor: "transparent",
      tooltip: { trigger: "item" },
      legend: { bottom: 0, textStyle: { color: "#c4d0d7" } },
      series: [
        {
          type: "pie",
          radius: ["36%", "68%"],
          center: ["50%", "42%"],
          label: { color: "#d7e2e8", formatter: "{b}: {c}" },
          data: statusRows,
          color: ["#90efb6", "#ffb066", "#ff7f74", "#9ca8b5"],
          itemStyle: { borderColor: "rgba(9,13,18,0.9)", borderWidth: 2 },
        },
      ],
    },
    true,
  );
}

function initAblationDeltaChart(charts) {
  const rows = Array.isArray(charts.top_ablation_deltas) ? charts.top_ablation_deltas : [];
  if (!els.ablationChart || !window.echarts) return;

  const chart = getChart("ablationChart", els.ablationChart);
  chart.setOption(
    {
      backgroundColor: "transparent",
      tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
      grid: { left: 88, right: 22, top: 18, bottom: 28 },
      xAxis: {
        type: "value",
        axisLabel: { color: "#9fb0ba" },
        splitLine: { lineStyle: { color: "rgba(123,226,217,0.12)" } },
      },
      yAxis: {
        type: "category",
        data: rows.map((item) => truncate(item.label, 42)),
        axisLabel: { color: "#d7e2e8" },
      },
      series: [
        {
          type: "bar",
          data: rows.map((item) => item.delta),
          itemStyle: {
            color: ({ value }) => (value < 0 ? "#79e7cf" : "#ff8c74"),
            borderRadius: [0, 6, 6, 0],
          },
          label: {
            show: true,
            position: "right",
            color: "#f4e6d0",
            formatter: ({ value }) => formatNumber(value),
          },
        },
      ],
    },
    true,
  );
}

function initTimelineChart(charts) {
  const rows = Array.isArray(charts.timeline) ? charts.timeline.filter((row) => row.day !== "unknown") : [];
  if (!els.timelineChart || !window.echarts) return;

  const chart = getChart("timelineChart", els.timelineChart);
  chart.setOption(
    {
      backgroundColor: "transparent",
      tooltip: { trigger: "axis" },
      legend: { textStyle: { color: "#c4d0d7" }, top: 0 },
      grid: { left: 44, right: 20, top: 36, bottom: 24 },
      xAxis: {
        type: "category",
        data: rows.map((row) => row.day),
        axisLabel: { color: "#9fb0ba" },
        axisLine: { lineStyle: { color: "rgba(123,226,217,0.24)" } },
      },
      yAxis: {
        type: "value",
        axisLabel: { color: "#9fb0ba" },
        splitLine: { lineStyle: { color: "rgba(123,226,217,0.12)" } },
      },
      series: [
        makeTimelineSeries("ok", rows, "#90efb6"),
        makeTimelineSeries("warn", rows, "#ffb066"),
        makeTimelineSeries("error", rows, "#ff7f74"),
        makeTimelineSeries("unknown", rows, "#95a5b4"),
      ],
    },
    true,
  );
}

function makeTimelineSeries(key, rows, color) {
  return {
    name: key,
    type: "line",
    smooth: true,
    symbolSize: 7,
    data: rows.map((row) => row[key] || 0),
    lineStyle: { width: 2, color },
    itemStyle: { color },
    areaStyle: { color: hexToRgba(color, 0.14) },
  };
}

function applyFilters() {
  const query = els.searchInput.value.trim().toLowerCase();
  const category = els.categoryFilter.value;
  const status = els.statusFilter.value;
  const metric = els.metricSelect.value;
  const direction = els.sortDir.value;

  const filtered = state.records.filter((record) => {
    if (category !== "all" && record.category !== category) return false;
    if (status !== "all" && record.status !== status) return false;
    if (!query) return true;

    const haystack = [
      record.path,
      record.rel_path,
      record.category,
      record.experiment_group,
      record.run_tag,
      record.timestamp_hint,
      record.snippet,
      ...(record.notes || []),
      ...(record.keywords || []),
      ...Object.entries(record.metrics || {}).map(([key, value]) => `${key} ${value}`),
    ]
      .filter(Boolean)
      .join(" ")
      .toLowerCase();

    return haystack.includes(query);
  });

  const sorted = [...filtered].sort((a, b) => compareRecords(a, b, metric, direction));
  state.filtered = sorted;

  updateVisibleStats(sorted, { query, category, status, metric, direction });
  els.resultCount.textContent = `${sorted.length} of ${state.records.length} records`;
  renderTable(sorted, metric);

  if (!sorted.length) {
    setDetail(null);
    updateBestMetricCard(metric, []);
    return;
  }

  const preferred = state.selectedId ? sorted.find((record) => record.id === state.selectedId) : null;
  setDetail(preferred || sorted[0]);
  updateBestMetricCard(metric, sorted);
}

function updateVisibleStats(records, filters) {
  const experiments = new Set(records.map((record) => record.experiment_group).filter(Boolean));
  const visibleErrors = records.filter((record) => record.status === "error").length;
  els.visibleCount.textContent = records.length;
  els.visibleExperiments.textContent = experiments.size;
  els.visibleErrors.textContent = visibleErrors;

  const bits = [];
  if (filters.query) bits.push(`query: ${filters.query}`);
  if (filters.category !== "all") bits.push(`category: ${filters.category}`);
  if (filters.status !== "all") bits.push(`status: ${filters.status}`);
  if (filters.metric) bits.push(`metric: ${filters.metric}`);
  bits.push(`order: ${filters.direction}`);
  els.activeQuery.textContent = bits.join(" | ");
}

function compareRecords(a, b, metric, direction) {
  const factor = direction === "asc" ? 1 : -1;
  if (metric) {
    const av = numericMetric(a, metric);
    const bv = numericMetric(b, metric);
    if (av !== bv) {
      if (Number.isFinite(av) && Number.isFinite(bv)) return (av - bv) * factor;
      if (Number.isFinite(av)) return -1 * factor;
      if (Number.isFinite(bv)) return 1 * factor;
    }
  }

  const at = a.timestamp_hint || "";
  const bt = b.timestamp_hint || "";
  if (at !== bt) {
    return at.localeCompare(bt) * -1;
  }
  return (a.path || "").localeCompare(b.path || "");
}

function renderTable(records, metric) {
  if (!records.length) {
    els.recordsBody.innerHTML = '<tr><td colspan="5" class="empty-cell">No records match the current filter deck.</td></tr>';
    return;
  }

  els.recordsBody.innerHTML = records
    .map((record) => {
      const status = record.status || "unknown";
      return `
        <tr class="record-row status-${escapeHtml(status)}" data-id="${escapeHtml(record.id)}">
          <td><span class="status-pill status-${escapeHtml(status)}">${escapeHtml(status)}</span></td>
          <td>
            <div class="record-main">
              <span class="record-title">${escapeHtml(record.run_tag || record.category || "record")}</span>
              <span class="record-sub">${escapeHtml(record.category || "unknown")}</span>
            </div>
          </td>
          <td>
            <div class="record-main">
              <span class="record-title">${escapeHtml(record.experiment_group || "unknown")}</span>
              <span class="record-sub">${escapeHtml(record.timestamp_hint || "no timestamp hint")}</span>
            </div>
          </td>
          <td>${formatMetricCell(record, metric)}</td>
          <td class="path-cell"><span class="record-path">${escapeHtml(record.rel_path || record.path || "")}</span></td>
        </tr>
      `;
    })
    .join("");

  els.recordsBody.querySelectorAll(".record-row").forEach((row) => {
    row.addEventListener("click", () => {
      const found = state.filtered.find((record) => record.id === row.dataset.id);
      if (found) setDetail(found);
    });
  });

  highlightSelected();
}

function formatMetricCell(record, metric) {
  if (metric && Number.isFinite(numericMetric(record, metric))) {
    return `<span class="metric-pill">${escapeHtml(metricLabels[metric] || metric)} ${escapeHtml(formatNumber(numericMetric(record, metric)))}</span>`;
  }

  const pairs = metricOrder
    .filter((key) => Number.isFinite(numericMetric(record, key)))
    .slice(0, 2)
    .map((key) => `${metricLabels[key] || key}: ${formatNumber(numericMetric(record, key))}`);

  return pairs.length ? escapeHtml(pairs.join(" | ")) : "-";
}

function setDetail(record) {
  state.selectedId = record ? record.id : null;
  highlightSelected();

  if (!record) {
    els.detailTitle.textContent = "No record selected";
    els.detailStatusBadge.textContent = "unknown";
    els.detailStatusBadge.className = "status-pill status-unknown";
    els.detailPathCode.textContent = "No path selected.";
    els.detailMeta.innerHTML = "";
    els.detailWriteup.innerHTML = "<p class=\"writeup-line\">Select a record to generate an analysis writeup.</p>";
    els.detailMetrics.innerHTML = '<div class="metric-item"><span>Metrics</span><strong>None extracted</strong></div>';
    els.detailNotes.innerHTML = '<li class="note-empty">No notes extracted.</li>';
    els.detailSnippet.textContent = "No record selected.";
    els.copyPathBtn.disabled = true;
    els.copyPathBtn.dataset.path = "";
    return;
  }

  const status = record.status || "unknown";
  els.detailTitle.textContent = record.run_tag || record.rel_path || record.path || record.id;
  els.detailStatusBadge.textContent = status;
  els.detailStatusBadge.className = `status-pill status-${status}`;
  els.detailPathCode.textContent = record.rel_path || record.path || "";

  els.copyPathBtn.disabled = false;
  els.copyPathBtn.dataset.path = record.path || record.rel_path || "";

  const metaRows = [
    ["Category", record.category],
    ["Experiment group", record.experiment_group],
    ["Run tag", record.run_tag],
    ["Timestamp hint", record.timestamp_hint],
    ["Status", status],
    ["Absolute path", record.path],
  ];

  els.detailMeta.innerHTML = metaRows
    .map(([label, value]) => `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value || "-")}</dd></div>`)
    .join("");

  els.detailWriteup.innerHTML = renderWriteupHtml(record);
  els.detailMetrics.innerHTML = renderMetricGrid(record.metrics || {});
  els.detailNotes.innerHTML = (record.notes || []).length
    ? record.notes.map((note) => `<li>${styleInlineSignal(note)}</li>`).join("")
    : '<li class="note-empty">No notes extracted.</li>';
  els.detailSnippet.textContent = record.snippet || "No snippet extracted.";
}

function renderWriteupHtml(record) {
  const metrics = record.metrics || {};
  const metricEntries = metricOrder.filter((key) => Number.isFinite(metrics[key]));
  const metricBits = metricEntries.length
    ? metricEntries.map((key) => `<span class="metric-strong">${escapeHtml(metricLabels[key] || key)}: ${escapeHtml(formatNumber(metrics[key]))}</span>`).join("<br>")
    : "<span class=\"writeup-muted\">No critical metrics extracted.</span>";

  const statusLine =
    record.status === "error"
      ? "This run is classified as a failure and should be treated as a blocker or invalid signal."
      : record.status === "warn"
        ? "This run is a watchlist signal (proxy or partial confidence) and needs confirmation."
        : record.status === "ok"
          ? "This run is currently a stable metric-bearing signal."
          : "Status is unknown; treat this as contextual evidence only.";

  const keywordLine = (record.keywords || []).length
    ? `Keywords: ${(record.keywords || []).map((kw) => styleInlineSignal(kw)).join(", ")}`
    : "Keywords: none extracted.";

  const snippetLine = record.snippet ? styleInlineSignal(record.snippet) : "No snippet extracted.";

  return `
    <p class="writeup-line">${styleInlineSignal(statusLine)}</p>
    <p class="writeup-line"><strong>Critical numbers</strong><br>${metricBits}</p>
    <p class="writeup-line"><strong>Signal summary</strong><br>${snippetLine}</p>
    <p class="writeup-line"><strong>Risk scan</strong><br>${keywordLine}</p>
  `;
}

function renderMetricGrid(metrics) {
  const entries = Object.entries(metrics);
  if (!entries.length) {
    return '<div class="metric-item"><span>Metrics</span><strong>None extracted</strong></div>';
  }

  return entries
    .sort((a, b) => sortMetricKeys(a[0], b[0]))
    .map(([key, value]) => `<div class="metric-item"><span>${escapeHtml(metricLabels[key] || key)}</span><strong>${escapeHtml(formatNumber(value))}</strong></div>`)
    .join("");
}

function sortMetricKeys(a, b) {
  const ai = metricOrder.indexOf(a);
  const bi = metricOrder.indexOf(b);
  if (ai === -1 && bi === -1) return a.localeCompare(b);
  if (ai === -1) return 1;
  if (bi === -1) return -1;
  return ai - bi;
}

function updateBestMetricCard(metric, records) {
  if (!metric) {
    els.bestMetricValue.textContent = "-";
    els.bestMetricLabel.textContent = "Choose a metric to identify the strongest record in view.";
    return;
  }

  const ranked = records
    .map((record) => ({ record, value: numericMetric(record, metric) }))
    .filter((item) => Number.isFinite(item.value))
    .sort((a, b) => a.value - b.value);

  if (!ranked.length) {
    els.bestMetricValue.textContent = "-";
    els.bestMetricLabel.textContent = `No ${metric} values in the current slice.`;
    return;
  }

  const best = ranked[0];
  els.bestMetricValue.textContent = formatNumber(best.value);
  els.bestMetricLabel.textContent = `${metricLabels[metric] || metric} lead: ${best.record.run_tag || best.record.rel_path || best.record.path}`;
}

function highlightSelected() {
  document.querySelectorAll(".record-row").forEach((row) => {
    row.classList.toggle("selected", row.dataset.id === state.selectedId);
  });
}

function resetFilters() {
  els.searchInput.value = "";
  els.categoryFilter.value = "all";
  els.statusFilter.value = "all";
  els.sortDir.value = "asc";
  els.metricSelect.value = metricOrder.find((key) => els.metricSelect.querySelector(`option[value="${key}"]`))?.value || "";
  applyFilters();
}

async function copySelectedPath() {
  const path = els.copyPathBtn.dataset.path || "";
  if (!path) return;
  try {
    await navigator.clipboard.writeText(path);
    const original = els.copyPathBtn.textContent;
    els.copyPathBtn.textContent = "Copied";
    setTimeout(() => {
      els.copyPathBtn.textContent = original;
    }, 1200);
  } catch (error) {
    console.error(error);
  }
}

function numericMetric(record, metric) {
  const value = record?.metrics?.[metric];
  return typeof value === "number" ? value : Number.NaN;
}

function formatNumber(value) {
  if (!Number.isFinite(value)) return "-";
  if (Math.abs(value) >= 100) return value.toFixed(2);
  if (Math.abs(value) >= 10) return value.toFixed(3);
  return value.toFixed(4);
}

function formatBytes(value) {
  if (!Number.isFinite(value) || value <= 0) return "n/a";
  const units = ["B", "KB", "MB", "GB"];
  let bytes = value;
  let idx = 0;
  while (bytes >= 1024 && idx < units.length - 1) {
    bytes /= 1024;
    idx += 1;
  }
  return `${bytes.toFixed(idx === 0 ? 0 : 2)} ${units[idx]}`;
}

function formatConsiderationValue(item) {
  if (!item || !Number.isFinite(item.value)) return "n/a";
  const metricUsed = (item.metric_used || item.metric_key || "").toLowerCase();
  if (metricUsed === "model_size_bytes" || item.category === "lowest_file_size") {
    return formatBytes(item.value);
  }
  return formatNumber(item.value);
}

function formatTimestamp(value) {
  if (!value) return "Unknown";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function styleInlineSignal(raw) {
  let out = escapeHtml(String(raw || ""));

  out = out.replace(/(\b(?:val_bpb|cap_val_bpb|diag_bpb|sliding_bpb|ngram9_bpb|delta)\b\s*[:=]?\s*)([+-]?\d+(?:\.\d+)?)/gi, '<span class="metric-strong">$1$2</span>');

  const errorRegex = new RegExp(`\\b(${errorKeywords.map(escapeRegex).join("|")})\\b`, "gi");
  const warnRegex = new RegExp(`\\b(${warnKeywords.map(escapeRegex).join("|")})\\b`, "gi");

  out = out.replace(errorRegex, '<span class="keyword-error">$1</span>');
  out = out.replace(warnRegex, '<span class="keyword-warn">$1</span>');

  return out;
}

function escapeRegex(input) {
  return input.replace(/[.*+?^${}()|[\\]\\]/g, "\\$&");
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function truncate(text, max) {
  const value = String(text || "");
  if (value.length <= max) return value;
  return `${value.slice(0, max - 1)}...`;
}

function getChart(key, el) {
  if (!state.charts[key]) {
    state.charts[key] = window.echarts.init(el);
  }
  return state.charts[key];
}

function resizeCharts() {
  Object.values(state.charts).forEach((chart) => {
    try {
      chart.resize();
    } catch (error) {
      console.error(error);
    }
  });
}

function hexToRgba(hex, alpha) {
  const h = hex.replace("#", "");
  const bigint = Number.parseInt(h, 16);
  const r = (bigint >> 16) & 255;
  const g = (bigint >> 8) & 255;
  const b = bigint & 255;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}
