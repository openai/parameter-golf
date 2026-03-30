const state = {
  data: null,
  records: [],
  filtered: [],
  selectedId: null,
};

const els = {};

const metricLabels = {
  val_bpb: "val_bpb",
  cap_val_bpb: "cap_val_bpb",
  diag_bpb: "diag_bpb",
  sliding_bpb: "sliding_bpb",
  ngram9_bpb: "ngram9_bpb",
  delta: "delta",
};

const metricOrder = ["val_bpb", "cap_val_bpb", "diag_bpb", "sliding_bpb", "ngram9_bpb", "delta"];

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
    "statTotal",
    "statOk",
    "statWarn",
    "statError",
    "bestMetricValue",
    "bestMetricLabel",
    "searchInput",
    "categoryFilter",
    "statusFilter",
    "metricSelect",
    "sortDir",
    "resetBtn",
    "recordsBody",
    "resultCount",
    "detailTitle",
    "detailMeta",
    "detailSnippet",
    "detailMetrics",
    "detailNotes",
    "copyPathBtn",
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
}

async function loadData() {
  try {
    const response = await fetch("./data/hub_index.json", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Failed to load index: ${response.status}`);
    }
    state.data = await response.json();
    state.records = Array.isArray(state.data.records) ? state.data.records : [];
    hydrateControls();
    renderHeader();
    applyFilters();
  } catch (error) {
    renderLoadFailure(error);
  }
}

function renderLoadFailure(error) {
  els.resultCount.textContent = "Unable to load hub index.";
  els.recordsBody.innerHTML = `<tr><td colspan="6" class="path-cell">Error: ${escapeHtml(error.message)}</td></tr>`;
  els.detailTitle.textContent = "Load failed";
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
  placeholder.textContent = "Sort by path";
  select.appendChild(placeholder);

  values
    .filter((value) => value && value !== "all")
    .forEach((value) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = value;
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
  els.scopeLabel.textContent = (state.data?.source_roots || []).join(", ") || "experiments";
  els.generatedAt.textContent = formatTimestamp(state.data?.generated_at);
  els.statTotal.textContent = counts.total_records ?? state.records.length;
  els.statOk.textContent = counts.by_status?.ok ?? 0;
  els.statWarn.textContent = counts.by_status?.warn ?? 0;
  els.statError.textContent = counts.by_status?.error ?? 0;
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
    if (query) {
      const haystack = [
        record.path,
        record.rel_path,
        record.category,
        record.experiment_group,
        record.run_tag,
        record.timestamp_hint,
        record.snippet,
        ...(record.notes || []),
        ...Object.entries(record.metrics || {}).map(([key, value]) => `${key} ${value}`),
      ]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();
      if (!haystack.includes(query)) return false;
    }
    return true;
  });

  const sorted = [...filtered].sort((a, b) => compareRecords(a, b, metric, direction));
  state.filtered = sorted;

  els.resultCount.textContent = `${sorted.length} of ${state.records.length} records`;
  renderTable(sorted, metric);

  if (!sorted.length) {
    setDetail(null);
    return;
  }

  const preferred = state.selectedId ? sorted.find((record) => record.id === state.selectedId) : null;
  setDetail(preferred || sorted[0]);
  updateBestMetricCard(metric, sorted);
}

function compareRecords(a, b, metric, direction) {
  const factor = direction === "asc" ? 1 : -1;
  if (metric && metric !== "all") {
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
  if (at !== bt) return at.localeCompare(bt) * factor;
  return (a.path || "").localeCompare(b.path || "") * factor;
}

function renderTable(records, metric) {
  els.recordsBody.innerHTML = "";
  const fragment = document.createDocumentFragment();
  records.forEach((record) => {
    const row = document.createElement("tr");
    row.dataset.id = record.id;
    row.className = `record-row status-${record.status || "unknown"}`;
    row.innerHTML = `
      <td data-col="status"><span class="status-pill status-${record.status || "unknown"}">${escapeHtml(record.status || "unknown")}</span></td>
      <td data-col="category">${escapeHtml(record.category || "unknown")}</td>
      <td data-col="experiment">${escapeHtml(record.experiment_group || "unknown")}</td>
      <td data-col="runTag">${escapeHtml(record.run_tag || "unknown")}</td>
      <td data-col="metric">${formatMetricCell(record, metric)}</td>
      <td data-col="path" class="path-cell">${escapeHtml(record.rel_path || record.path || "")}</td>
    `;
    row.addEventListener("click", () => setDetail(record));
    fragment.appendChild(row);
  });
  els.recordsBody.appendChild(fragment);
  highlightSelected();
}

function formatMetricCell(record, metric) {
  if (metric && metric !== "all" && Number.isFinite(numericMetric(record, metric))) {
    return `<span class="metric-pill">${escapeHtml(metric)} ${escapeHtml(formatNumber(numericMetric(record, metric)))}</span>`;
  }

  const pairs = metricOrder
    .filter((key) => Number.isFinite(numericMetric(record, key)))
    .slice(0, 2)
    .map((key) => `${key}: ${formatNumber(numericMetric(record, key))}`);

  return pairs.length ? escapeHtml(pairs.join(" | ")) : "—";
}

function setDetail(record) {
  state.selectedId = record ? record.id : null;
  highlightSelected();

  if (!record) {
    els.detailTitle.textContent = "No record selected";
    els.detailMeta.innerHTML = "";
    els.detailSnippet.textContent = "No record selected.";
    els.detailMetrics.innerHTML = "";
    els.detailNotes.innerHTML = "";
    els.copyPathBtn.disabled = true;
    return;
  }

  els.detailTitle.textContent = record.path || record.rel_path || record.id;
  els.copyPathBtn.disabled = false;
  els.copyPathBtn.dataset.path = record.path || record.rel_path || "";

  const metaRows = [
    ["Category", record.category],
    ["Experiment group", record.experiment_group],
    ["Run tag", record.run_tag],
    ["Timestamp hint", record.timestamp_hint],
    ["Status", record.status],
    ["Path", record.rel_path || record.path],
  ];
  els.detailMeta.innerHTML = metaRows
    .map(([label, value]) => `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value || "—")}</dd></div>`)
    .join("");

  els.detailSnippet.textContent = record.snippet || "No snippet extracted.";
  els.detailMetrics.innerHTML = renderMetricGrid(record.metrics || {});
  els.detailNotes.innerHTML = (record.notes || []).length
    ? (record.notes || []).map((note) => `<li>${escapeHtml(note)}</li>`).join("")
    : "<li>No notes extracted.</li>";
}

function renderMetricGrid(metrics) {
  const entries = Object.entries(metrics);
  if (!entries.length) {
    return '<div class="metric-item"><span>Metrics</span><strong>None extracted</strong></div>';
  }
  return entries
    .sort((a, b) => metricOrder.indexOf(a[0]) - metricOrder.indexOf(b[0]))
    .map(([key, value]) => `<div class="metric-item"><span>${escapeHtml(key)}</span><strong>${escapeHtml(formatNumber(value))}</strong></div>`)
    .join("");
}

function updateBestMetricCard(metric, records) {
  if (!metric || metric === "all") {
    els.bestMetricValue.textContent = "—";
    els.bestMetricLabel.textContent = "Select a metric to rank records";
    return;
  }
  const ranked = records
    .map((record) => ({ record, value: numericMetric(record, metric) }))
    .filter((item) => Number.isFinite(item.value));
  if (!ranked.length) {
    els.bestMetricValue.textContent = "—";
    els.bestMetricLabel.textContent = `No ${metric} values available in this slice`;
    return;
  }
  ranked.sort((a, b) => a.value - b.value);
  const best = ranked[0];
  els.bestMetricValue.textContent = `${formatNumber(best.value)}`;
  els.bestMetricLabel.textContent = `${metricLabels[metric] || metric} best on ${best.record.run_tag || best.record.rel_path || best.record.path}`;
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
  els.sortDir.value = "desc";
  els.metricSelect.value = metricOrder.find((key) => els.metricSelect.querySelector(`option[value="${key}"]`))?.value || "";
  applyFilters();
}

async function copySelectedPath() {
  const path = els.copyPathBtn.dataset.path || "";
  if (!path) return;
  try {
    await navigator.clipboard.writeText(path);
    els.copyPathBtn.textContent = "Copied";
    setTimeout(() => {
      els.copyPathBtn.textContent = "Copy path";
    }, 1200);
  } catch {
    const ta = document.createElement("textarea");
    ta.value = path;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand("copy");
    ta.remove();
  }
}

function numericMetric(record, metric) {
  const value = record?.metrics?.[metric];
  return typeof value === "number" ? value : Number.parseFloat(value);
}

function formatNumber(value) {
  if (!Number.isFinite(Number(value))) return "—";
  const num = Number(value);
  if (Math.abs(num) >= 1000) return num.toFixed(0);
  if (Math.abs(num) >= 100) return num.toFixed(1);
  if (Math.abs(num) >= 10) return num.toFixed(2);
  return num.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
}

function formatTimestamp(value) {
  if (!value) return "Unknown";
  return value;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}
