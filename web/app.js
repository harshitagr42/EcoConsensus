// EcoConsensus website: upload -> confirm column and label mapping -> follow the stages -> results and downloads.
// Every limit is enforced by the backend; this page only shows them and explains every error.
"use strict";

const CFG = window.ECO_CONFIG || {};
const API = (new URLSearchParams(location.search).get("api") || CFG.API_BASE_URL || "").replace(/\/+$/, "");
const SPACE_PAGE = CFG.SPACE_PAGE_URL || "";
const CLASSES = ["favour", "neutral", "against"];
const CLASS_TEXT = { favour: "favour", neutral: "neutral (None)", against: "against", ignore: "ignore (not scored)" };
const LS = { limits: "eco.limits", job: "eco.job" };

const $ = (s) => document.querySelector(s);
const state = { limits: null, upload: null, jobId: null, poll: null, health: null, offline: false };

// ---------------------------------------------------------------- helpers
function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null) continue;
    if (k === "class") e.className = v;
    else if (k === "text") e.textContent = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  for (const k of kids.flat()) if (k != null) e.append(k instanceof Node ? k : document.createTextNode(String(k)));
  return e;
}
const store = {
  get(k) { try { return JSON.parse(localStorage.getItem(k)); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* private mode */ } },
  del(k) { try { localStorage.removeItem(k); } catch { /* ignore */ } },
};
const fmt = (v, d = 3) => (v == null || Number.isNaN(v) ? "—" : Number(v).toFixed(d));
const pct = (v) => (v == null ? "—" : `${Math.round(v * 100)} %`);
const mins = (s) => (s >= 3600 ? `${Math.round(s / 3600)} h` : `${Math.max(1, Math.round(s / 60))} min`);

class ApiError extends Error {
  constructor(kind, message, status, retryAfter) { super(message); this.kind = kind; this.status = status; this.retryAfter = retryAfter; }
}

async function api(path, opts = {}) {
  if (!API || API.includes("YOUR-HF-USERNAME")) {
    throw new ApiError("config", "The website is not connected to a backend yet (API_BASE_URL in config.js).");
  }
  let r;
  try {
    r = await fetch(API + path, opts);
  } catch {
    throw new ApiError("offline", "The backend cannot be reached. It is paused, asleep or restarting.");
  }
  let body = null;
  try { body = await r.json(); } catch { /* not JSON */ }
  if (!r.ok) {
    const retry = Number(r.headers.get("Retry-After")) || (body && body.retry_after) || null;
    const kind = r.status === 429 ? "rate" : r.status === 503 ? "busy" : r.status === 404 ? "missing" : "http";
    let msg = (body && body.detail) || `The backend answered with HTTP ${r.status}.`;
    if (!body && (r.status === 502 || r.status === 503)) msg = "The backend is starting or asleep.";
    throw new ApiError(kind, msg, r.status, retry);
  }
  return body;
}

function banner(msg, { warn = false, wake = false } = {}) {
  const b = $("#banner");
  if (!msg) { b.hidden = true; return; }
  b.hidden = false;
  b.className = "banner" + (warn ? " warn" : "");
  b.replaceChildren(el("span", { text: msg }));
  if (wake && SPACE_PAGE && !SPACE_PAGE.includes("YOUR-HF-USERNAME")) {
    b.append(" ", el("a", { href: SPACE_PAGE, target: "_blank", rel: "noopener", text: "Open the backend page to wake it up" }),
      " (takes about a minute), then reload this page.");
  }
}

function explain(e) {
  if (e.kind === "rate") return `${e.message}${e.retryAfter ? ` (about ${mins(e.retryAfter)})` : ""}`;
  return e.message;
}

// ---------------------------------------------------------------- backend status + limits
async function checkHealth() {
  const pill = $("#status");
  try {
    const h = await api("/health");
    state.health = h;
    state.offline = false;
    const llmIssues = [];
    for (const l of Object.values(h.llm || {})) {
      if (!l.configured) llmIssues.push(`${l.name} is not configured`);
      else if (l.quota_reached) llmIssues.push(`${l.name}: daily quota used up on every key`);
      else if (l.last_problem) llmIssues.push(`${l.name}: ${l.last_problem}`);
    }
    if (h.models === "ready") {
      pill.className = "pill pill-ok";
      pill.textContent = h.running ? `Online · busy (${h.queue} waiting)` : "Online";
    } else if (h.models === "loading") {
      pill.className = "pill pill-wait";
      pill.textContent = "Starting… (models loading)";
    } else {
      pill.className = "pill pill-bad";
      pill.textContent = "Backend error";
    }
    if (h.models === "error") banner(`The backend cannot load its models: ${h.models_error}`);
    else if (llmIssues.length) banner(`${llmIssues.join(" · ")}. Predictions still work: rows without an LLM answer use the ensemble without that LLM.`, { warn: true });
    else if (!$("#banner").dataset.sticky) banner(null);
  } catch (e) {
    state.offline = true;
    pill.className = "pill pill-bad";
    pill.textContent = e.kind === "config" ? "Not connected" : "Offline";
    banner(e.kind === "config" ? e.message
      : "The backend is offline (paused, asleep or restarting). Uploads are not possible right now; this page checks again every 30 s.",
      { wake: e.kind !== "config" });
  }
}

async function loadLimits() {
  try {
    const l = await api("/limits");
    state.limits = l;
    store.set(LS.limits, l);
    renderLimits(l, false);
  } catch {
    const cached = store.get(LS.limits);
    if (cached) { state.limits = cached; renderLimits(cached, true); }
    else $("#limits").replaceChildren(el("li", { text: "Limits are shown when the backend is online." }));
  }
}

function renderLimits(l, stale) {
  const you = l.you || {};
  const items = [
    `CSV files only, at most ${l.max_rows.toLocaleString()} rows and ${l.max_file_mb} MB.`,
    `${l.jobs_per_ip_per_hour} prediction runs per hour per visitor` +
      (stale || you.left_this_hour == null ? "." : ` — you have ${you.left_this_hour} left this hour.`),
    `${l.jobs_per_day} runs per day for all visitors together` +
      (stale || you.left_today == null ? "." : ` — ${you.left_today} left today.`),
    `One file is processed at a time; others wait in a queue (at most ${l.queue_max}).`,
    "About 1–3 minutes for 1,000 rows (both transformers on CPU, then both LLMs).",
    "If an LLM's daily quota runs out (after trying every key), the run still finishes; those rows use the ensemble without that LLM, and the results say so.",
    `Uploads and results are deleted after ${l.result_ttl_hours} hours.`,
  ];
  $("#limits").replaceChildren(...items.map((t) => el("li", { text: t })));
  $("#limits-note").textContent = stale ? "Last known limits (the backend is offline)." : "";
  document.querySelectorAll(".ttl").forEach((n) => (n.textContent = l.result_ttl_hours));
  if (l.default_claim) $("#claim").textContent = l.default_claim;
}

// ---------------------------------------------------------------- 1. upload
function show(step) {
  for (const id of ["step-upload", "step-map", "step-run", "step-results"]) $("#" + id).hidden = !step.includes(id);
}

async function onFile(file) {
  if (!file) return;
  banner(null);
  $("#drop-text").textContent = `Uploading ${file.name}…`;
  const lim = state.limits;
  if (lim && file.size > lim.max_file_mb * 1048576) {
    $("#drop-text").textContent = "Choose a CSV file or drop it here";
    banner(`${file.name} is ${(file.size / 1048576).toFixed(1)} MB; the limit is ${lim.max_file_mb} MB.`);
    return;
  }
  const fd = new FormData();
  fd.append("file", file);
  try {
    state.upload = await api("/inspect", { method: "POST", body: fd });
    renderMapping();
    show(["step-map"]);
  } catch (e) {
    banner(explain(e), { wake: e.kind === "offline" });
  } finally {
    $("#drop-text").textContent = "Choose a CSV file or drop it here";
    $("#file").value = "";
  }
}

// ---------------------------------------------------------------- 2. mapping
function colSelect(id, first, guess) {
  const s = el("select", { id });
  if (first) s.append(el("option", { value: "", text: first }));
  else s.append(el("option", { value: "", text: "Choose…" }));
  for (const c of state.upload.columns) s.append(el("option", { value: c, text: c }));
  s.value = guess || "";
  s.addEventListener("change", () => { if (id === "c-label") renderLabelTable(); validate(); });
  return s;
}

function renderMapping() {
  const u = state.upload;
  $("#file-info").textContent = `${u.filename}: ${u.rows.toLocaleString()} rows, ${u.columns.length} columns` +
    (u.delimiter && u.delimiter !== "," ? ` (separator “${u.delimiter}”)` : "");
  // preview
  const head = el("tr", {}, u.columns.map((c) => el("th", { text: c })));
  const rows = u.preview.map((r) => el("tr", {}, u.columns.map((c) => el("td", { text: String(r[c] ?? "").slice(0, 120) }))));
  $("#preview").replaceChildren(el("thead", {}, head), el("tbody", {}, rows));

  const g = u.guess || {};
  const claimNote = g.claim
    ? `Column “${g.claim}” looks like a claim/topic column. The models were trained on the default claim only, so it is not used unless you pick it here.`
    : `Default: “${(state.limits && state.limits.default_claim) || $("#claim").textContent}”`;
  const tr = (what, sel, note) => el("tr", {}, el("td", { text: what }), el("td", {}, sel), el("td", { class: "muted small", text: note }));
  $("#colmap tbody").replaceChildren(
    tr("Comment text *", colSelect("c-comment", null, g.comment), "The text that is classified."),
    tr("Claim", colSelect("c-claim", "Use the default claim", ""), claimNote),
    tr("True stance (for metrics)", colSelect("c-label", "None — predictions only, no metrics", g.label),
      "The correct answer per row, if your file has it (it may be called stance, label, target, prediction…)."),
    tr("Row ID", colSelect("c-id", "Row number (1, 2, 3, …)", g.id), "Copied to the prediction file."),
  );
  // output style
  const styles = (state.limits && state.limits.output_styles) ||
    { file: "same values as the uploaded file", ps: "Favour / None / Against", numeric: "our ids 0 / 1 / 2" };
  $("#style").replaceChildren(...Object.entries(styles).map(([k, v]) => el("option", { value: k, text: `Predictions as ${v}` })));
  $("#style").onchange = validate;
  renderLabelTable();
  validate();
}

function renderLabelTable() {
  const u = state.upload;
  const col = $("#c-label").value;
  $("#labels-block").hidden = !col;
  $("#style").value = col ? "file" : "ps";
  if (!col) return;
  const vals = u.values[col];
  const body = $("#labelmap tbody");
  if (!vals) {
    body.replaceChildren(el("tr", {}, el("td", { colspan: 4, class: "err",
      text: `“${col}” has more than 30 different values, so it cannot be a stance column. Choose another column or “None”.` })));
    return;
  }
  const guesses = (u.label_guess || {})[col] || {};
  body.replaceChildren(...vals.map(({ value, count }) => {
    const s = el("select", { "data-value": value });
    s.append(el("option", { value: "", text: "Choose…" }));
    for (const c of [...CLASSES, "ignore"]) s.append(el("option", { value: c, text: CLASS_TEXT[c] }));
    s.value = guesses[value] || "";
    const id = el("td", { class: "num-cell" });
    const upd = () => { id.textContent = CLASSES.includes(s.value) ? String(CLASSES.indexOf(s.value)) : "—"; validate(); };
    s.addEventListener("change", upd);
    const shown = value === "" ? "(empty)" : value;
    const row = el("tr", {}, el("td", {}, el("code", { text: shown })), el("td", { class: "num-cell", text: count }), el("td", {}, s), id);
    id.textContent = CLASSES.includes(s.value) ? String(CLASSES.indexOf(s.value)) : "—";
    return row;
  }));
}

function currentMapping() {
  const label = $("#c-label").value;
  const map = {};
  if (label) document.querySelectorAll("#labelmap select").forEach((s) => (map[s.dataset.value] = s.value));
  return {
    upload_id: state.upload.upload_id,
    comment_col: $("#c-comment").value,
    claim_col: $("#c-claim").value || null,
    label_col: label || null,
    id_col: $("#c-id").value || null,
    label_map: map,
    output_style: $("#style").value,
  };
}

function validate() {
  const m = currentMapping();
  const errors = [], warns = [], infos = [];
  const chosen = [m.comment_col, m.claim_col, m.label_col, m.id_col].filter(Boolean);
  if (!m.comment_col) errors.push("Choose the comment column.");
  if (new Set(chosen).size !== chosen.length) errors.push("One column is chosen for two roles.");
  $("#c-comment").classList.toggle("need", !m.comment_col);

  if (m.label_col) {
    if (!state.upload.values[m.label_col]) errors.push("The chosen stance column has too many different values.");
    const entries = Object.entries(m.label_map);
    const unset = entries.filter(([, v]) => !v).map(([k]) => (k === "" ? "(empty)" : k));
    document.querySelectorAll("#labelmap select").forEach((s) => s.classList.toggle("need", !s.value));
    if (unset.length) errors.push(`Say what these values mean: ${unset.join(", ")}.`);
    const numeric = entries.filter(([k]) => /^-?\d+(\.0+)?$/.test(k.trim()));
    if (numeric.length) infos.push("Numeric values: check them carefully — in your file 0 may mean something other than our 0 (favour).");
    for (const c of CLASSES) {
      const which = entries.filter(([, v]) => v === c).map(([k]) => k);
      if (which.length > 1) warns.push(`${which.join(", ")} are all mapped to ${c}. Fine if they are different spellings of the same stance; a mistake otherwise.`);
    }
    const used = new Set(entries.map(([, v]) => v));
    const missing = CLASSES.filter((c) => !used.has(c));
    if (entries.length && !unset.length && missing.length < 3 && missing.length)
      infos.push(`No value is mapped to ${missing.join(", ")}; that is fine if your file has no such rows.`);
    if (entries.length && !unset.length && missing.length === 3) errors.push("Map at least one value to favour, neutral or against.");
  }
  const msgs = $("#label-msgs");
  msgs.replaceChildren(...errors.map((t) => el("li", { class: "bad", text: t })),
    ...warns.map((t) => el("li", { class: "warn", text: t })), ...infos.map((t) => el("li", { class: "muted", text: t })));

  // plain-language plan
  const style = m.output_style;
  const styleText = { file: "the same values as your file", ps: "Favour / None / Against", numeric: "our ids 0 / 1 / 2" }[style];
  $("#style-note").textContent = style === "file"
    ? "Uses your mapping in reverse, e.g. if your 0 means neutral, a neutral prediction is written as 0."
    : style === "numeric" ? "0 = favour, 1 = neutral (None), 2 = against." : "";
  const parts = [`${state.upload.rows.toLocaleString()} rows; comments from “${m.comment_col || "?"}”`,
    m.claim_col ? `claim from “${m.claim_col}”` : "the default claim",
    m.label_col ? `true stance from “${m.label_col}” (${Object.entries(m.label_map).filter(([, v]) => v)
      .map(([k, v]) => `${k === "" ? "(empty)" : k} → ${v}`).join(", ") || "values not mapped yet"})`
      : "no true stance: predictions only, no metrics",
    `predictions written as ${styleText}`];
  $("#plan").textContent = "Plan: " + parts.join("; ") + ".";
  $("#run").disabled = errors.length > 0 || state.offline;
  return errors.length === 0;
}

async function run() {
  if (!validate()) return;
  const m = currentMapping();
  if (!m.label_col) {
    const ok = confirm("No true-stance column is selected, so no metrics can be computed — only predictions.\n\nRun anyway?");
    if (!ok) return;
    m.confirm_no_labels = true;
  }
  $("#run").disabled = true;
  $("#map-error").hidden = true;
  try {
    const job = await api("/jobs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(m) });
    startPolling(job.id);
    loadLimits();
  } catch (e) {
    $("#map-error").textContent = explain(e);
    $("#map-error").hidden = false;
    if (e.kind === "offline") checkHealth();
    $("#run").disabled = false;
  }
}

// ---------------------------------------------------------------- 3. progress
function startPolling(id) {
  state.jobId = id;
  store.set(LS.job, id);
  show(["step-run"]);
  $("#run-error").hidden = true;
  clearTimeout(state.poll);
  poll();
}

let pollFailures = 0;
async function poll() {
  try {
    const j = await api(`/jobs/${state.jobId}`);
    pollFailures = 0;
    renderStages(j);
    if (j.status === "done") return loadResults();
    if (j.status === "error") {
      $("#run-error").textContent = `The run failed: ${j.error}`;
      $("#run-error").hidden = false;
      $("#step-run").append(el("div", { class: "actions" }, el("button", { class: "btn ghost", onclick: reset, text: "Start again" })));
      store.del(LS.job);
      return;
    }
  } catch (e) {
    if (e.kind === "missing") {
      store.del(LS.job);
      banner("That run is no longer on the server (results expire, and a backend restart clears them). Please upload the file again.", { warn: true });
      return reset();
    }
    pollFailures += 1;
    if (pollFailures >= 3) banner("Lost contact with the backend; still retrying. If it restarted, the run is lost and you need to upload again.", { warn: true, wake: true });
  }
  state.poll = setTimeout(poll, 2000);
}

function renderStages(j) {
  $("#run-file").textContent = `· ${j.filename} (${(j.rows || 0).toLocaleString()} rows) · ${Math.round(j.elapsed)} s`;
  $("#queue").hidden = !j.position;
  $("#queue").textContent = j.position ? `Waiting in the queue: position ${j.position}. Another file is being processed.` : "";
  const icon = { done: "✓", warning: "!", error: "✕" };
  $("#stages").replaceChildren(...j.stages.map((s) => {
    const what = el("div", { class: "what" }, el("div", { text: s.label }), s.detail ? el("div", { class: "detail", text: s.detail }) : null);
    if (s.state === "running" && s.total && s.done != null) what.append(el("div", { class: "bar" }, el("span", { style: `width:${Math.min(100, (100 * s.done) / s.total)}%` })));
    return el("li", { class: `st-${s.state}` }, el("span", { class: "icon", text: icon[s.state] || "" }), what,
      el("span", { class: "time", text: s.seconds != null ? `${s.seconds} s` : "" }));
  }));
  $("#run-warnings").replaceChildren(...(j.warnings || []).map((w) => el("li", { text: w })));
}

// ---------------------------------------------------------------- 4. results
async function loadResults() {
  let d;
  try { d = await api(`/jobs/${state.jobId}/results`); } catch (e) { banner(explain(e)); return; }
  const r = d.results;
  show(["step-run", "step-results"]);
  const ens = (r.metrics || []).find((m) => m.key === "stacked");
  const stat = (v, t) => el("div", { class: "stat" }, el("b", { text: v }), el("span", { text: t }));
  $("#stats").replaceChildren(
    stat(r.rows.toLocaleString(), "rows"),
    stat(r.labelled.toLocaleString(), "with a true stance"),
    stat(ens ? fmt(ens.macro_f1) : "—", "ensemble Macro-F1"),
    stat(`${Math.round(d.elapsed)} s`, "processing time"),
  );
  // downloads
  const base = `${API}/jobs/${state.jobId}/files/`;
  $("#downloads").replaceChildren(...Object.entries(d.files).map(([name, desc]) =>
    el("a", { class: name === "all.zip" ? "btn" : "btn ghost", href: base + encodeURIComponent(name), title: desc, text: `⬇ ${name}` })));
  // metrics
  $("#metrics-block").hidden = !(r.metrics && r.metrics.length);
  if (r.metrics && r.metrics.length) {
    const best = Math.max(...r.metrics.map((m) => m.macro_f1 ?? -1));
    const head = el("tr", {}, ["System", "Macro-F1", "95 % CI", "Accuracy", "F1 favour", "F1 neutral", "F1 against", "Coverage", "Rows"].map((t) => el("th", { text: t })));
    const rows = r.metrics.map((m) => el("tr", { class: m.macro_f1 === best ? "best" : null },
      el("td", { text: m.system }), el("td", { class: "num-cell", text: fmt(m.macro_f1) }),
      el("td", { class: "num-cell", text: m.ci95_low != null ? `${fmt(m.ci95_low)}–${fmt(m.ci95_high)}` : "—" }),
      el("td", { class: "num-cell", text: fmt(m.accuracy) }), el("td", { class: "num-cell", text: fmt(m.f1_favour) }),
      el("td", { class: "num-cell", text: fmt(m.f1_neutral) }), el("td", { class: "num-cell", text: fmt(m.f1_against) }),
      el("td", { class: "num-cell", text: pct(m.coverage) }), el("td", { class: "num-cell", text: m.n })));
    $("#metrics").replaceChildren(el("thead", {}, head), el("tbody", {}, rows));
    if (r.confusion && r.confusion.matrix) {
      const L = r.confusion.labels;
      const h = el("tr", {}, el("th", { text: "true ↓ / predicted →" }), L.map((l) => el("th", { text: l })));
      const b = r.confusion.matrix.map((row, i) => el("tr", {}, el("th", { text: L[i] }), row.map((v, j) => el("td", { class: i === j ? "right" : null, text: v }))));
      $("#confusion").replaceChildren(el("thead", {}, h), el("tbody", {}, b));
    }
  }
  // distribution (always useful; the main table when there are no labels)
  const dist = r.distribution || {};
  $("#dist-block").hidden = !Object.keys(dist).length;
  $("#dist").replaceChildren(el("thead", {}, el("tr", {}, ["Model", "favour", "neutral", "against", "no answer"].map((t) => el("th", { text: t })))),
    el("tbody", {}, Object.entries(dist).map(([k, v]) => el("tr", {}, el("td", { text: k }),
      ["favour", "neutral", "against", "no answer"].map((c) => el("td", { class: "num-cell", text: v[c] }))))));
  // mapping used
  const mp = r.mapping;
  const lm = Object.entries(r.label_map || {});
  const llm = Object.entries(r.llm || {}).map(([k, i]) => `${k === "llm_a" ? "LLM-A" : "LLM-B"}${i.model ? ` (${i.model})` : ""}: ${i.answered ?? 0} answered` +
    (i.key_used > 1 ? `, used key ${i.key_used} of ${i.keys}` : "") + (i.status !== "ok" ? ` (${i.stopped || i.status})` : ""));
  const variants = Object.entries(r.variants || {}).map(([k, v]) => `${k}: ${v}`).join(", ");
  $("#used").replaceChildren(el("ul", {},
    el("li", { text: `Comments: “${mp.comment_col}”; claim: ${mp.claim_col ? `“${mp.claim_col}”` : "default claim"}; ID: ${mp.id_col ? `“${mp.id_col}”` : "row number"}` }),
    el("li", { text: mp.label_col ? `True stance: “${mp.label_col}” — ${lm.map(([k, v]) => `${k === "" ? "(empty)" : k} → ${v}${CLASSES.includes(v) ? ` (our ${CLASSES.indexOf(v)})` : ""}`).join(", ")}` : "No true stance (no metrics)." }),
    el("li", { text: `Predictions written as ${r.output_style_text}.` }),
    llm.length ? el("li", { text: `LLMs — ${llm.join(" · ")}` }) : null,
    el("li", { text: `Meta-model used per row — ${variants}` }),
  ));
  // predictions preview
  const cols = r.columns;
  const head = el("tr", {}, cols.map(([, t]) => el("th", { text: t })));
  const rows = r.preview.map((p) => el("tr", {}, cols.map(([k]) => {
    let cls = k === "comment" ? "comment" : null;
    if (k === "final") cls = "final" + (p.gold_class ? (p.final_class === p.gold_class ? " right" : " wrong") : "");
    return el("td", { class: cls, text: p[k] ?? "" });
  })));
  $("#preds").replaceChildren(el("thead", {}, head), el("tbody", {}, rows));
  $("#report").textContent = r.report || "";
  store.del(LS.job);
}

// ---------------------------------------------------------------- start
function reset() {
  clearTimeout(state.poll);
  state.upload = null;
  state.jobId = null;
  store.del(LS.job);
  document.querySelectorAll("#step-run .actions").forEach((n) => n.remove());
  show(["step-upload"]);
  loadLimits();
}

function init() {
  $("#file").addEventListener("change", (e) => onFile(e.target.files[0]));
  const drop = $("#drop");
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => onFile(e.dataTransfer.files[0]));
  $("#run").addEventListener("click", run);
  $("#reset1").addEventListener("click", reset);
  $("#reset2").addEventListener("click", reset);
  show(["step-upload"]);
  checkHealth().then(loadLimits);
  setInterval(checkHealth, 30000);
  const pending = store.get(LS.job);
  if (pending) startPolling(pending);
}

init();
