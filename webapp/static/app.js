/* SEAL · PPG — frontend controller.
 *
 * Vanilla JS, Plotly for plots. The page is one document; rendering replaces
 * DOM subtrees and Plotly.react keeps charts in place. While a capture is
 * running we poll /api/recording/status once per second; otherwise everything
 * is event-driven (session click → signals + analysis in parallel).
 */
"use strict";

// ── design tokens — keep in sync with style.css ─────────────────────────────
const C = {
  ink:    "#ece4d3",
  inkMute:"#7d8294",
  inkFaint:"#4d5363",
  hair:   "#232c3b",
  hairSoft:"#1a2230",
  panel:  "#11161f",
  panel2: "#161c27",
  ecg:    "#ff5e5e",
  ecgSoft:"rgba(255,94,94,0.15)",
  good:   "#87d49b",
  warn:   "#e8b070",
  bad:    "#ef6a6a",
  chPalette: ["#6ba8f0", "#e8b070", "#87d49b", "#c982ff",
              "#ff9bb5", "#82d8ff", "#ffd960", "#b0ddff"],
};
const SITES = ["", "finger", "forehead", "earlobe", "shoulder", "wrist", "other"];
const N_CHANNELS = 8;
const POLL_MS = 1000;

const PLOT_BASE = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor:  "rgba(0,0,0,0)",
  font: {
    family: "system-ui, -apple-system, 'Segoe UI', Roboto, Arial, sans-serif",
    color: C.inkMute, size: 12,
  },
  margin: { l: 56, r: 18, t: 14, b: 44 },
  hovermode: "x",
};
const PLOT_CFG = { displaylogo: false, responsive: true,
                   modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d"] };

// ── state ────────────────────────────────────────────────────────────────────
const state = {
  selectedSession: null,
  sessionDetail: null,
  analysis: null,            // last /analyze response for selectedSession
  signals: null,
  sessions: [],
  batch: null,               // last /api/analyze_all response (every-session view)
  batchSource: null,         // "fresh" | {batch_id, created_at} when loaded from archive
  batchArchives: [],         // list of {batch_id, created_at, n_sessions_analyzed, crop_window}
  history: null,             // last /api/sessions/{name}/history response
  historyOpen: false,        // is the run-history list expanded
  pollTimer: null,
  channelSites: loadSiteMap(),
  bpChannels: new Set(),     // PPG channels currently showing the 0.6-3.3 Hz bandpass
  window: { start: null, end: null },  // pre-processing crop, seconds since session t0
  batchSiteSort: { col: null, dir: 1 },  // sortable per-site table state
  modalStack: [],            // active modal ids (for Escape handler)
};

// ── DOM helpers ──────────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);

window.addEventListener("DOMContentLoaded", () => {
  buildSitesGrid();
  refreshPorts();
  refreshSessions().then(() => {
    if (state.sessions.length) selectSession(state.sessions[0].name);
  });
  refreshBatchArchives();
  pollStatus();
  restoreSidebarState();

  $("port-refresh").onclick = refreshPorts;
  $("btn-start").onclick    = startRecording;
  $("btn-stop").onclick     = stopRecording;
  $("btn-analyze").onclick  = () => runAnalysis(true);
  $("btn-reload-signals").onclick = () => loadSignals(state.selectedSession);
  $("btn-save-meta").onclick = saveMetadata;
  $("btn-analyze-all").onclick = () => runBatchAnalysis(true);
  $("btn-batch-rerun").onclick = () => runBatchAnalysis(true);
  $("btn-batch-close").onclick = closeBatchView;
  $("btn-batch-export").onclick = exportBatchCSV;
  $("btn-bp-all").onclick = toggleAllBandpass;
  $("btn-win-apply").onclick = applyWindow;
  $("btn-win-full").onclick  = clearWindow;
  ["win-start", "win-end"].forEach(id => {
    $(id).addEventListener("keydown", e => { if (e.key === "Enter") applyWindow(); });
  });
  $("btn-delete").onclick = openDeleteModal;
  $("delete-cancel").onclick = closeDeleteModal;
  $("delete-confirm").onclick = performDelete;
  // Delete stays disabled until the user explicitly picks the delete
  // option from the dropdown — the deliberate confirmation gesture.
  $("delete-confirm-select").onchange = (e) => {
    setEnabled("delete-confirm", e.target.value === "delete");
  };
  $("session-filter").oninput = renderSessionList;
  document.querySelectorAll(".col-toggle").forEach(btn => {
    btn.addEventListener("click", () => toggleSidebar(parseInt(btn.dataset.col, 10)));
  });

  // Receiver-log modal
  $("btn-receiver-log").onclick = openReceiverLog;
  $("recv-log-close").onclick = () => closeModal("recv-log-modal");
  $("recv-log-refresh").onclick = refreshReceiverLog;

  // Batch-archive modal
  $("btn-browse-archive").onclick = openArchiveModal;
  $("batch-archive-close").onclick = () => closeModal("batch-archive-modal");

  // Run-history controls
  $("btn-history-toggle").onclick = toggleHistory;
  $("btn-history-reload").onclick = () => loadHistory(state.selectedSession, true);

  // Onboarding
  $("btn-help").onclick = () => openOnboarding(true);
  $("onb-next").onclick = onbNext;
  $("onb-back").onclick = onbBack;
  $("onb-skip-cb").onchange = (e) => {
    try { localStorage.setItem("seal_ppg_onboarded", e.target.checked ? "1" : ""); }
    catch {}
  };

  // Wire all modals with backdrop-click + Escape via the central helper.
  ["delete-modal", "recv-log-modal", "batch-archive-modal", "onb-modal"].forEach(wireModal);
  document.addEventListener("keydown", onGlobalKeydown);

  // First-launch onboarding
  if (!localStorage.getItem("seal_ppg_onboarded")) {
    setTimeout(() => openOnboarding(false), 500);
  }
});

// Generic modal helpers — every new modal goes through these so the Escape
// stack stays well-defined and backdrop clicks work uniformly.
function wireModal(id) {
  const el = $(id);
  if (!el) return;
  el.addEventListener("click", (e) => {
    if (e.target.id === id) closeModal(id);  // backdrop click
  });
}
function openModal(id) {
  $(id)?.classList.remove("hidden");
  if (!state.modalStack.includes(id)) state.modalStack.push(id);
}
function closeModal(id) {
  $(id)?.classList.add("hidden");
  state.modalStack = state.modalStack.filter(x => x !== id);
  // delete-modal has its own internal reset semantics handled in
  // closeDeleteModal — but keep the simple cases consistent.
}
function onGlobalKeydown(e) {
  if (e.key === "Escape" && state.modalStack.length) {
    const top = state.modalStack[state.modalStack.length - 1];
    closeModal(top);
  }
}

function toggleSidebar(n) {
  const main = $("main");
  const cls = "hide-col" + n;
  const hide = !main.classList.contains(cls);
  main.classList.toggle(cls, hide);
  updateToggleIcon(n, hide);
  try { localStorage.setItem("seal_ppg_hide_col" + n, hide ? "1" : ""); } catch {}
  // Let CSS finish its transition before Plotly remeasures.
  setTimeout(() => {
    document.querySelectorAll(".plot, .ppg-plot, .ba-plot").forEach(el => {
      if (el._fullLayout) Plotly.Plots.resize(el);
    });
  }, 220);
}

function updateToggleIcon(n, collapsed) {
  // « when expanded (click to collapse leftward); » when collapsed (click to expand).
  const btn = document.querySelector(`.col-toggle[data-col="${n}"]`);
  if (btn) btn.textContent = collapsed ? "»" : "«";
}

function restoreSidebarState() {
  const main = $("main");
  for (const n of [1, 2]) {
    const hidden = !!(localStorage.getItem("seal_ppg_hide_col" + n));
    main.classList.toggle("hide-col" + n, hidden);
    updateToggleIcon(n, hidden);
  }
}


/* ═══════════════════════════════════════════════════════════════════════════
   RECORDING
   ═══════════════════════════════════════════════════════════════════════════ */

async function startRecording() {
  const port = $("port-select").value;
  const baud = parseInt($("baud-input").value, 10) || 115200;
  if (!port) { flash("Pick a serial port first."); return; }

  const participant = collectMetadataForm();
  setEnabled("btn-start", false);
  setStatus("recording", "recording");

  try {
    const r = await postJSON("/api/recording/start", {port, baud, participant});
    await refreshSessions();
    selectSession(r.session_name, {liveMode: true});
  } catch (e) {
    setStatus("idle", "idle");
    setEnabled("btn-start", true);
    flash("Start failed: " + e.message);
    return;
  }
  setEnabled("btn-stop", true);
  $("record-log").classList.add("visible");
  startPolling();
}

async function stopRecording() {
  setEnabled("btn-stop", false);
  try { await postJSON("/api/recording/stop", {}); }
  catch (e) { flash("Stop failed: " + e.message); }
  stopPolling();
  setStatus("idle", "idle");
  setEnabled("btn-start", true);
  $("record-log").classList.remove("visible");
  await refreshSessions();
  if (state.selectedSession) {
    await loadSessionFull(state.selectedSession);  // signals + auto-analysis
  }
}

function startPolling() {
  stopPolling();
  state.pollTimer = setInterval(pollStatus, POLL_MS);
}
function stopPolling() {
  if (state.pollTimer) clearInterval(state.pollTimer);
  state.pollTimer = null;
}

async function pollStatus() {
  let s;
  try { s = await fetch("/api/recording/status").then(r => r.json()); }
  catch { return; }

  if (s.active) {
    setStatus("recording", "recording");
    setEnabled("btn-start", false);
    setEnabled("btn-stop", true);
    $("record-log").classList.add("visible");
    renderCounters(s.sample_counts);
    renderLog(s.recent_log);
    if (!state.pollTimer) startPolling();
    if (state.selectedSession !== s.session_name && s.session_name) {
      selectSession(s.session_name, {liveMode: true});
    }
    setChip("recording → " + s.session_name);
    // Refresh the live signal plots every status tick so the operator
    // can see signal quality (or lack of it) mid-run.
    if (s.session_name) pollLiveSignals(s.session_name);
  } else if (state.pollTimer) {
    // Transition: active → inactive. Surface unexpected exits.
    stopPolling();
    setStatus("idle", "idle");
    setEnabled("btn-start", true);
    setEnabled("btn-stop", false);
    $("record-log").classList.remove("visible");
    setChip("");
    await refreshSessions();
    if (s.exit_code != null && s.exit_code !== 0) {
      const tail = (s.recent_log || []).slice(-12).join("\n");
      flash(`Receiver exited (code ${s.exit_code}).\n${tail}`);
    }
  }
}


/* ═══════════════════════════════════════════════════════════════════════════
   PORTS / SESSION LIST
   ═══════════════════════════════════════════════════════════════════════════ */

async function refreshPorts() {
  let ports = [];
  try { ports = await fetch("/api/ports").then(r => r.json()); }
  catch { ports = []; }
  const sel = $("port-select");
  const prev = sel.value;
  sel.innerHTML = "";
  if (!ports.length) {
    const opt = document.createElement("option");
    opt.value = ""; opt.textContent = "no ports detected";
    sel.appendChild(opt);
    return;
  }
  ports.forEach(p => {
    const opt = document.createElement("option");
    opt.value = p.device;
    opt.textContent = p.device + (p.description ? " · " + p.description : "");
    sel.appendChild(opt);
  });
  if (prev) sel.value = prev;
}

async function refreshSessions() {
  state.sessions = await fetch("/api/sessions").then(r => r.json());
  renderSessionList();
}

function renderSessionList() {
  const ul = $("session-list");
  ul.innerHTML = "";
  const filter = ($("session-filter").value || "").toLowerCase();

  const visible = state.sessions.filter(s => {
    const pid = s.participant?.participant_id || "";
    return !filter || (s.name + " " + pid).toLowerCase().includes(filter);
  });

  $("session-count").textContent = filter
    ? `${visible.length} of ${state.sessions.length}`
    : `${state.sessions.length}`;

  if (!visible.length) {
    const li = document.createElement("li");
    li.className = "session-list-empty";
    li.textContent = state.sessions.length ? "no match" : "no sessions yet";
    ul.appendChild(li);
    return;
  }

  visible.forEach(s => {
    const li = document.createElement("li");
    li.className = "session-card" + (s.name === state.selectedSession ? " active" : "");
    const pid = s.participant?.participant_id || "";
    const fst = s.participant?.fitzpatrick;
    const ts = (s.started_at || s.name).replace("T", " ").slice(0, 19);
    const channelTags = [
      ...(s.has_ecg ? ['<span class="ch ecg">ecg</span>'] : []),
      ...(s.channels || []).map(c => `<span class="ch">ch${c}</span>`),
    ];

    // Persistence badges: small chips that distinguish (a) analyzed
    // sessions from (b) sessions with only a history log from (c)
    // untouched sessions, plus an analysis-count chip on heavy-use rows.
    const isActive = s.name === state.selectedSession;
    const analyzedAgo = s.last_analyzed_at ? relTimeAgo(s.last_analyzed_at) : null;
    const badges = [];
    if (analyzedAgo) badges.push(`<span class="badge analyzed" title="Last analysis ${s.last_analyzed_at}">analysed ${analyzedAgo}</span>`);
    if ((s.analysis_count || 0) > 1) badges.push(`<span class="badge count" title="Total analysis runs">×${s.analysis_count}</span>`);
    if (!s.last_analyzed_at && (s.history_count || 0) > 0) {
      badges.push(`<span class="badge log-only" title="History events present but no cached analysis">log only</span>`);
    }
    const analyzedDot = analyzedAgo ? '<span class="analyzed-dot" aria-label="analyzed"></span>' : "";

    li.innerHTML = `
      <div class="row1">
        <span class="pid${pid ? "" : " empty"}">${analyzedDot}${pid || "unassigned"}</span>
        ${fst ? `<span class="fst">FST ${"I II III IV V VI".split(" ")[fst-1]}</span>` : ""}
      </div>
      <span class="ts">${ts}</span>
      <div class="channels">${channelTags.length ? channelTags.join("") : '<span class="none">no signals</span>'}</div>
      ${badges.length ? `<div class="badges">${badges.join("")}</div>` : ""}`;
    li.tabIndex = 0;
    li.onclick = () => selectSession(s.name);
    li.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); selectSession(s.name); } };
    // Avoid double-active highlight when state.selectedSession === s.name
    // and the card is also focus-visible. CSS handles both via existing rule.
    if (isActive) { /* class already applied above */ }
    ul.appendChild(li);
  });
}

// Lightweight ISO-timestamp -> "5m ago" / "3h ago" / "2d ago" helper.
function relTimeAgo(iso) {
  if (!iso) return null;
  const ms = Date.now() - Date.parse(iso);
  if (!isFinite(ms) || ms < 0) return null;
  const s = Math.floor(ms / 1000);
  if (s < 60)    return s + "s ago";
  if (s < 3600)  return Math.floor(s / 60) + "m ago";
  if (s < 86400) return Math.floor(s / 3600) + "h ago";
  if (s < 86400 * 14) return Math.floor(s / 86400) + "d ago";
  return Math.floor(s / 86400) + "d ago";
}


/* ═══════════════════════════════════════════════════════════════════════════
   SESSION DETAIL
   ═══════════════════════════════════════════════════════════════════════════ */

async function selectSession(name, opts = {}) {
  state.selectedSession = name;
  state.analysis = null;
  state.signals = null;
  state.history = null;
  renderSessionList();
  $("no-session").classList.add("hidden");
  $("batch").classList.add("hidden");        // batch and per-session views are mutually exclusive
  $("detail").classList.remove("hidden");
  $("detail-title").innerHTML = formatSessionTitle(name);
  setChip("→ " + name);
  setEnabled("btn-analyze", true);
  setEnabled("btn-reload-signals", true);
  setEnabled("btn-save-meta", true);
  syncWindowInputs();   // window persists across sessions; show it

  // Reset the per-session interpretation + history UI for the new session.
  $("session-interp-block").classList.add("hidden");
  $("history-list").classList.add("hidden");
  $("history-list").innerHTML = "";
  $("btn-history-toggle").textContent = "▸ Show history";
  $("btn-history-toggle").setAttribute("aria-expanded", "false");
  state.historyOpen = false;

  try {
    const s = await fetch(`/api/sessions/${name}`).then(r => r.json());
    state.sessionDetail = s;
    renderMeta(s);
    populateMetadataForm(s.participant);
    // Receiver log only meaningful when the file actually exists on disk.
    setEnabled("btn-receiver-log", !!s.has_receiver_log);
    $("btn-receiver-log").title = s.has_receiver_log
      ? "View the captured receiver subprocess log"
      : "No receiver.log captured for this session (older recording, or capture failed)";
    // Lazy history summary line — count from the meta payload; full list
    // streams in when the user toggles the panel open.
    const histCount = s.history_count || 0;
    $("history-summary").textContent = histCount
      ? `${histCount} event${histCount === 1 ? "" : "s"} on disk`
      : "no events recorded";
    $("btn-history-toggle").textContent = histCount
      ? `▸ Show history (${histCount})`
      : "▸ Show history";
    setEnabled("btn-history-toggle", histCount > 0);
  } catch (e) { console.error(e); }

  if (opts.liveMode) {
    // Recording is active — show signals only (no SQI/BA on partial data).
    $("sqi-block").classList.add("hidden");
    $("ba-block").classList.add("hidden");
    pollLiveSignals(name);
  } else {
    await loadSessionFull(name);
  }
}

async function pollLiveSignals(name) {
  if (state.selectedSession !== name) return;   // user navigated away
  try {
    const sig = await fetch(
      `/api/sessions/${name}/signals?max_points=1500&tail_seconds=30`
    ).then(r => r.json());
    state.signals = sig;
    renderECGBlock();
    renderPPGBlock();
  } catch { /* receiver may not have flushed first row yet */ }
}

// Build the &start_s=..&end_s=.. suffix for the active crop window.
// Only emits a bound that's actually set, so "from 10s on" works.
function windowQS() {
  const { start, end } = state.window;
  let qs = "";
  if (start != null) qs += `&start_s=${start}`;
  if (end != null)   qs += `&end_s=${end}`;
  return qs;
}

async function loadSessionFull(name) {
  // Kick signals + analysis in parallel; render whichever returns first.
  // Both carry the same crop window so the plots and the SQI/CCC table
  // describe exactly the same span of data.
  $("analysis-status").textContent = "loading…";
  setStatus("analyzing", "analyzing");
  const w = windowQS();
  const sigP = fetch(`/api/sessions/${name}/signals?max_points=4500${w}`).then(r => r.json());
  const anaP = fetch(`/api/sessions/${name}/analyze?${w.slice(1)}`, {method:"POST"}).then(r => r.json());

  const [sig, ana] = await Promise.all([
    sigP.catch(e => ({error: e.message})),
    anaP.catch(e => ({error: e.message})),
  ]);

  state.signals = sig;
  state.analysis = ana;
  setStatus("idle", "idle");
  $("analysis-status").textContent = "";
  renderEverything();
}

// Reload only the signal traces (ECG + PPG) for the current crop window,
// without re-running the SQI/CCC analysis. Backs the "Reload signals"
// button (previously wired to an undefined function).
async function loadSignals(name) {
  if (!name) return;
  try {
    state.signals = await fetch(
      `/api/sessions/${name}/signals?max_points=4500${windowQS()}`
    ).then(r => r.json());
    renderECGBlock();
    renderPPGBlock();
  } catch (e) { flash("Reload failed: " + e.message); }
}

async function runAnalysis(showBusy) {
  const name = state.selectedSession;
  if (!name) return;
  if (showBusy) {
    $("analysis-status").textContent = "running…";
    setStatus("analyzing", "analyzing");
  }
  try {
    state.analysis = await fetch(
      `/api/sessions/${name}/analyze?${windowQS().slice(1)}`, { method: "POST" }
    ).then(r => { if (!r.ok) throw new Error(r.statusText); return r.json(); });
    renderEverything();
  } catch (e) { flash("Analysis failed: " + e.message); }
  finally {
    setStatus("idle", "idle");
    $("analysis-status").textContent = "";
  }
}

function applyWindow() {
  const s = parseFloat($("win-start").value);
  const e = parseFloat($("win-end").value);
  const start = isFinite(s) ? s : null;
  const end   = isFinite(e) ? e : null;
  if (start != null && end != null && start >= end) {
    flash("Window start must be less than end.");
    return;
  }
  state.window = { start, end };
  syncWindowInputs();
  if (state.selectedSession) loadSessionFull(state.selectedSession);
}

function clearWindow() {
  state.window = { start: null, end: null };
  $("win-start").value = "";
  $("win-end").value = "";
  syncWindowInputs();
  if (state.selectedSession) loadSessionFull(state.selectedSession);
}

// Reflect the active window into the inputs and flag the control as
// active so it's visually obvious the data is cropped.
function syncWindowInputs() {
  const { start, end } = state.window;
  $("win-start").value = start != null ? start : "";
  $("win-end").value   = end != null ? end : "";
  document.querySelector(".range-ctl")
    .classList.toggle("active", start != null || end != null);
}

function renderEverything() {
  renderECGBlock();
  renderPPGBlock();
  renderSessionInterpretation();
  renderSQITable();
  renderBlandAltmanGrid();
}

function formatSessionTitle(name) {
  // session_20260516_142931 → "2026-05-16  14:29:31"
  const m = name.match(/^session_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$/);
  if (!m) return name;
  return `${m[1]}-${m[2]}-${m[3]}<span class="sep">·</span>${m[4]}:${m[5]}:${m[6]}`;
}

function renderMeta(s) {
  const p = s.participant || {};
  const fst = p.fitzpatrick ? ["I","II","III","IV","V","VI"][p.fitzpatrick - 1] : null;
  const sites = p.channel_sites || {};
  const siteSummary = s.channels.map(c =>
    `<span class="mono">ch${c}</span>${sites[c] ? " · " + sites[c] : ""}`
  ).join("&ensp;");

  $("meta-grid").innerHTML = `
    ${cell("PID",      p.participant_id, "serif")}
    ${cell("FST",      fst, "ecg")}
    ${cell("Duration", s.duration_s != null ? s.duration_s.toFixed(1) + " s" : null, "mono")}
    ${cell("Channels", s.channels.length ? s.channels.join(",") : null, "mono")}
    ${cell("ECG",      s.has_ecg ? "present" : "absent", s.has_ecg ? "ecg" : "mute")}
    ${cell("Started",  (s.started_at || "").replace("T", " ").slice(0,19), "mono")}
  `;

  function cell(k, v, cls) {
    const empty = !v;
    return `<div class="meta-cell">
      <div class="k">${k}</div>
      <div class="v ${empty ? "mute" : cls || ""}">${empty ? "—" : v}</div>
    </div>`;
  }
}


/* ═══════════════════════════════════════════════════════════════════════════
   ECG HERO BLOCK
   ═══════════════════════════════════════════════════════════════════════════ */

function renderECGBlock() {
  const ecg = state.signals?.ecg;
  const block = $("ecg-block");
  if (!ecg) {
    block.classList.add("hidden");
    return;
  }
  block.classList.remove("hidden");

  const ana = state.analysis?.ecg;
  const parts = [`${ecg.n_samples.toLocaleString()} samp`,
                 `${fmtFs(ecg.fs_hz)}`];
  if (ana?.n_peaks) parts.push(`${ana.n_peaks} R-peaks`);
  if (ana?.mean_hr_bpm) parts.push(`HR ${ana.mean_hr_bpm.toFixed(0)} bpm`);
  if (ana?.leads_off_samples) parts.push(`${ana.leads_off_samples} leads-off`);
  $("ecg-summary").textContent = parts.join("  ·  ");

  const traces = [{
    x: ecg.time_s, y: ecg.signal,
    type: "scatter", mode: "lines",
    line: { width: 1, color: C.ecg },
    name: "ECG",
    hovertemplate: "%{x:.2f}s · %{y:.0f}<extra></extra>",
  }];

  if (ana?.peak_times_s?.length) {
    const ys = interpY(ecg.time_s, ecg.signal, ana.peak_times_s);
    traces.push({
      x: ana.peak_times_s, y: ys,
      type: "scatter", mode: "markers",
      marker: { size: 6, color: C.ecg, line: { color: "#1a0606", width: 1 } },
      name: "R-peaks",
      hovertemplate: "R-peak @ %{x:.2f}s<extra></extra>",
    });
  }

  const shapes = (ecg.leads_off_spans || []).map(([a, b]) => ({
    type: "rect", xref: "x", yref: "paper",
    x0: a, x1: b, y0: 0, y1: 1,
    fillcolor: C.ecgSoft, line: { width: 0 },
  }));

  Plotly.react("ecg-plot", traces, {
    ...PLOT_BASE,
    height: 480,
    showlegend: false,
    xaxis: axisStyle({ title: "time (s)", showticks: true }),
    yaxis: axisStyle({ title: "ECG (ADC)", showticks: true }),
    shapes,
  }, PLOT_CFG);
}


/* ═══════════════════════════════════════════════════════════════════════════
   PPG CHANNEL CARDS
   ═══════════════════════════════════════════════════════════════════════════ */

function renderPPGBlock() {
  const chans = state.signals?.channels || [];
  const block = $("ppg-block");
  const grid = $("ppg-grid");
  // Tear down the previous session's Plotly instances before innerHTML
  // discards their divs, so their listeners/internals don't leak.
  purgePlots(grid);
  if (!chans.length) {
    block.classList.add("hidden");
    grid.innerHTML = "";
    $("btn-bp-all").classList.add("hidden");
    return;
  }
  block.classList.remove("hidden");

  const sites = state.sessionDetail?.participant?.channel_sites || {};
  const totalSamp = chans.reduce((a, c) => a + c.n_samples, 0);
  $("ppg-summary").textContent = `${chans.length} channels  ·  ${totalSamp.toLocaleString()} samp total`;

  // signal_bp is the 0.6-3.3 Hz cardiac bandpass (computed server-side,
  // same filter as ppgvis.py); null when the channel's fs/length can't
  // support it, in which case its checkbox is disabled.
  const anyBpAble = chans.some(c => c.signal_bp);
  $("btn-bp-all").classList.toggle("hidden", !anyBpAble);

  grid.innerHTML = chans.map(c => {
    const able = !!c.signal_bp;
    const on = able && state.bpChannels.has(c.channel);
    return `
    <div class="ppg-card">
      <div class="pcap">
        <span class="name">CH${c.channel}</span>
        <span class="site">${sites[c.channel] || "—"}</span>
        <span class="meta">${c.n_samples.toLocaleString()} @ ${fmtFs(c.fs_hz)}</span>
        <label class="bp-toggle${able ? "" : " disabled"}">
          <input type="checkbox" class="bp-cb" data-ch="${c.channel}"
                 ${on ? "checked" : ""} ${able ? "" : "disabled"}> bandpass
        </label>
      </div>
      <div class="ppg-plot" id="ppg-plot-${c.channel}"></div>
    </div>`;
  }).join("");

  grid.querySelectorAll(".bp-cb").forEach(cb => {
    cb.addEventListener("change", () => {
      const ch = parseInt(cb.dataset.ch, 10);
      if (cb.checked) state.bpChannels.add(ch);
      else state.bpChannels.delete(ch);
      drawPPGChannel(chans.find(c => c.channel === ch));
    });
  });

  chans.forEach(drawPPGChannel);
}

function drawPPGChannel(c) {
  if (!c) return;
  const color = chColor(c.channel);
  const useBp = !!c.signal_bp && state.bpChannels.has(c.channel);
  const xx = useBp ? c.time_bp_s : c.time_s;
  const yy = useBp ? c.signal_bp : c.signal;

  const traces = [{
    x: xx, y: yy,
    type: "scatter", mode: "lines",
    line: { width: 1, color },
    hovertemplate: "%{x:.2f}s · %{y:.0f}<extra></extra>",
  }];
  // Peak markers were detected on the raw signal; anchoring them to the
  // displayed series (raw or bandpassed) by timestamp keeps them aligned.
  const r = (state.analysis?.results || []).find(x => x.channel === c.channel);
  if (r?.ppg_peak_times_s?.length) {
    const ys = interpY(xx, yy, r.ppg_peak_times_s);
    traces.push({
      x: r.ppg_peak_times_s, y: ys,
      type: "scatter", mode: "markers",
      marker: { size: 4, color, line: { color: C.panel, width: 1 } },
      hovertemplate: "PPG peak @ %{x:.2f}s<extra></extra>",
    });
  }
  Plotly.react("ppg-plot-" + c.channel, traces, {
    ...PLOT_BASE,
    height: 360,
    showlegend: false,
    xaxis: axisStyle({ title: "time (s)", showticks: true }),
    yaxis: axisStyle({ title: useBp ? "PPG bandpassed (0.6-3.3 Hz)" : "PPG (ADC)", showticks: true }),
  }, PLOT_CFG);
}

function toggleAllBandpass() {
  const chans = (state.signals?.channels || []).filter(c => c.signal_bp);
  if (!chans.length) return;
  // If any filterable channel is still raw, turn all on; otherwise all off.
  const turnOn = chans.some(c => !state.bpChannels.has(c.channel));
  chans.forEach(c => {
    if (turnOn) state.bpChannels.add(c.channel);
    else state.bpChannels.delete(c.channel);
  });
  document.querySelectorAll(".bp-cb").forEach(cb => {
    cb.checked = state.bpChannels.has(parseInt(cb.dataset.ch, 10));
  });
  chans.forEach(drawPPGChannel);
}


/* ═══════════════════════════════════════════════════════════════════════════
   SQI TABLE
   ═══════════════════════════════════════════════════════════════════════════ */

function renderSQITable() {
  const r = state.analysis;
  const tbl = $("sqi-table");
  const block = $("sqi-block");
  if (!r || r.error) {
    block.classList.remove("hidden");
    tbl.innerHTML = `<thead><tr><th>${r?.error || "Analysis pending"}</th></tr></thead>`;
    return;
  }
  block.classList.remove("hidden");
  const goodCount = (r.results || []).filter(x => x.stats && x.stats.ccc > 0.95).length;
  $("sqi-summary").textContent = `${(r.results || []).length} channels  ·  ${goodCount} ≥ 0.95 CCC`;

  // 12 columns total: ZSQI μ/σ and RMSE/MAE are paired into single cells
  // so the table fits the main column without horizontal scrolling.
  let html = `<thead><tr>
    <th>Channel</th><th>Site</th>
    <th>fs (Hz)</th><th>SSQI</th>
    <th>ZSQI μ ± σ</th>
    <th>Matched</th>
    <th>CCC</th><th>ICC</th><th>Pearson</th>
    <th>Bias (ms)</th><th>LOA± (ms)</th>
    <th>RMSE / MAE</th>
  </tr></thead><tbody>`;

  // ECG reference row at top — it's the agreement denominator
  if (r.ecg) {
    const e = r.ecg;
    html += `<tr>
      <td class="ch-name ecg">ECG ref</td>
      <td class="site">Einthoven</td>
      <td>${fmt(e.fs_hz, 1)}</td>
      <td colspan="3" class="muted">${e.n_peaks} R-peaks · HR ${fmt(e.mean_hr_bpm, 0)} bpm · ${e.leads_off_samples} leads-off</td>
      <td colspan="6" class="muted">—</td>
    </tr>`;
  }

  (r.results || []).forEach(row => {
    const s = row.stats;
    const cccCls = s ? gradeCCC(s.ccc) : "";
    const iccCls = s ? gradeCCC(s.icc) : "";
    html += `<tr>
      <td class="ch-name">ch${row.channel}</td>
      <td class="site">${row.site || "—"}</td>
      <td>${fmt(row.ppg_fs_hz, 1)}</td>
      <td>${fmt(row.ssqi, 3)}</td>
      <td>${fmt(row.zsqi_mean, 3)} ± ${fmt(row.zsqi_std, 3)}</td>
      <td>${row.n_matched_beats}</td>
      <td class="${cccCls}">${s ? fmt(s.ccc, 3) : "—"}</td>
      <td class="${iccCls}">${s ? fmt(s.icc, 3) : "—"}</td>
      <td>${s ? fmt(s.pearson_r, 3) : "—"}</td>
      <td>${s ? fmtSigned(s.bias_ms, 1) : "—"}</td>
      <td>${s ? fmtSigned(s.loa_lower_ms, 0) + " / " + fmtSigned(s.loa_upper_ms, 0) : "—"}</td>
      <td>${s ? fmt(s.rmse_ms, 1) + " / " + fmt(s.mae_ms, 1) : "—"}</td>
    </tr>`;
    if (row.error) {
      html += `<tr class="err-row"><td colspan="12">⚠ ${row.error}</td></tr>`;
    }
  });

  tbl.innerHTML = html + "</tbody>";
}


/* ═══════════════════════════════════════════════════════════════════════════
   BLAND-ALTMAN GRID
   ═══════════════════════════════════════════════════════════════════════════ */

function renderBlandAltmanGrid() {
  const r = state.analysis;
  const block = $("ba-block");
  const grid = $("ba-grid");
  purgePlots(grid);
  const withStats = (r?.results || []).filter(x => x.stats);
  if (!withStats.length) {
    block.classList.add("hidden");
    grid.innerHTML = "";
    return;
  }
  block.classList.remove("hidden");

  grid.innerHTML = withStats.map(row => `
    <div class="ba-cell">
      <div class="ba-head">
        <span class="name">CH${row.channel}</span>
        <span class="site">${row.site || "—"}</span>
      </div>
      <div class="ba-stats">CCC <b>${fmt(row.stats.ccc, 3)}</b>
         · bias <b>${fmtSigned(row.stats.bias_ms, 1)} ms</b>
         · LOA <b>${fmtSigned(row.stats.loa_lower_ms, 0)} / ${fmtSigned(row.stats.loa_upper_ms, 0)} ms</b>
         · n=<b>${row.n_matched_beats}</b></div>
      <div class="ba-plot" id="ba-${row.channel}"></div>
    </div>
  `).join("");

  withStats.forEach(row => {
    const rr = row.stats.matched_rr_ms;
    const ppi = row.stats.matched_ppi_ms;
    const means = rr.map((v, i) => (v + ppi[i]) / 2);
    const diffs = rr.map((v, i) => ppi[i] - v);
    const color = chColor(row.channel);

    // Regular scatter (canvas, not WebGL) — the matched-beat count per
    // channel is tens, not thousands, so SVG/canvas is cheaper and we
    // dodge the per-page WebGL-context limit that bites with one GL
    // chart per PPG channel plus one for the hero ECG.
    Plotly.react("ba-" + row.channel, [{
      x: means, y: diffs,
      type: "scatter", mode: "markers",
      marker: { size: 5, color, opacity: 0.85,
                line: { color: C.panel, width: 0.5 } },
      hovertemplate: "mean %{x:.0f}ms · diff %{y:.0f}ms<extra></extra>",
    }], {
      ...PLOT_BASE,
      height: 380,
      showlegend: false,
      xaxis: axisStyle({ title: "mean (ms)", showticks: true }),
      yaxis: axisStyle({ title: "PPI − RR (ms)", showticks: true, side: "left" }),
      shapes: [
        hline(row.stats.bias_ms,      C.inkMute, "dash"),
        hline(row.stats.loa_upper_ms, C.ecg,    "dot"),
        hline(row.stats.loa_lower_ms, C.ecg,    "dot"),
      ],
    }, PLOT_CFG);
  });
}


/* ═══════════════════════════════════════════════════════════════════════════
   METADATA FORM
   ═══════════════════════════════════════════════════════════════════════════ */

function buildSitesGrid() {
  const g = $("sites-grid");
  for (let i = 0; i < N_CHANNELS; i++) {
    const tile = document.createElement("div");
    tile.className = "site-tile";
    tile.id = "site-tile-" + i;
    const ch = document.createElement("span");
    ch.className = "ch"; ch.textContent = "ch" + i;
    const sel = document.createElement("select");
    sel.id = "site-ch" + i;
    SITES.forEach(s => {
      const opt = document.createElement("option");
      opt.value = s; opt.textContent = s || "—";
      sel.appendChild(opt);
    });
    if (state.channelSites[i] != null) {
      sel.value = state.channelSites[i];
      if (sel.value) tile.classList.add("assigned");
    }
    sel.onchange = () => {
      state.channelSites[i] = sel.value;
      saveSiteMap(state.channelSites);
      tile.classList.toggle("assigned", !!sel.value);
    };
    tile.appendChild(ch);
    tile.appendChild(sel);
    g.appendChild(tile);
  }
}

function collectMetadataForm() {
  const channel_sites = {};
  for (let i = 0; i < N_CHANNELS; i++) {
    const v = $("site-ch" + i)?.value || "";
    if (v) channel_sites[i] = v;
  }
  const fst = parseInt($("fst-select").value, 10);
  return {
    participant_id: $("pid-input").value.trim(),
    fitzpatrick: isNaN(fst) ? null : fst,
    notes: $("notes-input").value.trim(),
    channel_sites,
  };
}

function populateMetadataForm(p) {
  $("pid-input").value  = p?.participant_id || "";
  $("fst-select").value = p?.fitzpatrick != null ? String(p.fitzpatrick) : "";
  $("notes-input").value = p?.notes || "";
  for (let i = 0; i < N_CHANNELS; i++) {
    const el = $("site-ch" + i);
    const tile = $("site-tile-" + i);
    if (!el) continue;
    const v = p?.channel_sites?.[String(i)] || "";
    el.value = v;
    tile?.classList.toggle("assigned", !!v);
  }
}

async function saveMetadata() {
  const name = state.selectedSession;
  if (!name) return;
  const meta = collectMetadataForm();
  await postJSON(`/api/sessions/${name}/metadata`, meta);
  await refreshSessions();
  if (state.sessionDetail) {
    state.sessionDetail.participant = meta;
    renderMeta(state.sessionDetail);
    renderPPGBlock();   // refresh site labels on plot cards
  }
  flash("Metadata saved.", 1200);
}


/* ═══════════════════════════════════════════════════════════════════════════
   DELETE SESSION
   ═══════════════════════════════════════════════════════════════════════════ */

function openDeleteModal() {
  const name = state.selectedSession;
  if (!name) return;
  const pid = state.sessionDetail?.participant?.participant_id;
  $("delete-modal-text").innerHTML =
    `You are about to delete <b>${name}</b>${pid ? ` (participant <b>${pid}</b>)` : ""}.`;
  // Reset the confirm gesture each time the modal opens.
  $("delete-confirm-select").value = "";
  setEnabled("delete-confirm", false);
  openModal("delete-modal");
}

function closeDeleteModal() {
  closeModal("delete-modal");
}

async function performDelete() {
  const name = state.selectedSession;
  if (!name) { closeDeleteModal(); return; }
  if ($("delete-confirm-select").value !== "delete") return;  // belt-and-suspenders

  setEnabled("delete-confirm", false);
  try {
    const r = await fetch(`/api/sessions/${name}`, { method: "DELETE" });
    if (!r.ok) throw new Error((await r.text()) || r.statusText);
  } catch (e) {
    flash("Delete failed: " + e.message);
    return;
  } finally {
    closeDeleteModal();
  }

  // Drop the now-gone session and move on: select the next one, or fall
  // back to the empty state if that was the last session.
  state.selectedSession = null;
  state.sessionDetail = null;
  state.analysis = null;
  state.signals = null;
  await refreshSessions();
  if (state.sessions.length) {
    selectSession(state.sessions[0].name);
  } else {
    $("detail").classList.add("hidden");
    $("no-session").classList.remove("hidden");
    setChip("");
  }
  flash(`Deleted ${name}.`, 1500);
}

function loadSiteMap() {
  try { return JSON.parse(localStorage.getItem("seal_ppg_sites") || "{}"); }
  catch { return {}; }
}
function saveSiteMap(map) {
  try { localStorage.setItem("seal_ppg_sites", JSON.stringify(map)); }
  catch { /* private mode etc */ }
}


/* ═══════════════════════════════════════════════════════════════════════════
   LIVE COUNTERS / LOG
   ═══════════════════════════════════════════════════════════════════════════ */

function renderCounters(counts) {
  const el = $("counters");
  if (!counts || !Object.keys(counts).length) { el.innerHTML = ""; return; }
  const keys = Object.keys(counts).sort((a, b) => a === "ecg" ? -1 : b === "ecg" ? 1 : a.localeCompare(b));
  el.innerHTML = keys.map(k => `
    <div class="counter ${k === "ecg" ? "ecg" : ""}">
      <div class="k">${k}</div>
      <div class="v">${counts[k].toLocaleString()}</div>
    </div>`).join("");
}
function renderLog(lines) {
  const el = $("record-log");
  el.textContent = (lines || []).join("\n");
  el.scrollTop = el.scrollHeight;
}


/* ═══════════════════════════════════════════════════════════════════════════
   UTILITIES
   ═══════════════════════════════════════════════════════════════════════════ */

function setStatus(cls, label) {
  $("status").className = "status " + cls;
  $("status-label").textContent = label;
}
function setChip(text)   { $("session-chip").textContent = text || ""; }
function setEnabled(id, on) { const e = $(id); if (e) e.disabled = !on; }

// Tear down every Plotly chart inside a container before its innerHTML
// is replaced, so per-plot listeners/internals are released cleanly on
// each session switch instead of being orphaned.
function purgePlots(container) {
  if (!container) return;
  container.querySelectorAll(".js-plotly-plot").forEach(el => {
    try { Plotly.purge(el); } catch {}
  });
}

async function postJSON(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(await r.text() || r.statusText);
  return r.json();
}

function fmt(v, d) {
  if (v == null || (typeof v === "number" && !isFinite(v))) return "—";
  return Number(v).toFixed(d);
}
function fmtSigned(v, d) {
  if (v == null || !isFinite(v)) return "—";
  return (v >= 0 ? "+" : "") + Number(v).toFixed(d);
}
function fmtFs(hz) {
  if (!hz || !isFinite(hz)) return "?";
  return hz.toFixed(1) + " Hz";
}
function gradeCCC(c) {
  if (c == null || !isFinite(c)) return "";
  if (c > 0.95) return "grade-good";
  if (c > 0.90) return "grade-ok";
  return "grade-bad";
}
function chColor(ch) { return C.chPalette[ch % C.chPalette.length]; }

function hline(y, color, dash) {
  return {
    type: "line", xref: "paper", x0: 0, x1: 1, y0: y, y1: y,
    line: { color, width: 1.1, dash },
  };
}

function axisStyle(opts) {
  return {
    color: C.inkMute,
    gridcolor: C.hairSoft,
    zerolinecolor: C.hair,
    zeroline: false,
    linecolor: C.hair,
    tickcolor: C.hair,
    ticks: opts.showticks === false ? "" : "outside",
    showticklabels: opts.showticks !== false,
    title: opts.title ? { text: opts.title, font: { size: 12 }, standoff: 8 } : undefined,
    side: opts.side,
  };
}

// Linear interpolation: y-value at each query x within the downsampled signal.
// Used to anchor peak markers to the visible (downsampled) trace.
function interpY(xs, ys, queries) {
  const out = [];
  let j = 0;
  for (const q of queries) {
    while (j < xs.length - 1 && xs[j + 1] < q) j++;
    if (j >= xs.length - 1) { out.push(ys[ys.length - 1]); continue; }
    const x0 = xs[j], x1 = xs[j + 1];
    if (q < x0) { out.push(ys[j]); continue; }
    const t = (q - x0) / (x1 - x0);
    out.push(ys[j] + t * (ys[j + 1] - ys[j]));
  }
  return out;
}

function flash(msg, ttl = 2400) {
  let bar = document.getElementById("flash-bar");
  if (!bar) {
    bar = document.createElement("div");
    bar.id = "flash-bar";
    bar.style.cssText = `position:fixed; bottom:18px; left:50%; transform:translateX(-50%);
      background:#1c2330; border:1px solid #ff5e5e; color:#ece4d3;
      padding:12px 18px; border-radius:3px; font:14px system-ui, sans-serif;
      max-width:520px; z-index:1000; white-space:pre-wrap;`;
    document.body.appendChild(bar);
  }
  bar.textContent = msg;
  bar.style.opacity = "1";
  setTimeout(() => { bar.style.opacity = "0"; }, ttl);
}


/* ═══════════════════════════════════════════════════════════════════════════
   BATCH (every-session) ANALYSIS
   ═══════════════════════════════════════════════════════════════════════════ */

// Run analyze_all_sessions on the backend, swap the detail column into
// the batch view, render the per-site aggregate and per-session detail
// tables. The same crop window (Window controls in the detail header)
// is reused for every session — keeps numbers comparable to whatever
// single-session window the user was last looking at.
async function runBatchAnalysis(showBusy) {
  if (showBusy) {
    $("batch-status").textContent = "running…";
    setStatus("analyzing", "analyzing");
  }
  // Reveal the batch panel and tear down the per-session detail so the
  // two layouts don't fight for the main column.
  $("no-session").classList.add("hidden");
  $("detail").classList.add("hidden");
  $("batch").classList.remove("hidden");
  setChip("→ batch · MDPIdata");

  let payload;
  try {
    const r = await fetch(`/api/analyze_all?${windowQS().slice(1)}`, { method: "POST" });
    if (!r.ok) throw new Error(await r.text() || r.statusText);
    payload = await r.json();
  } catch (e) {
    $("batch-status").textContent = "";
    setStatus("idle", "idle");
    flash("Batch analysis failed: " + e.message);
    return;
  }

  setStatus("idle", "idle");
  $("batch-status").textContent = "";
  state.batch = payload;
  state.batchSource = "fresh";
  state.batchSiteSort = { col: null, dir: 1 };  // reset sort on a fresh load
  renderBatchSourceTag();
  renderBatchMeta(payload);
  renderBatchInterpretation(payload);
  renderBatchPerSite(payload);
  renderBatchPerChannel(payload);
  refreshBatchArchives();  // a fresh run added one to the archive
}

function closeBatchView() {
  $("batch").classList.add("hidden");
  state.batchSource = null;
  $("batch-source-tag").innerHTML = "";
  if (state.selectedSession) {
    $("detail").classList.remove("hidden");
    setChip("→ " + state.selectedSession);
  } else {
    $("no-session").classList.remove("hidden");
    setChip("");
  }
}

function renderBatchMeta(p) {
  const w = p.crop_window || {};
  const win = (w.start_s != null || w.end_s != null)
    ? `${w.start_s ?? "0"}–${w.end_s ?? "end"} s` : "full";
  $("batch-meta").innerHTML = `
    <div class="meta-cell"><div class="k">Sessions</div>
      <div class="v mono">${p.n_sessions_analyzed} / ${p.n_sessions_total}</div></div>
    <div class="meta-cell"><div class="k">Failed</div>
      <div class="v ${p.failed_sessions.length ? "ecg" : "mute"}">${p.failed_sessions.length || "—"}</div></div>
    <div class="meta-cell"><div class="k">Sites</div>
      <div class="v mono">${p.per_site.length}</div></div>
    <div class="meta-cell"><div class="k">FST strata</div>
      <div class="v ${p.fst_unavailable ? "mute" : "ecg"}">${p.fst_unavailable ? "unavailable" : "available"}</div></div>
    <div class="meta-cell"><div class="k">Crop window</div>
      <div class="v mono">${win}</div></div>
    <div class="meta-cell"><div class="k">Folder</div>
      <div class="v mono">MDPIdata/</div></div>
  `;
}

// Per-site aggregate table. Headers are sortable: clicking toggles
// ascending → descending → unsorted (back to backend order). The
// mean-of-{mean,std} cells extract `.mean` for the comparator.
const SITE_COLS = [
  { key: "site",              label: "Site",            getter: r => r.site,                        type: "str"  },
  { key: "n_channels",        label: "#ch",             getter: r => r.n_channels,                  type: "num"  },
  { key: "matched_beats_total", label: "Σ matched",     getter: r => r.matched_beats_total,         type: "num"  },
  { key: "ssqi",              label: "SSQI μ±σ",        getter: r => r.ssqi?.mean,                  type: "num"  },
  { key: "zsqi_mean",         label: "ZSQI μ μ±σ",      getter: r => r.zsqi_mean?.mean,             type: "num"  },
  { key: "ccc",               label: "CCC μ±σ",         getter: r => r.ccc?.mean,                   type: "num"  },
  { key: "icc",               label: "ICC μ±σ",         getter: r => r.icc?.mean,                   type: "num"  },
  { key: "pearson_r",         label: "Pearson μ±σ",     getter: r => r.pearson_r?.mean,             type: "num"  },
  { key: "bias_ms",           label: "Bias (ms) μ±σ",   getter: r => r.bias_ms?.mean,               type: "num"  },
  { key: "loa_span_ms",       label: "LOA span (ms) μ±σ", getter: r => r.loa_span_ms?.mean,         type: "num"  },
  { key: "rmse_ms",           label: "RMSE / MAE (ms)", getter: r => r.rmse_ms?.mean,               type: "num"  },
];

function renderBatchPerSite(p) {
  const tbl = $("batch-per-site-table");
  const rawSites = p.per_site || [];
  const { col, dir } = state.batchSiteSort;
  let sites = rawSites.slice();
  if (col != null && SITE_COLS[col]) {
    const g = SITE_COLS[col].getter;
    const isStr = SITE_COLS[col].type === "str";
    sites.sort((a, b) => {
      const va = g(a), vb = g(b);
      if (va == null && vb == null) return 0;
      if (va == null) return 1;       // null/NaN sink to the bottom regardless of dir
      if (vb == null) return -1;
      if (isStr) return String(va).localeCompare(String(vb)) * dir;
      return ((+va) - (+vb)) * dir;
    });
  }
  $("batch-per-site-summary").textContent =
    `${rawSites.length} body sites · grouped across ${p.n_sessions_analyzed} sessions`;

  const ths = SITE_COLS.map((c, i) => {
    const isSorted = state.batchSiteSort.col === i;
    const glyph = !isSorted ? "▴▾" : (state.batchSiteSort.dir > 0 ? "▴" : "▾");
    return `<th class="sortable${isSorted ? " sorted" : ""}" data-col="${i}">${c.label}<span class="sort-glyph">${glyph}</span></th>`;
  }).join("");
  let html = `<thead><tr>${ths}</tr></thead><tbody>`;

  if (!sites.length) {
    html += `<tr><td colspan="${SITE_COLS.length}" class="muted">no sessions analyzed</td></tr>`;
  }

  sites.forEach(row => {
    const cccCls = gradeCCC(row.ccc?.mean);
    const iccCls = gradeCCC(row.icc?.mean);
    html += `<tr>
      <td class="ch-name">${row.site}</td>
      <td>${row.n_channels}</td>
      <td>${(row.matched_beats_total || 0).toLocaleString()}</td>
      <td>${fmtMS(row.ssqi)}</td>
      <td>${fmtMS(row.zsqi_mean)}</td>
      <td class="${cccCls}">${fmtMS(row.ccc)}</td>
      <td class="${iccCls}">${fmtMS(row.icc)}</td>
      <td>${fmtMS(row.pearson_r)}</td>
      <td>${fmtMSsigned(row.bias_ms, 1)}</td>
      <td>${fmtMS(row.loa_span_ms, 0)}</td>
      <td>${fmtMS(row.rmse_ms, 1)} / ${fmtMS(row.mae_ms, 1)}</td>
    </tr>`;
  });
  tbl.innerHTML = html + "</tbody>";
  tbl.querySelectorAll("thead th.sortable").forEach(th => {
    th.tabIndex = 0;
    const act = () => sortBatchSite(parseInt(th.dataset.col, 10));
    th.addEventListener("click", act);
    th.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); act(); }});
  });
}

// Tri-state sort cycle: asc → desc → unsorted. Re-renders the table
// in place from the cached state.batch payload — no re-fetch.
function sortBatchSite(colIdx) {
  const cur = state.batchSiteSort;
  if (cur.col !== colIdx) state.batchSiteSort = { col: colIdx, dir: 1 };
  else if (cur.dir === 1) state.batchSiteSort = { col: colIdx, dir: -1 };
  else state.batchSiteSort = { col: null, dir: 1 };
  if (state.batch) renderBatchPerSite(state.batch);
}

function renderBatchPerChannel(p) {
  const tbl = $("batch-per-channel-table");
  const sess = p.sessions || [];
  const totalRows = sess.reduce((a, s) => a + (s.results || []).length, 0);
  $("batch-per-channel-summary").textContent =
    `${totalRows} channel rows · ${sess.length} sessions`;

  // Tighter column set — session identity is in the group-header row
  // above each block, so the channel rows don't carry PID/FST/Session.
  // RMSE/MAE share a cell so the table fits without horizontal scroll.
  let html = `<thead><tr>
    <th>Ch</th><th>Site</th>
    <th>fs (Hz)</th><th>SSQI</th>
    <th>ZSQI μ</th><th>Matched</th>
    <th>CCC</th><th>ICC</th><th>Pearson</th>
    <th>Bias (ms)</th><th>LOA± (ms)</th>
    <th>RMSE / MAE</th>
  </tr></thead><tbody>`;
  const COL_SPAN = 12;

  if (!sess.length) {
    html += `<tr><td colspan="${COL_SPAN}" class="muted">no sessions analyzed</td></tr>`;
  }

  sess.forEach((sx, sessIdx) => {
    const pid = sx.participant?.participant_id || "";
    const fst = sx.participant?.fitzpatrick;
    const fstStr = fst ? ["I","II","III","IV","V","VI"][fst - 1] : "";
    const sessName = sx.session_name;
    const ecg = sx.ecg || {};
    const niceTs = (sx.started_at || sessName).replace("T", " ").slice(0, 19);
    const meta = [
      `<span class="sess-id-full mono">${sessName}</span>`,
      `<span class="sess-ts">${niceTs}</span>`,
      pid ? `<span class="sess-tag">PID <b>${pid}</b></span>` : "",
      fstStr ? `<span class="sess-tag fst">FST <b>${fstStr}</b></span>` : "",
      ecg.mean_hr_bpm ? `<span class="sess-tag">HR <b>${ecg.mean_hr_bpm.toFixed(0)} bpm</b></span>` : "",
      ecg.duration_s ? `<span class="sess-tag">duration <b>${ecg.duration_s.toFixed(1)} s</b></span>` : "",
      `<span class="sess-tag">${(sx.results || []).length} channels</span>`,
    ].filter(Boolean).join("");

    // Session group header — full session_YYYYMMDD_HHMMSS id is the
    // headline, with PID/FST/HR/duration as inline tags. Clicking the
    // ID drills into the per-session detail view.
    html += `<tr class="sess-group" data-sess-idx="${sessIdx + 1}">
      <td colspan="${COL_SPAN}" class="sess-group-cell">
        <span class="sess-counter mono">[ ${sessIdx + 1} / ${sess.length} ]</span>
        <a href="#" data-sess="${sessName}" class="sess-link">${meta}</a>
      </td>
    </tr>`;

    (sx.results || []).forEach(row => {
      const s = row.stats;
      const cccCls = s ? gradeCCC(s.ccc) : "";
      const iccCls = s ? gradeCCC(s.icc) : "";
      html += `<tr class="sess-row">
        <td class="ch-name">ch${row.channel}</td>
        <td class="site">${row.site || "—"}</td>
        <td>${fmt(row.ppg_fs_hz, 1)}</td>
        <td>${fmt(row.ssqi, 3)}</td>
        <td>${fmt(row.zsqi_mean, 3)}</td>
        <td>${row.n_matched_beats}</td>
        <td class="${cccCls}">${s ? fmt(s.ccc, 3) : "—"}</td>
        <td class="${iccCls}">${s ? fmt(s.icc, 3) : "—"}</td>
        <td>${s ? fmt(s.pearson_r, 3) : "—"}</td>
        <td>${s ? fmtSigned(s.bias_ms, 1) : "—"}</td>
        <td>${s ? fmtSigned(s.loa_lower_ms, 0) + " / " + fmtSigned(s.loa_upper_ms, 0) : "—"}</td>
        <td>${s ? fmt(s.rmse_ms, 1) + " / " + fmt(s.mae_ms, 1) : "—"}</td>
      </tr>`;
    });
  });

  tbl.innerHTML = html + "</tbody>";
  tbl.querySelectorAll(".sess-link").forEach(a => {
    a.addEventListener("click", (e) => {
      e.preventDefault();
      selectSession(a.dataset.sess);
    });
  });
}

// "mean ± std" formatter for the per-site row cells. ``n`` controls
// decimal places (default 3 — works for SSQI/CCC/ICC). NaN reduces to "—".
function fmtMS(stat, n = 3) {
  if (!stat || !isFinite(stat.mean)) return "—";
  return `${stat.mean.toFixed(n)} ± ${(stat.std ?? 0).toFixed(n)}`;
}
function fmtMSsigned(stat, n = 1) {
  if (!stat || !isFinite(stat.mean)) return "—";
  const sgn = stat.mean >= 0 ? "+" : "";
  return `${sgn}${stat.mean.toFixed(n)} ± ${(stat.std ?? 0).toFixed(n)}`;
}


/* ═══════════════════════════════════════════════════════════════════════════
   INTERPRETATION RENDERING — session & batch plain-English summaries
   ═══════════════════════════════════════════════════════════════════════════ */

// Render the per-session interpretation block above the SQI table.
// Reads state.analysis.interpretation (backend-emitted). Silent no-op if
// the analysis errored or the field is absent (older cached payloads).
function renderSessionInterpretation() {
  const block = $("session-interp-block");
  const host = $("session-interp");
  const ana = state.analysis;
  const interp = ana?.interpretation;
  if (!interp || !interp.headline) {
    block.classList.add("hidden");
    host.innerHTML = "";
    return;
  }
  block.classList.remove("hidden");

  const cards = (interp.channel_summaries || []).map(c => verdictCardHTML(c)).join("");
  const notes = (interp.notes || []);

  host.innerHTML = `
    <h2 class="interp-headline">${escHTML(interp.headline)}</h2>
    ${interp.ecg_text ? `<p class="interp-sub">${escHTML(interp.ecg_text)}</p>` : ""}
    ${cards ? `<div class="verdict-grid">${cards}</div>` : ""}
    ${notes.length ? notesBlockHTML(notes) : ""}
  `;
}

// Render the batch-level interpretation block above the per-site table.
function renderBatchInterpretation(p) {
  const block = $("batch-interp-block");
  const host = $("batch-interp");
  const interp = p?.interpretation;
  if (!interp || !interp.headline) {
    block.classList.add("hidden");
    host.innerHTML = "";
    return;
  }
  block.classList.remove("hidden");

  const rows = (interp.site_summaries || []).map(s => {
    const cls = "vc-" + (s.grade || "warn");
    return `
      <div class="site-verdict ${cls}">
        <span class="sv-site">${escHTML(s.site || "—")}</span>
        <span class="sv-pill">${escHTML((s.grade || "—").toUpperCase())}</span>
        <span class="sv-text">${escHTML(s.text || "")}</span>
      </div>`;
  }).join("");

  host.innerHTML = `
    <h2 class="interp-headline">${escHTML(interp.headline)}</h2>
    ${rows ? `<div class="site-verdict-list">${rows}</div>` : ""}
    ${(interp.notes || []).length ? notesBlockHTML(interp.notes) : ""}
  `;
}

// One per-channel verdict card. The grade controls the left border and
// pill colour. Lines are rendered as bullets; advice is a callout box.
function verdictCardHTML(c) {
  const cls = "vc-" + (c.grade || "warn");
  const lines = (c.lines || []).map(l => `<li>${escHTML(l)}</li>`).join("");
  const advice = c.advice ? `<div class="vc-advice">${escHTML(c.advice)}</div>` : "";
  return `
    <div class="verdict-card ${cls}">
      <div class="vc-head">
        <span class="vc-grade-pill">${escHTML((c.grade || "—").toUpperCase())}</span>
        <span>${escHTML(c.verdict || "")}</span>
      </div>
      ${lines ? `<ul class="vc-lines">${lines}</ul>` : ""}
      ${advice}
    </div>`;
}

function notesBlockHTML(notes) {
  const items = notes.map(n => `<li>${escHTML(n)}</li>`).join("");
  return `
    <div class="interp-notes">
      <span class="nlabel">Notes</span>
      <ul>${items}</ul>
    </div>`;
}

// Minimal HTML escaper — backend text may contain <, &, etc. and we
// inject via innerHTML for layout flexibility, so explicit escape.
function escHTML(s) {
  if (s == null) return "";
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}


/* ═══════════════════════════════════════════════════════════════════════════
   RUN HISTORY — collapsible timeline above the run-history block
   ═══════════════════════════════════════════════════════════════════════════ */

async function toggleHistory() {
  const open = !state.historyOpen;
  state.historyOpen = open;
  $("btn-history-toggle").setAttribute("aria-expanded", open ? "true" : "false");
  if (!open) {
    $("history-list").classList.add("hidden");
    const n = state.sessionDetail?.history_count || 0;
    $("btn-history-toggle").textContent = n ? `▸ Show history (${n})` : "▸ Show history";
    return;
  }
  // Open: fetch (or refresh) and reveal.
  await loadHistory(state.selectedSession, false);
  $("history-list").classList.remove("hidden");
  const n = (state.history || []).length;
  $("btn-history-toggle").textContent = `▾ Hide history (${n})`;
}

async function loadHistory(name, forceVisible) {
  if (!name) return;
  const list = $("history-list");
  list.innerHTML = `<li class="history-empty">loading…</li>`;
  if (forceVisible) {
    list.classList.remove("hidden");
    state.historyOpen = true;
    $("btn-history-toggle").setAttribute("aria-expanded", "true");
  }
  try {
    const events = await fetch(`/api/sessions/${name}/history?limit=500`)
      .then(r => r.json());
    state.history = Array.isArray(events) ? events : [];
  } catch (e) {
    list.innerHTML = `<li class="history-empty">failed to load history: ${escHTML(e.message)}</li>`;
    return;
  }
  renderHistoryList();
  if (forceVisible) {
    const n = state.history.length;
    $("btn-history-toggle").textContent = `▾ Hide history (${n})`;
  }
}

function renderHistoryList() {
  const list = $("history-list");
  const events = state.history || [];
  if (!events.length) {
    list.innerHTML = `<li class="history-empty">no events recorded for this session yet</li>`;
    return;
  }
  list.innerHTML = events.map(ev => {
    const ts = (ev.ts || "").replace("T", " ").slice(0, 19);
    const pill = `<span class="hpill evt-${escHTML(ev.event || "unknown")}">${escHTML(ev.event || "?")}</span>`;
    return `<li>
      <span class="hts">${escHTML(ts)}</span>
      ${pill}
      <span class="hsum">${escHTML(summarizeEvent(ev))}</span>
    </li>`;
  }).join("");
}

// Best-effort one-line summary for each event-type. Falls back to a
// compact JSON snippet so unknown event types still show something useful.
function summarizeEvent(ev) {
  const d = ev.data || {};
  switch (ev.event) {
    case "recording_started": {
      const port = d.port ? ` on ${d.port}` : "";
      return `recording started${port}`;
    }
    case "recording_stopped": {
      const code = d.exit_code != null ? `exit ${d.exit_code}` : "stopped";
      return `recording ${code}`;
    }
    case "metadata_edited": {
      const after = d.after || {};
      const pid = after.participant_id ? `PID ${after.participant_id}` : "no PID";
      const fst = after.fitzpatrick ? `FST ${["I","II","III","IV","V","VI"][after.fitzpatrick - 1]}` : "";
      const sites = after.channel_sites ? Object.keys(after.channel_sites).length + " site mappings" : "";
      return [pid, fst, sites].filter(Boolean).join(", ");
    }
    case "analysis_run": {
      const n = d.n_channels || 0;
      const sum = d.summary || {};
      const ccs = Object.entries(sum).map(([ch, s]) => [ch, s?.ccc]).filter(([, c]) => c != null);
      if (!ccs.length) return `${n} channels analysed`;
      const best = ccs.reduce((a, b) => (b[1] > a[1] ? b : a));
      const crop = d.crop_window || {};
      const win = (crop.start_s != null || crop.end_s != null)
        ? ` (window ${crop.start_s ?? "0"}–${crop.end_s ?? "end"} s)` : "";
      return `${n} channels, best CCC ${(+best[1]).toFixed(3)} on ch${best[0]}${win}`;
    }
    case "batch_analysis_included": {
      const id = d.batch_id || "?";
      const pos = (d.session_position && d.total) ? `position ${d.session_position}/${d.total}` : "";
      return `included in batch ${id}${pos ? " · " + pos : ""}`;
    }
    default:
      try { return JSON.stringify(d).slice(0, 140); } catch { return ""; }
  }
}


/* ═══════════════════════════════════════════════════════════════════════════
   RECEIVER LOG — modal showing the captured subprocess stdout/stderr
   ═══════════════════════════════════════════════════════════════════════════ */

async function openReceiverLog() {
  const name = state.selectedSession;
  if (!name) return;
  if (!state.sessionDetail?.has_receiver_log) {
    flash("No receiver log on disk for this session.");
    return;
  }
  $("recv-log-meta").innerHTML = `Showing the last 500 lines from <b>${escHTML(name)}</b>/receiver.log.`;
  $("recv-log-pre").textContent = "loading…";
  openModal("recv-log-modal");
  await refreshReceiverLog();
}

async function refreshReceiverLog() {
  const name = state.selectedSession;
  if (!name) return;
  const pre = $("recv-log-pre");
  pre.classList.remove("empty");
  pre.textContent = "loading…";
  try {
    const r = await fetch(`/api/sessions/${name}/receiver_log?tail=500`).then(r => r.json());
    const log = (r && r.log) || "";
    if (!log.trim()) {
      pre.textContent = "(receiver.log is empty)";
      pre.classList.add("empty");
    } else {
      pre.textContent = log;
      pre.scrollTop = pre.scrollHeight;
    }
  } catch (e) {
    pre.textContent = "failed to load: " + e.message;
    pre.classList.add("empty");
  }
}


/* ═══════════════════════════════════════════════════════════════════════════
   BATCH ARCHIVE — sidebar count chip + browse modal + reload past runs
   ═══════════════════════════════════════════════════════════════════════════ */

async function refreshBatchArchives() {
  try {
    state.batchArchives = await fetch("/api/batch_analyses").then(r => r.json());
  } catch {
    state.batchArchives = [];
  }
  const n = state.batchArchives.length;
  $("batch-archive-count").textContent = n;
  setEnabled("btn-browse-archive", n > 0);
}

function openArchiveModal() {
  if (!state.batchArchives.length) return;
  const ul = $("archive-list");
  ul.innerHTML = state.batchArchives.map(b => {
    const ts = (b.created_at || "").replace("T", " ").slice(0, 19);
    const w = b.crop_window || {};
    const win = (w.start_s != null || w.end_s != null)
      ? `${w.start_s ?? "0"}–${w.end_s ?? "end"} s` : "full window";
    return `<li tabindex="0" data-id="${escHTML(b.batch_id)}">
      <span class="arc-id">${escHTML(b.batch_id)}</span>
      <span class="arc-meta">${escHTML(ts)} · ${escHTML(win)}</span>
      <span class="arc-n">${b.n_sessions_analyzed ?? 0} sess</span>
    </li>`;
  }).join("") || `<li class="archive-empty">no saved batch runs yet</li>`;
  ul.querySelectorAll("li[data-id]").forEach(li => {
    const id = li.dataset.id;
    const act = () => loadBatchFromArchive(id);
    li.addEventListener("click", act);
    li.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); act(); }});
  });
  openModal("batch-archive-modal");
}

async function loadBatchFromArchive(batchId) {
  closeModal("batch-archive-modal");
  setStatus("analyzing", "analyzing");
  $("batch-status").textContent = "loading archive…";
  // Mirror runBatchAnalysis's view-swap so the user sees the same screen.
  $("no-session").classList.add("hidden");
  $("detail").classList.add("hidden");
  $("batch").classList.remove("hidden");
  setChip("→ batch · " + batchId);

  let payload;
  try {
    payload = await fetch(`/api/batch_analyses/${batchId}`).then(r => {
      if (!r.ok) throw new Error(r.statusText);
      return r.json();
    });
  } catch (e) {
    setStatus("idle", "idle");
    $("batch-status").textContent = "";
    flash("Failed to load batch " + batchId + ": " + e.message);
    return;
  }
  setStatus("idle", "idle");
  $("batch-status").textContent = "";
  state.batch = payload;
  state.batchSource = { batch_id: batchId, created_at: payload.created_at };
  state.batchSiteSort = { col: null, dir: 1 };
  renderBatchSourceTag();
  renderBatchMeta(payload);
  renderBatchInterpretation(payload);
  renderBatchPerSite(payload);
  renderBatchPerChannel(payload);
}

function renderBatchSourceTag() {
  const tag = $("batch-source-tag");
  if (!tag) return;
  if (!state.batchSource) { tag.innerHTML = ""; return; }
  if (state.batchSource === "fresh") {
    tag.innerHTML = `<span class="batch-archive-tag fresh">● live run</span>`;
  } else {
    const id = state.batchSource.batch_id;
    const ts = (state.batchSource.created_at || "").replace("T", " ").slice(0, 19);
    tag.innerHTML = `<span class="batch-archive-tag archive" title="${escHTML(ts)}">archive · ${escHTML(id)}</span>`;
  }
}

// Per-site aggregate → CSV. Emits the same column order as the table,
// flattens {mean, std} into two columns each. UTF-8 BOM for Excel.
function exportBatchCSV() {
  const p = state.batch;
  if (!p || !p.per_site) { flash("Nothing to export — run a batch first."); return; }
  const header = ["site", "n_channels", "matched_beats_total"];
  const metricCols = ["ssqi", "zsqi_mean", "ccc", "icc", "pearson_r",
                      "bias_ms", "loa_span_ms", "rmse_ms", "mae_ms"];
  metricCols.forEach(c => { header.push(c + "_mean", c + "_std"); });
  const lines = [header.join(",")];

  (p.per_site || []).forEach(row => {
    const cells = [csvCell(row.site), row.n_channels, row.matched_beats_total];
    metricCols.forEach(c => {
      const s = row[c] || {};
      cells.push(isFinite(s.mean) ? s.mean : "", isFinite(s.std) ? s.std : "");
    });
    lines.push(cells.join(","));
  });

  const tag = (state.batchSource && state.batchSource !== "fresh")
    ? state.batchSource.batch_id
    : ("batch_" + new Date().toISOString().replace(/[-:T.Z]/g, "").slice(0, 14));
  const blob = new Blob(["﻿" + lines.join("\n") + "\n"],
                       { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = `per_site_${tag}.csv`;
  document.body.appendChild(a); a.click();
  setTimeout(() => { document.body.removeChild(a); URL.revokeObjectURL(url); }, 0);
}
function csvCell(v) {
  if (v == null) return "";
  const s = String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}


/* ═══════════════════════════════════════════════════════════════════════════
   ONBOARDING — first-launch multi-step guide, also accessible from the
   header "?" button. Six panels, dot navigation, "don't show again" flag.
   ═══════════════════════════════════════════════════════════════════════════ */

const ONB_STEPS = [
  {
    title: "Welcome to SEAL · PPG",
    body: `
      <p>This dashboard captures multi-site photoplethysmography (PPG) and an ECG reference, then scores agreement between them per channel, per body site, and across the whole cohort.</p>
      <p>The layout has three columns:</p>
      <ul>
        <li><strong>Left — Record.</strong> Pick a serial port, fill participant metadata, start/stop a capture.</li>
        <li><strong>Middle — Sessions.</strong> Every recording in <code>MDPIdata/</code>, filterable, with persistence badges.</li>
        <li><strong>Right — Detail / Batch.</strong> ECG hero, per-channel PPG, plain-English verdicts, SQI &amp; agreement table, Bland-Altman plots — or the cohort-wide batch view.</li>
      </ul>
    `,
  },
  {
    title: "Recording a session",
    body: `
      <p>From the <strong>Record</strong> column:</p>
      <ul>
        <li>Pick the serial port (use <code>⟳</code> to rescan) and the baud (defaults to 115200).</li>
        <li>Enter Participant ID, Fitzpatrick skin type, and notes — these write into <code>participant.json</code>.</li>
        <li>Assign each channel to a body site (finger, forehead, earlobe, …). The map persists locally so you don't re-enter it next session.</li>
        <li>Press <strong>● Start</strong>. A new <code>session_YYYYMMDD_HHMMSS/</code> folder appears under <code>MDPIdata/</code>; counters and the live log update once per second.</li>
        <li>Press <strong>■ Stop</strong> when done — the receiver subprocess flushes its log and the dashboard auto-runs analysis.</li>
      </ul>
    `,
  },
  {
    title: "One session vs the whole batch",
    body: `
      <p>There are two analysis modes:</p>
      <ul>
        <li><strong>Per-session.</strong> Click a session in the middle column, then <strong>Run analysis</strong> in the detail header (or wait for the auto-run on recording stop). Optional <strong>Window</strong> inputs crop the data to a specific second range.</li>
        <li><strong>Batch (cohort).</strong> Use <strong>▦ Analyze all sessions</strong> in the Sessions sidebar. Every session in <code>MDPIdata/</code> gets re-scored with the same crop window, then aggregated per body site.</li>
      </ul>
      <p>Past batch runs are auto-archived. The <strong>Browse archive</strong> button reloads any saved run without re-computing it.</p>
    `,
  },
  {
    title: "Where files land on disk",
    body: `
      <p>Everything written by the dashboard lives under <code>MDPIdata/</code>:</p>
      <ul class="onb-file-list">
        <li class="dir">session_YYYYMMDD_HHMMSS/</li><li> </li>
        <li>ecg_data.csv</li>
        <li>ppg_data_ch{0..N}.csv</li>
        <li>participant.json</li>
        <li>analysis.json</li>
        <li>history.jsonl</li>
        <li>receiver.log</li>
        <li class="dir">batch_analyses/</li>
        <li>batch_YYYYMMDD_HHMMSS.json</li>
      </ul>
      <p style="margin-top:14px;">Deleting a session from the dashboard removes the entire <code>session_*/</code> folder. Batch archives never auto-prune.</p>
    `,
  },
  {
    title: "Reading the interpretation grades",
    body: `
      <p>After every analysis run, each channel gets a plain-English verdict card. The grade colour summarises agreement with the ECG reference:</p>
      <div class="onb-grid">
        <div class="onb-grade good"><span class="pill">GOOD</span><span class="desc">CCC &gt; 0.95 — ECG-grade signal.</span></div>
        <div class="onb-grade ok"><span class="pill">OK</span><span class="desc">CCC &gt; 0.90 — usable with caveats.</span></div>
        <div class="onb-grade warn"><span class="pill">WARN</span><span class="desc">CCC &gt; 0.50 — inspect before using.</span></div>
        <div class="onb-grade bad"><span class="pill">BAD</span><span class="desc">No agreement — re-check sensor.</span></div>
      </div>
      <p style="margin-top:14px;">The batch view applies the same grades at the per-site level, so you can see which body sites are reliable across the cohort.</p>
    `,
  },
  {
    title: "SQI · CCC · ICC — what the metrics mean",
    body: `
      <p>The SQI &amp; agreement table breaks down per-channel quality:</p>
      <ul>
        <li><strong>SSQI</strong> — skewness of the PPG amplitude; positive values mean systolic peaks dominate (good morphology).</li>
        <li><strong>ZSQI</strong> — beat-by-beat z-score statistic; tight μ±σ implies a stable waveform.</li>
        <li><strong>CCC / ICC</strong> — Lin's Concordance Correlation and Intra-class Correlation between matched PPI (PPG) and RR (ECG) intervals. Above 0.95 is substantial agreement; above 0.90 is moderate; below 0.50 is failing.</li>
        <li><strong>Bias / LOA / RMSE</strong> — Bland-Altman pieces: mean signed offset, ±1.96σ limits-of-agreement, and root-mean-square error per beat.</li>
      </ul>
      <p>The Bland-Altman plot at the bottom of each session visualises bias and LOA directly. You can re-open this guide any time via the <strong>?</strong> button in the top bar.</p>
    `,
  },
];
let onbStep = 0;

function openOnboarding(reset) {
  onbStep = 0;
  $("onb-skip-cb").checked = !!localStorage.getItem("seal_ppg_onboarded");
  renderOnboardingStep();
  buildOnbDots();
  openModal("onb-modal");
}
function buildOnbDots() {
  const host = $("onb-dots");
  // Numbered step indicators: each button shows its 1-based index. Past
  // steps get .completed, current step gets .active. Much easier to read
  // than tiny anonymous dots, and you can jump straight to any step.
  host.innerHTML = ONB_STEPS.map((_, i) => {
    const cls = i === onbStep ? "active" : (i < onbStep ? "completed" : "");
    return `<button class="onb-dot ${cls}" data-i="${i}" aria-label="Step ${i+1}"${i === onbStep ? ' aria-current="step"' : ''}>${i + 1}</button>`;
  }).join("");
  host.querySelectorAll(".onb-dot").forEach(d => {
    d.addEventListener("click", () => { onbStep = parseInt(d.dataset.i, 10); renderOnboardingStep(); buildOnbDots(); });
  });
}
function renderOnboardingStep() {
  const s = ONB_STEPS[onbStep];
  $("onb-step").textContent = `Step ${onbStep + 1} of ${ONB_STEPS.length}`;
  $("onb-title").textContent = s.title;
  $("onb-body").innerHTML = s.body;
  $("onb-back").disabled = onbStep === 0;
  $("onb-next").textContent = onbStep === ONB_STEPS.length - 1 ? "Done" : "Next";
}
function onbNext() {
  if (onbStep === ONB_STEPS.length - 1) {
    // Always set the flag on Done, regardless of checkbox state — Done = "I've finished it".
    try { localStorage.setItem("seal_ppg_onboarded", "1"); } catch {}
    closeModal("onb-modal");
    return;
  }
  onbStep++;
  renderOnboardingStep();
  buildOnbDots();
}
function onbBack() {
  if (onbStep > 0) {
    onbStep--;
    renderOnboardingStep();
    buildOnbDots();
  }
}
