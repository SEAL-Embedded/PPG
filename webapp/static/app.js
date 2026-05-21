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
  pollTimer: null,
  channelSites: loadSiteMap(),
  bpChannels: new Set(),     // PPG channels currently showing the 0.6-3.3 Hz bandpass
  window: { start: null, end: null },  // pre-processing crop, seconds since session t0
};

// ── DOM helpers ──────────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);

window.addEventListener("DOMContentLoaded", () => {
  buildSitesGrid();
  refreshPorts();
  refreshSessions().then(() => {
    if (state.sessions.length) selectSession(state.sessions[0].name);
  });
  pollStatus();
  restoreSidebarState();

  $("port-refresh").onclick = refreshPorts;
  $("btn-start").onclick    = startRecording;
  $("btn-stop").onclick     = stopRecording;
  $("btn-analyze").onclick  = () => runAnalysis(true);
  $("btn-reload-signals").onclick = () => loadSignals(state.selectedSession);
  $("btn-save-meta").onclick = saveMetadata;
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
  $("delete-modal").addEventListener("click", (e) => {
    if (e.target.id === "delete-modal") closeDeleteModal();  // click backdrop to cancel
  });
  $("session-filter").oninput = renderSessionList;
  document.querySelectorAll(".col-toggle").forEach(btn => {
    btn.addEventListener("click", () => toggleSidebar(parseInt(btn.dataset.col, 10)));
  });
});

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
    li.innerHTML = `
      <div class="row1">
        <span class="pid${pid ? "" : " empty"}">${pid || "unassigned"}</span>
        ${fst ? `<span class="fst">FST ${"I II III IV V VI".split(" ")[fst-1]}</span>` : ""}
      </div>
      <span class="ts">${ts}</span>
      <div class="channels">${channelTags.length ? channelTags.join("") : '<span class="none">no signals</span>'}</div>`;
    li.onclick = () => selectSession(s.name);
    ul.appendChild(li);
  });
}


/* ═══════════════════════════════════════════════════════════════════════════
   SESSION DETAIL
   ═══════════════════════════════════════════════════════════════════════════ */

async function selectSession(name, opts = {}) {
  state.selectedSession = name;
  state.analysis = null;
  state.signals = null;
  renderSessionList();
  $("no-session").classList.add("hidden");
  $("detail").classList.remove("hidden");
  $("detail-title").innerHTML = formatSessionTitle(name);
  setChip("→ " + name);
  setEnabled("btn-analyze", true);
  setEnabled("btn-reload-signals", true);
  setEnabled("btn-save-meta", true);
  syncWindowInputs();   // window persists across sessions; show it

  try {
    const s = await fetch(`/api/sessions/${name}`).then(r => r.json());
    state.sessionDetail = s;
    renderMeta(s);
    populateMetadataForm(s.participant);
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

  let html = `<thead><tr>
    <th>Channel</th><th>Site</th>
    <th>fs (Hz)</th><th>SSQI</th>
    <th>ZSQI μ</th><th>ZSQI σ</th>
    <th>Matched</th>
    <th>CCC</th><th>Pearson</th>
    <th>Bias (ms)</th><th>LOA± (ms)</th>
    <th>RMSE</th><th>MAE</th>
  </tr></thead><tbody>`;

  // ECG reference row at top — it's the agreement denominator
  if (r.ecg) {
    const e = r.ecg;
    html += `<tr>
      <td class="ch-name ecg">ECG ref</td>
      <td class="site">Einthoven</td>
      <td>${fmt(e.fs_hz, 1)}</td>
      <td colspan="3" class="muted">${e.n_peaks} R-peaks · HR ${fmt(e.mean_hr_bpm, 0)} bpm · ${e.leads_off_samples} leads-off</td>
      <td colspan="7" class="muted">—</td>
    </tr>`;
  }

  (r.results || []).forEach(row => {
    const s = row.stats;
    const cccCls = s ? gradeCCC(s.ccc) : "";
    html += `<tr>
      <td class="ch-name">ch${row.channel}</td>
      <td class="site">${row.site || "—"}</td>
      <td>${fmt(row.ppg_fs_hz, 1)}</td>
      <td>${fmt(row.ssqi, 3)}</td>
      <td>${fmt(row.zsqi_mean, 3)}</td>
      <td>${fmt(row.zsqi_std, 3)}</td>
      <td>${row.n_matched_beats}</td>
      <td class="${cccCls}">${s ? fmt(s.ccc, 4) : "—"}</td>
      <td>${s ? fmt(s.pearson_r, 4) : "—"}</td>
      <td>${s ? fmtSigned(s.bias_ms, 1) : "—"}</td>
      <td>${s ? fmtSigned(s.loa_lower_ms, 0) + " / " + fmtSigned(s.loa_upper_ms, 0) : "—"}</td>
      <td>${s ? fmt(s.rmse_ms, 1) : "—"}</td>
      <td>${s ? fmt(s.mae_ms, 1) : "—"}</td>
    </tr>`;
    if (row.error) {
      html += `<tr class="err-row"><td colspan="13">⚠ ${row.error}</td></tr>`;
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
  $("delete-modal").classList.remove("hidden");
}

function closeDeleteModal() {
  $("delete-modal").classList.add("hidden");
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
