#!/usr/bin/env python
"""Build the redesigned standalone ScaleComm Explorer UI.

This version is intentionally map-first. The paper-facing default follows the
CellNEST-style interaction pattern: a clear tissue-surface communication map,
LR/cell-type filters, top-signal range control, and click-to-inspect event
details. The goal is a tool screenshot that looks like a usable product rather
than a decorative 3D demo.
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

from plotly.offline import get_plotlyjs

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.current.build_spatialself_explorer_static import build_payload


OUT_DIR = ROOT / "outputs/current_web_ui"


def build_html() -> str:
    payload = build_payload()
    payload_json = json.dumps(payload, separators=(",", ":"))
    plotly_js = get_plotlyjs()

    css = """
    <style>
    :root {
      --ink: #172126;
      --muted: #65747B;
      --line: #D9E2E0;
      --soft: #F6F8F6;
      --paper: #FBFCFA;
      --sender: #237A78;
      --receiver: #B85450;
      --edge: #1F2A30;
      --accent: #6C9E68;
      --amber: #D9992B;
    }
    * { box-sizing: border-box; }
    html, body {
      width: 100vw; height: 100vh; margin: 0; overflow: hidden;
      background: var(--paper); color: var(--ink);
      font-family: Arial, Helvetica, sans-serif;
      -webkit-font-smoothing: antialiased;
    }
    .page {
      width: 100vw; height: 100vh; padding: 14px 18px 16px;
      display: grid; grid-template-rows: 44px minmax(0, 1fr); gap: 10px;
      background: var(--paper);
    }
    .topbar {
      display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 16px;
      align-items: center; border: 1px solid rgba(217,226,224,.95);
      padding: 8px 12px; background: rgba(255,255,255,.92);
      border-radius: 8px;
      box-shadow: 0 10px 28px rgba(23,33,38,.045);
    }
    .brand { display: flex; align-items: baseline; gap: 13px; min-width: 0; }
    .brand h1 {
      margin: 0; font-size: 20px; line-height: 1; letter-spacing: 0;
      font-weight: 700; color: var(--ink);
    }
    .brand span {
      color: var(--muted); font-size: 12px; white-space: nowrap;
      overflow: hidden; text-overflow: ellipsis;
    }
    .top-summary {
      display: flex; align-items: center; justify-content: flex-end; gap: 8px;
      color: #405057; font-size: 11px; white-space: nowrap;
    }
    .top-summary b { color: var(--sender); font-weight: 700; }
    .top-summary .sep {
      width: 4px; height: 4px; border-radius: 50%; background: #B9C1D1; display: inline-block;
    }
    .shell {
      min-height: 0; display: grid; grid-template-columns: minmax(0, 1fr) 326px;
      gap: 10px;
    }
    .canvas {
      position: relative; min-width: 0; min-height: 0; border: 1px solid var(--line);
      background: #fff; overflow: hidden; border-radius: 8px;
      box-shadow: 0 12px 32px rgba(23,33,38,.055), inset 0 0 0 1px rgba(255,255,255,.80);
    }
    #plot { width: 100%; height: 100%; }
    .canvas-title {
      position: absolute; left: 16px; top: 14px; z-index: 5;
      pointer-events: none;
    }
    .canvas-title b {
      display: block; color: var(--ink); font-size: 14px; line-height: 1.05;
      letter-spacing: 0;
    }
    .canvas-title span {
      display: block; color: var(--muted); font-size: 11px; margin-top: 4px;
    }
    .toolbar {
      position: absolute; right: 14px; top: 14px; z-index: 7;
      display: flex; align-items: center; gap: 6px;
    }
    button, .button {
      border: 1px solid rgba(214, 224, 228, .95); background: rgba(255,255,255,.88);
      color: #334047; font: 400 11px/1 Arial, Helvetica, sans-serif;
      padding: 7px 9px; cursor: pointer; border-radius: 6px;
    }
    button:hover { background: #fff; color: var(--ink); border-color: #C8D3D8; }
    .mode.active { color: #fff; background: #26343A; border-color: #26343A; }
    .view-toggle {
      display: inline-flex; gap: 0; border: 1px solid rgba(214, 224, 228, .95);
      background: rgba(255,255,255,.88); border-radius: 6px; overflow: hidden;
    }
    .view-toggle button {
      border: 0; border-right: 1px solid rgba(214, 224, 228, .95);
      background: transparent; padding: 7px 9px;
    }
    .view-toggle button:last-child { border-right: 0; }
    .view-toggle button.active { background: #26343A; color: #fff; }
    .plot-legend {
      position: absolute; left: 16px; bottom: 14px; z-index: 6;
      display: flex; gap: 10px; align-items: center; flex-wrap: wrap;
      padding: 7px 9px; border: 1px solid rgba(221,229,232,.88);
      background: rgba(255,255,255,.82); color: #506068; font-size: 10.5px;
      border-radius: 7px;
      backdrop-filter: blur(3px);
    }
    .plot-legend span { display: inline-flex; align-items: center; gap: 5px; white-space: nowrap; }
    .dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
    .sender { background: var(--sender); }
    .receiver { background: var(--receiver); }
    .cell { background: #B9C1D1; }
    .edge-mark { width: 20px; height: 2px; background: var(--edge); display: inline-block; opacity: .68; }
    .popover {
      position: absolute; right: 13px; bottom: 50px; z-index: 8;
      width: 294px; display: none; border: 1px solid #C9D5D9;
      background: rgba(255,255,255,.97); box-shadow: 0 16px 38px rgba(25,39,45,.13);
      padding: 10px 11px; border-radius: 8px;
    }
    .popover.on { display: block; }
    .popover-head { display: flex; justify-content: space-between; gap: 8px; align-items: start; }
    .popover-title { font-size: 13px; font-weight: 700; line-height: 1.2; color: var(--ink); }
    .popover-close {
      padding: 0; border: 0; background: transparent; font-size: 14px; line-height: 1;
      color: #7B878D;
    }
    .kv { display: grid; grid-template-columns: 72px minmax(0, 1fr); gap: 8px; padding-top: 7px; }
    .kv span { color: var(--muted); font-size: 10.5px; }
    .kv b { color: var(--ink); font-size: 11.5px; font-weight: 600; overflow-wrap: anywhere; }
    .side {
      min-height: 0; border: 1px solid var(--line); background: #fff;
      padding: 14px 14px 12px; display: flex; flex-direction: column;
      gap: 10px; overflow: hidden; border-radius: 6px;
      box-shadow: none;
    }
    .side-section {
      border: 0; border-bottom: 1px solid #E5ECEA; background: transparent;
      padding: 0 0 10px; border-radius: 0; box-shadow: none;
    }
    .controls-card {
      background: transparent;
    }
    .axis-card {
      border: 0;
      border-bottom: 1px solid #E5ECEA;
      border-radius: 0;
      padding: 0 0 10px;
      background: transparent;
    }
    .section-head {
      display: flex; align-items: center; justify-content: space-between; gap: 8px;
      margin-bottom: 8px;
    }
    .section-head b { font-size: 12px; color: var(--ink); }
    .section-head span { color: var(--muted); font-size: 10.5px; }
    label {
      display: block; color: #4D5A61; font-size: 10px; font-weight: 700;
      margin-bottom: 4px; letter-spacing: .01em;
    }
    input[type="text"], select {
      width: 100%; border: 1px solid #D9E3E6; background: #fff;
      color: var(--ink); outline: none; border-radius: 6px;
      font: 400 12px/1.2 Arial, Helvetica, sans-serif; padding: 7px 8px;
    }
    input[type="text"]:focus, select:focus { border-color: #9CBBC2; box-shadow: 0 0 0 2px rgba(156,187,194,.18); }
    .control-grid { display: grid; grid-template-columns: 1fr; gap: 8px; }
    .lr-strip {
      display: none;
    }
    .lr-chip {
      border: 1px solid #E0E7E5; background: #FAFBFA; color: #516168;
      padding: 6px 7px; border-radius: 7px; cursor: pointer;
      font-size: 10.5px; line-height: 1.12; white-space: nowrap;
      overflow: hidden; text-overflow: ellipsis;
    }
    .lr-chip.active {
      color: #fff; background: #237A78; border-color: #237A78;
      box-shadow: 0 5px 12px rgba(35,122,120,.18);
    }
    .page.paper-view #context-select { display: block; }
    .range-row { display: grid; grid-template-columns: minmax(0, 1fr) 54px; gap: 9px; align-items: center; }
    input[type=range] { width: 100%; accent-color: var(--accent); }
    #top-label { font-size: 13px; color: var(--ink); text-align: right; }
    .lr-title { font-size: 18px; font-weight: 700; line-height: 1; color: var(--ink); }
    .lr-meta { color: var(--muted); font-size: 11px; line-height: 1.28; margin-top: 6px; }
    .stats {
      display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0;
      border-top: 1px solid #E5ECEA; border-bottom: 1px solid #E5ECEA;
    }
    .stat {
      border: 0; background: transparent; padding: 8px 0; min-width: 0;
      border-radius: 0;
    }
    .stat:nth-child(odd) { padding-right: 10px; border-right: 1px solid #E5ECEA; }
    .stat:nth-child(even) { padding-left: 10px; }
    .stat:nth-child(n+3) { border-top: 1px solid #EEF3F1; }
    .stat span {
      display: block; color: var(--muted); font-size: 10px; line-height: 1.1;
      white-space: normal; overflow: visible; text-overflow: clip;
    }
    .stat b { display: block; margin-top: 4px; color: var(--ink); font-size: 15px; line-height: 1; }
    .stat.primary b { color: var(--sender); }
    .stat.secondary b { color: #28343A; }
    .summary-grid { display: grid; grid-template-columns: 1fr; gap: 9px; }
    .mini-title { color: var(--muted); font-size: 10px; margin-bottom: 5px; }    .scale-bars { display: grid; gap: 5px; }
    .scale-row {
      display: grid; grid-template-columns: 39px minmax(0, 1fr) 29px; gap: 6px; align-items: center;
      font-size: 10px; color: var(--muted);
    }
    .scale-row div { height: 7px; background: #EAF0F2; overflow: hidden; border-radius: 999px; }
    .scale-row i { display: block; height: 100%; background: linear-gradient(90deg, #93CC82, #F3D78A, #B9E5FA, #FBC9C4); border-radius: 999px; }
    .scale-row b { text-align: right; font-weight: 600; color: #303C42; }
    .pair-list { overflow: hidden; display: grid; gap: 4px; }
    .pair-row { display: grid; grid-template-columns: minmax(0, 1fr) 42px; gap: 7px; align-items: center; }
    .pair-name { color: #3E4C52; font-size: 10px; line-height: 1.08; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .pair-bar { height: 6px; background: #EDF2F3; overflow: hidden; margin-top: 2px; border-radius: 999px; }
    .pair-bar i { display: block; height: 100%; background: #B9C1D1; border-radius: 999px; }
    .pair-row b { text-align: right; color: var(--ink); font-size: 10.5px; }
    .pair-row:nth-child(n+7) { display: none; }
    .celltype-list { display: grid; gap: 5px; max-height: 126px; overflow: hidden; }
    .celltype-row {
      display: grid;
      grid-template-columns: 11px minmax(0, 1fr) auto;
      gap: 7px;
      align-items: center;
      min-width: 0;
    }
    .celltype-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      box-shadow: 0 0 0 1px rgba(255,255,255,.92), 0 0 0 1.6px rgba(50,60,66,.10);
    }
    .celltype-name {
      color: #344249;
      font-size: 10.5px;
      line-height: 1.1;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .celltype-count {
      display: inline-flex;
      gap: 5px;
      color: var(--muted);
      font-size: 9.5px;
      white-space: nowrap;
    }
    .celltype-count b {
      color: #26343A;
      font-weight: 600;
    }
    .celltype-hint {
      color: var(--muted);
      font-size: 9.5px;
      line-height: 1.25;
      margin-top: -2px;
    }
    .detail-box { min-height: 58px; border: 1px solid #E2E9EC; background: #fff; padding: 8px 9px; border-radius: 6px; }
    .detail-title { font-size: 13px; font-weight: 700; color: var(--ink); line-height: 1.2; margin-bottom: 4px; }
    .detail-text { color: var(--muted); font-size: 11px; line-height: 1.35; }
    .footer-actions { display: grid; grid-template-columns: 1fr 1fr; gap: 7px; margin-top: auto; }
    .footer-actions button { background: #fff; padding: 9px 8px; font-size: 11.5px; }
    .footer-actions button:first-child { background: #EDF4F2; color: #1E615F; border-color: #CFE1DD; font-weight: 700; }
    .footer-actions button:first-child:hover { background: #E4EFEC; color: #174F4D; }
    .paper-card { display: none; }
    .page.paper-view { padding: 10px 14px 12px; grid-template-rows: 42px minmax(0, 1fr); gap: 8px; }
    .page.paper-view .topbar { padding: 7px 10px; }
    .page.paper-view .brand h1 { font-size: 19px; }
    .page.paper-view .brand span { font-size: 11px; }
    .page.paper-view .top-summary { display: none; }
    .page.paper-view .shell { grid-template-columns: minmax(0, 1fr) 306px; gap: 8px; }
    .page.paper-view .canvas-title span { display: none; }
    .page.paper-view .toolbar { display: none; }
    .page.paper-view .side { padding: 12px; gap: 9px; }
    .page.paper-view .side-section { padding: 0 0 9px; }
    .page.paper-view #lr-search-wrap, .page.paper-view .footer-actions, .page.paper-view .tool-only { display: none; }
    .page.paper-view .lr-strip { display: none; }
    .page.paper-view .paper-card { display: none; }
    .page.paper-view .lr-title { font-size: 18px; }
    .page.paper-view .stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .page.paper-view .summary-grid { grid-template-columns: 1fr; gap: 7px; }
    .page.paper-view .pair-row:nth-child(n+4) { display: none; }
    .paper-kicker { color: var(--muted); font-size: 10px; text-transform: uppercase; letter-spacing: .04em; }
    .paper-statement { color: var(--ink); font-size: 12px; line-height: 1.35; margin-top: 5px; }
    .paper-statement b { color: var(--sender); font-weight: 700; }
    .presentation-view {
      padding: 12px 16px 14px;
      grid-template-rows: 44px minmax(0, 1fr);
      background: #FBFCFA;
    }
    .presentation-view .topbar {
      border-radius: 10px;
      padding: 8px 13px;
      box-shadow: 0 10px 24px rgba(23,33,38,.035);
    }
    .presentation-view .shell {
      grid-template-columns: minmax(0, 1fr) 306px;
      gap: 10px;
    }
    .presentation-view .canvas { border-radius: 10px; box-shadow: 0 14px 32px rgba(23,33,38,.050); }
    .presentation-view .side { border-radius: 6px; box-shadow: none; }
    .presentation-view .canvas-title,
    .presentation-view .toolbar,
    .presentation-view .footer-actions,
    .presentation-view #lr-search-wrap,
    .presentation-view #context-select,
    .presentation-view .tool-only {
      display: none;
    }
    .presentation-view .side {
      padding: 12px 12px 11px;
      background: #FFFFFF;
    }
    .presentation-view .side-section {
      padding-bottom: 9px;
      margin-bottom: 0;
      border-bottom-color: #E7EEEC;
    }
    .presentation-view .controls-card label,
    .presentation-view .side-section label {
      font-size: 9px;
    }
    .presentation-view .lr-strip { display: none; }
    .presentation-view .lr-chip {
      padding: 7px 7px;
      font-size: 10px;
      background: #FFFFFF;
    }
    .presentation-view .lr-chip.active {
      background: #237A78;
      color: #fff;
    }
    .presentation-view .axis-card { padding: 0 0 9px; background: transparent; }
    .presentation-view .lr-title {
      font-size: 19px;
    }
    .presentation-view .stats {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    .presentation-view .summary-grid {
      grid-template-columns: 1fr;
      gap: 9px;
    }
    .presentation-view .plot-legend {
      left: 18px;
      bottom: 16px;
      background: rgba(255,255,255,.88);
      border-radius: 8px;
    }
    .presentation-view .detail-box { display: none; }
    .presentation-view .pair-row:nth-child(n+4) {
      display: none;
    }
    .presentation-view .scale-row {
      grid-template-columns: 36px minmax(0, 1fr) 26px;
      gap: 5px;
      font-size: 9px;
    }
    .presentation-view .scale-row div {
      height: 6px;
    }
    .presentation-view .pair-list {
      gap: 5px;
    }
    .presentation-view .celltype-list {
      max-height: 104px;
      gap: 4px;
    }
    .presentation-view .pair-name {
      font-size: 9.5px;
    }
    .capability-strip {
      display: none;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 6px;
    }
    .capability {
      border: 1px solid #E2EAE8;
      background: #FBFDFC;
      border-radius: 8px;
      padding: 7px 8px;
    }
    .capability span {
      display: block;
      color: var(--muted);
      font-size: 9px;
      line-height: 1;
      margin-bottom: 4px;
    }
    .capability b {
      display: block;
      color: var(--ink);
      font-size: 12px;
      line-height: 1;
      white-space: nowrap;
    }
    .presentation-view .capability-strip { display: none; }
    </style>
    """

    body = """
    <div class="page">
      <header class="topbar">
        <div class="brand">
          <h1>ScaleComm Explorer</h1>
          <span>Interactive browser for single-cell spatial communication events</span>
        </div>
        <div class="top-summary">
          <span>Slide-seqV2 hippocampus</span>
          <span class="sep"></span>
          <span><b id="top-lr-name">Cxcl12-Cxcr4</b></span>
          <span class="sep"></span>
          <span id="top-event-summary">404 top events</span>
        </div>
      </header>
      <main class="shell">
        <section class="canvas">
          <div class="canvas-title">
            <b>Spatial communication map</b>
            <span>tissue-surface events; drag, zoom, and click a point or line</span>
          </div>
          <div class="toolbar">
            <div class="view-toggle"><button id="tool-view" class="active">Tool</button><button id="paper-view">Figure</button></div>
            <button class="mode active" data-mode="overview">Overview</button>
            <button class="mode" data-mode="inspect">Inspect</button>
            <button class="mode" data-mode="dense">Dense</button>
            <button id="reset-camera">Reset</button>
          </div>
          <div id="plot"></div>
          <div id="event-popover" class="popover"></div>
          <div class="plot-legend">
            <span><i class="dot cell"></i>cell types</span>
            <span><i class="dot sender"></i>sender</span>
            <span><i class="dot receiver"></i>receiver</span>
            <span><i class="edge-mark"></i>ScaleComm event</span>
          </div>
        </section>
        <aside class="side">
          <section class="side-section controls-card">
            <div class="section-head"><b>Communication axis</b><span id="lr-count"></span></div>
            <div class="control-grid">
              <div id="lr-search-wrap"><label for="lr-search">Search LR or mechanism</label><input id="lr-search" type="text" placeholder="e.g. Cxcl12, chemokine, Reln"></div>
              <div><label for="lr-select">Ligand-receptor pair</label><select id="lr-select"></select></div>
              <div><label for="pair-select">Cell-type pair</label><select id="pair-select"></select></div>
              <div><label for="context-select">Cell context</label><select id="context-select"><option value="muted">Muted tissue</option><option value="involved" selected>Communication context</option><option value="cell_types">All cell types</option></select></div>
              <div><label for="celltype-select">Cell-type focus</label><select id="celltype-select"></select></div>
              <div><label for="top-range">Displayed signal strength</label><div class="range-row"><input id="top-range" type="range" min="5" max="50" step="5" value="20"><span id="top-label">top 20%</span></div></div>
            </div>
          </section>
          <section class="side-section axis-card">
            <div class="lr-title" id="ins-lr"></div>
            <div class="lr-meta" id="ins-meta"></div>
          </section>
          <section class="side-section paper-card">
            <div class="paper-kicker">Screenshot summary</div>
            <div class="paper-statement" id="paper-statement"></div>
          </section>
          <section class="side-section" style="min-height:0; overflow:hidden;">
            <div class="section-head"><b>Cell types in view</b><span>sender / receiver</span></div>
            <div id="celltype-list" class="celltype-list"></div>
          </section>
          <section class="side-section" style="min-height:0; overflow:hidden;">
            <div class="section-head"><b>Dominant cell-type pairs</b><span>embedded pool</span></div>
            <div id="pair-list" class="pair-list"></div>
          </section>
          <section>
            <div id="detail-box" class="detail-box"></div>
          </section>
          <div class="footer-actions">
            <button id="export-png">Export PNG</button>
            <button id="download-csv">Download CSV</button>
          </div>
        </aside>
      </main>
    </div>
    """

    js = """
    <script>
    const DATA = __PAYLOAD__;
    const COLORS = DATA.cell_colors;
    const SENDER = "#237A78", RECEIVER = "#B85450", EDGE = "#1F2A30";
    const plot = document.getElementById("plot");
    const lrSearch = document.getElementById("lr-search");
    const lrSelect = document.getElementById("lr-select");
    const pairSelect = document.getElementById("pair-select");
    const contextSelect = document.getElementById("context-select");
    const celltypeSelect = document.getElementById("celltype-select");
    const topRange = document.getElementById("top-range");
    const topLabel = document.getElementById("top-label");
    const detailBox = document.getElementById("detail-box");
    const popover = document.getElementById("event-popover");
    const modeButtons = Array.from(document.querySelectorAll(".mode[data-mode]"));
    const toolViewButton = document.getElementById("tool-view");
    const paperViewButton = document.getElementById("paper-view");
    const pageRoot = document.querySelector(".page");
    const bg = DATA.background;
    let currentLR = DATA.default_lr;
    let currentPair = "All interactions";
    let currentCellType = "All cell types";
    let currentTop = 20;
    let currentMode = "overview";
    let currentContext = "involved";
    let selected = null;
    let currentShowEdges = [];

    function esc(v){
      if(v===undefined || v===null || String(v)==="nan") return "";
      const d=document.createElement("div"); d.textContent=String(v); return d.innerHTML;
    }
    function safeName(s){ return String(s).replace(/[^A-Za-z0-9]+/g,"_").replace(/^_+|_+$/g,""); }
    function lrData(){ return DATA.lrs[currentLR]; }
    function fmtInt(n){ return Number(n||0).toLocaleString(); }
    function finite(v){ const n=Number(v); return Number.isFinite(n)?n:null; }
    function mechanismLabel(d){ return String(d.mechanism || "ScaleComm-ranked LR").replaceAll("_"," "); }
    function lrLabel(lr){
      const d=DATA.lrs[lr];
      const rank = d.rank_q95 ? `rank ${d.rank_q95}` : "ranked";
      return `${lr} | ${rank} | ${mechanismLabel(d)}`;
    }
    function pairLabel(pair){
      const d=lrData();
      if(pair==="All interactions") return `All interactions | n=${fmtInt(d.embedded_edges)}`;
      if(pair==="Annotated interactions") return `Annotated interactions | n=${fmtInt(d.annotated_edges)}`;
      const hit=(d.pair_summary||[]).find(x=>x.pair===pair);
      return hit ? `${pair} | n=${fmtInt(hit.n_edges)}` : pair;
    }
    function refreshLROptions(){
      const q = lrSearch.value.trim().toLowerCase();
      const all = Object.keys(DATA.lrs);
      const filtered = all.filter(lr => lrLabel(lr).toLowerCase().includes(q));
      const options = filtered.length ? filtered : all;
      lrSelect.innerHTML = "";
      options.forEach(lr => {
        const opt=document.createElement("option");
        opt.value=lr; opt.textContent=lrLabel(lr);
        lrSelect.appendChild(opt);
      });
      if(!options.includes(currentLR)) currentLR = options[0];
      lrSelect.value=currentLR;
      document.getElementById("lr-count").textContent = `${options.length}/${all.length}`;
      refreshPairOptions();
    }
    function setLR(lr){
      currentLR=lr; currentPair="All interactions"; currentCellType="All cell types"; resetDetail(); refreshLROptions(); draw();
    }
    function refreshPairOptions(){
      const d=lrData();
      pairSelect.innerHTML="";
      d.pair_names.forEach(p => {
        const opt=document.createElement("option");
        opt.value=p; opt.textContent=pairLabel(p);
        pairSelect.appendChild(opt);
      });
      if(!d.pair_names.includes(currentPair)) currentPair = d.pair_names[0];
      pairSelect.value = currentPair;
      refreshCellTypeOptions();
    }
    function cellTypesForPair(){
      const e=lrData().edges;
      const types=new Set();
      for(let i=0;i<e.score.length;i++){
        const st=String(e.sender_type[i]), rt=String(e.receiver_type[i]);
        let inPair=false;
        if(currentPair==="All interactions") inPair=true;
        else if(currentPair==="Annotated interactions") inPair=(st!=="unannotated" && rt!=="unannotated");
        else {
          const p=currentPair.split(" -> ");
          inPair=(st===p[0] && rt===p[1]);
        }
        if(!inPair) continue;
        if(st!=="unannotated" && st!=="unknown" && st!=="nan") types.add(st);
        if(rt!=="unannotated" && rt!=="unknown" && rt!=="nan") types.add(rt);
      }
      return Array.from(types).sort((a,b)=>a.localeCompare(b));
    }
    function refreshCellTypeOptions(){
      const options=["All cell types"].concat(cellTypesForPair());
      celltypeSelect.innerHTML="";
      options.forEach(ct=>{
        const opt=document.createElement("option");
        opt.value=ct; opt.textContent=ct;
        celltypeSelect.appendChild(opt);
      });
      if(!options.includes(currentCellType)) currentCellType="All cell types";
      celltypeSelect.value=currentCellType;
    }
    function edgeIndicesForPair(applyTop=true){
      const e=lrData().edges;
      let idx=[];
      for(let i=0;i<e.score.length;i++){
        const st=String(e.sender_type[i]), rt=String(e.receiver_type[i]);
        let keep=false;
        if(currentPair==="All interactions") keep=true;
        else if(currentPair==="Annotated interactions"){
          keep=(st!=="unannotated" && rt!=="unannotated");
        } else {
          const p=currentPair.split(" -> ");
          keep=(st===p[0] && rt===p[1]);
        }
        if(keep && currentCellType!=="All cell types") keep=(st===currentCellType || rt===currentCellType);
        if(keep) idx.push(i);
      }
      idx.sort((a,b)=>Number(e.score[b])-Number(e.score[a]));
      if(!applyTop) return idx;
      const n = Math.min(idx.length, Math.max(1, Math.ceil(idx.length * currentTop / 100)));
      return idx.slice(0,n);
    }
    function curve2d(x0,y0,x1,y1,j){
      const dx=x1-x0, dy=y1-y0, len=Math.max(Math.sqrt(dx*dx+dy*dy), 1e-6);
      const px=-dy/len, py=dx/len, sign=(j%2===0)?1:-1;
      const baseBend = 0.004 + 0.012*((j%17)/16);
      const maxBend = Math.max(0.0012, len * 0.45);
      const bend = sign * Math.min(baseBend, maxBend);
      const xs=[], ys=[];
      for(let k=0;k<=8;k++){
        const t=k/8, arc=Math.sin(Math.PI*t);
        xs.push((1-t)*x0+t*x1+bend*px*arc);
        ys.push((1-t)*y0+t*y1+bend*py*arc);
      }
      xs.push(null); ys.push(null);
      return [xs,ys];
    }
    function backgroundTraces(showIdx){
      const e=lrData().edges;
      const involved = new Set();
      showIdx.forEach(i => { involved.add(e.sender_type[i]); involved.add(e.receiver_type[i]); });
      const traces=[];
      const order = ["unannotated"].concat(Object.keys(COLORS).filter(ct => ct!=="unannotated"));
      order.forEach(ct => {
        const x=[], y=[], cd=[];
        for(let i=0;i<bg.x.length;i++){
          if(String(bg.cell_type[i]) !== ct) continue;
          x.push(bg.x[i]); y.push(bg.y[i]);
          cd.push(["cell", bg.cell_id[i], bg.cell_type[i], bg.niche[i]]);
        }
        if(!x.length) return;
        const highlighted = currentCellType!=="All cell types" ? (ct===currentCellType) : involved.has(ct);
        traces.push({
          type:"scattergl", mode:"markers", name:ct,
          x, y,
          marker:cellMarkerStyle(ct, highlighted),
          customdata:cd,
          hovertemplate:"cell=%{customdata[1]}<br>type=%{customdata[2]}<br>niche=%{customdata[3]}<extra></extra>",
          showlegend:false
        });
      });
      return traces;
    }
    function cellMarkerStyle(ct, highlighted){
      if(currentContext==="cell_types"){
        return {
          size: ct==="unannotated" ? 2.5 : 6.6,
          color: ct==="unannotated" ? "#DCE2E5" : (COLORS[ct] || "#B9C1D1"),
          opacity: ct==="unannotated" ? .10 : .82,
          line:{width:0}
        };
      }
      if(currentContext==="muted"){
        return {
          size: ct==="unannotated" ? 2.4 : 4.45,
          color: ct==="unannotated" ? "#E0E5E7" : "#BFC8CC",
          opacity: ct==="unannotated" ? .10 : .38,
          line:{width:0}
        };
      }
      return {
        size: ct==="unannotated" ? 2.3 : (highlighted ? 7.4 : 4.1),
        color: ct==="unannotated" ? "#E0E5E7" : (highlighted ? (COLORS[ct] || "#B9C1D1") : "#CAD2D5"),
        opacity: ct==="unannotated" ? .07 : (highlighted ? .88 : .24),
        line:{width:0}
      };
    }
    function eventTraces(showIdx){
      const e=lrData().edges;
      const traces = backgroundTraces(showIdx);
      const scores=showIdx.map(i=>Number(e.score[i]));
      const sorted=scores.map((s,i)=>[s,i]).sort((a,b)=>a[0]-b[0]);
      const q=new Array(scores.length); sorted.forEach((x,rank)=>q[x[1]]=(rank+1)/Math.max(scores.length,1));
      const dense = currentMode==="dense";
      const buckets=[
        {ok:p=>p<.58, color:"rgba(91,106,112,.18)", width:dense?0.95:0.68},
        {ok:p=>p>=.58&&p<.88, color:"rgba(48,63,70,.36)", width:dense?1.25:0.88},
        {ok:p=>p>=.88, color:"rgba(23,33,38,.58)", width:dense?1.60:1.12}
      ];
      buckets.forEach(b=>{
        const lx=[],ly=[];
        showIdx.forEach((edgeIdx,j)=>{
          if(!b.ok(q[j])) return;
          const c=curve2d(e.sx[edgeIdx],e.sy[edgeIdx],e.rx[edgeIdx],e.ry[edgeIdx],j);
          lx.push(...c[0]); ly.push(...c[1]);
        });
        if(lx.length) traces.push({
          type:"scatter", mode:"lines", x:lx, y:ly,
          line:{color:b.color,width:b.width}, hoverinfo:"skip", showlegend:false
        });
      });
      const minS=Math.min(...scores), maxS=Math.max(...scores), span=Math.max(maxS-minS,1e-9);
      const sx=[],sy=[],rx=[],ry=[],mx=[],my=[],senderCd=[],receiverCd=[],edgeCd=[];
      showIdx.forEach((edgeIdx)=>{
        sx.push(e.sx[edgeIdx]); sy.push(e.sy[edgeIdx]);
        rx.push(e.rx[edgeIdx]); ry.push(e.ry[edgeIdx]);
        mx.push((e.sx[edgeIdx]+e.rx[edgeIdx])/2); my.push((e.sy[edgeIdx]+e.ry[edgeIdx])/2);
        const dist=Number(e.distance[edgeIdx]).toFixed(1);
        edgeCd.push(["edge",e.sender_id[edgeIdx],e.sender_type[edgeIdx],e.receiver_id[edgeIdx],e.receiver_type[edgeIdx],dist,e.sender_niche[edgeIdx],e.receiver_niche[edgeIdx],e.sx[edgeIdx],e.sy[edgeIdx],e.rx[edgeIdx],e.ry[edgeIdx]]);
        senderCd.push(["sender",e.sender_id[edgeIdx],e.sender_type[edgeIdx],e.sender_niche[edgeIdx],e.receiver_id[edgeIdx],e.receiver_type[edgeIdx],dist,e.sx[edgeIdx],e.sy[edgeIdx],e.rx[edgeIdx],e.ry[edgeIdx]]);
        receiverCd.push(["receiver",e.receiver_id[edgeIdx],e.receiver_type[edgeIdx],e.receiver_niche[edgeIdx],e.sender_id[edgeIdx],e.sender_type[edgeIdx],dist,e.rx[edgeIdx],e.ry[edgeIdx],e.sx[edgeIdx],e.sy[edgeIdx]]);
      });
      const sizes=scores.map(s=>8.8 + 5.2*((s-minS)/span));
      traces.push({
        type:"scattergl", mode:"markers", name:"edge selector",
        x:mx, y:my,
        marker:{size:18.0, color:"rgba(0,0,0,0.02)", opacity:.05, line:{width:0}},
        customdata:edgeCd,
        hovertemplate:"sender=%{customdata[1]} (%{customdata[2]})<br>receiver=%{customdata[3]} (%{customdata[4]})<br>distance=%{customdata[5]}<extra></extra>",
        showlegend:false
      });
      traces.push({
        type:"scattergl", mode:"markers", name:"senders",
        x:sx, y:sy,
        marker:{size:sizes, color:SENDER, opacity:.96, line:{width:.95,color:"#FFFFFF"}},
        customdata:senderCd,
        hovertemplate:"sender=%{customdata[1]}<br>type=%{customdata[2]}<br>receiver=%{customdata[4]} (%{customdata[5]})<br>distance=%{customdata[6]}<extra></extra>",
        showlegend:false
      });
      traces.push({
        type:"scattergl", mode:"markers", name:"receivers",
        x:rx, y:ry,
        marker:{size:sizes.map(v=>v+.85), color:RECEIVER, opacity:.96, line:{width:.95,color:"#FFFFFF"}},
        customdata:receiverCd,
        hovertemplate:"receiver=%{customdata[1]}<br>type=%{customdata[2]}<br>sender=%{customdata[4]} (%{customdata[5]})<br>distance=%{customdata[6]}<extra></extra>",
        showlegend:false
      });
      return traces;
    }
    function selectionTraces(d){
      if(!d) return [];
      let sx,sy,rx,ry;
      if(d[0]==="edge"){ sx=finite(d[8]); sy=finite(d[9]); rx=finite(d[10]); ry=finite(d[11]); }
      else if(d[0]==="sender"){ sx=finite(d[7]); sy=finite(d[8]); rx=finite(d[9]); ry=finite(d[10]); }
      else if(d[0]==="receiver"){ rx=finite(d[7]); ry=finite(d[8]); sx=finite(d[9]); sy=finite(d[10]); }
      else return [];
      if([sx,sy,rx,ry].some(v=>v===null)) return [];
      const c=curve2d(sx,sy,rx,ry,2);
      return [
        {type:"scatter", mode:"lines", x:c[0], y:c[1], line:{color:"#11191D",width:3.0}, hoverinfo:"skip", showlegend:false},
        {type:"scatter", mode:"markers", x:[sx], y:[sy], marker:{size:18,color:SENDER,line:{width:1.8,color:"#fff"}}, hoverinfo:"skip", showlegend:false},
        {type:"scatter", mode:"markers", x:[rx], y:[ry], marker:{size:18,color:RECEIVER,line:{width:1.8,color:"#fff"}}, hoverinfo:"skip", showlegend:false}
      ];
    }
    function layout(){
      const b = plotBounds(currentShowEdges);
      return {
        autosize:true, margin:{l:6,r:6,t:2,b:6}, paper_bgcolor:"#fff", plot_bgcolor:"#fff", showlegend:false,
        xaxis:{visible:false,range:b.x, fixedrange:false,showgrid:false,zeroline:false},
        yaxis:{visible:false,range:b.y, fixedrange:false,showgrid:false,zeroline:false,scaleanchor:"x",scaleratio:1},
        hovermode:"closest",
        uirevision:`${currentLR}-${currentPair}-${currentTop}`
      };
    }
    function plotBounds(showIdx){
      const e=lrData().edges;
      const xs=[], ys=[];
      for(let i=0;i<bg.x.length;i++){
        if(String(bg.cell_type[i]) !== "unannotated"){
          xs.push(Number(bg.x[i])); ys.push(Number(bg.y[i]));
        }
      }
      showIdx.forEach(i=>{
        xs.push(Number(e.sx[i]), Number(e.rx[i]));
        ys.push(Number(e.sy[i]), Number(e.ry[i]));
      });
      const xmin=Math.max(0, Math.min(...xs)), xmax=Math.min(1, Math.max(...xs));
      const ymin=Math.max(0, Math.min(...ys)), ymax=Math.min(1, Math.max(...ys));
      const dx=Math.max(xmax-xmin, .18), dy=Math.max(ymax-ymin, .18);
      const padX=Math.max(.018, dx*.045), padY=Math.max(.019, dy*.050);
      return {
        x:[Math.max(0, xmin-padX), Math.min(1, xmax+padX)],
        y:[Math.max(0, ymin-padY), Math.min(1, ymax+padY)]
      };
    }
    function draw(){
      currentShowEdges=edgeIndicesForPair(true);
      let traces=eventTraces(currentShowEdges);
      if(currentMode==="inspect" && selected){
        traces.forEach(t=>{
          if(t.mode==="lines" && (t.type==="scattergl" || t.type==="scatter")) t.line.color="rgba(97,111,118,.12)";
          if(t.name==="senders"||t.name==="receivers") t.marker.opacity=.34;
        });
      }
      traces=traces.concat(selectionTraces(selected));
      Plotly.react(plot, traces, layout(), {responsive:true, displayModeBar:false, displaylogo:false, scrollZoom:true});
      updateInspector();
    }
    function updateInspector(){
      const d=lrData(), e=d.edges;
      const pool=edgeIndicesForPair(false);
      const visible=currentShowEdges;
      document.getElementById("ins-lr").textContent=currentLR;
      document.getElementById("top-lr-name").textContent=currentLR;
      document.getElementById("top-event-summary").textContent=`${fmtInt(visible.length)} top events`;
      const meta=[
        mechanismLabel(d),
        d.rank_q95 ? `rank ${d.rank_q95}` : null,
        d.median_distance_all ? `median distance ${d.median_distance_all}` : null,
        `${pairLabel(currentPair)}`,
        currentCellType !== "All cell types" ? `focus ${currentCellType}` : null
      ].filter(Boolean).join(" | ");
      document.getElementById("ins-meta").textContent=meta;
      renderCellTypes(visible);
      renderPairs(d.pair_summary || []);
      updatePaperStatement(d, visible.length, pool.length);
    }
    function updatePaperStatement(d, visibleN, poolN){
      const pair = currentPair === "All interactions" ? "all predicted events" : currentPair;
      const scaleVals = d.scale || [];
      const names=["local","near","mid","wide","broad","global"];
      let maxIdx=0;
      scaleVals.forEach((v,i)=>{ if(Number(v)>Number(scaleVals[maxIdx]||0)) maxIdx=i; });
      document.getElementById("paper-statement").innerHTML =
        `<b>${esc(currentLR)}</b> is displayed as ${fmtInt(visibleN)} top-ranked ScaleComm events from ${fmtInt(poolN)} ${esc(pair)}; the learned scale profile is enriched at <b>${names[maxIdx]||"scale "+maxIdx}</b>.`;
    }
    function renderCellTypes(showIdx){
      const box=document.getElementById("celltype-list");
      const e=lrData().edges;
      const counts=new Map();
      showIdx.forEach(i=>{
        const st=String(e.sender_type[i]||"unknown");
        const rt=String(e.receiver_type[i]||"unknown");
        if(st!=="unannotated"){
          const row=counts.get(st)||{sender:0,receiver:0};
          row.sender += 1; counts.set(st,row);
        }
        if(rt!=="unannotated"){
          const row=counts.get(rt)||{sender:0,receiver:0};
          row.receiver += 1; counts.set(rt,row);
        }
      });
      const rows=Array.from(counts.entries())
        .map(([cell_type,c])=>({cell_type, sender:c.sender, receiver:c.receiver, total:c.sender+c.receiver}))
        .sort((a,b)=>b.total-a.total || a.cell_type.localeCompare(b.cell_type))
        .slice(0,8);
      if(!rows.length){
        box.innerHTML='<div class="detail-text">No annotated sender or receiver cell type in the current view.</div>';
        return;
      }
      box.innerHTML="";
      rows.forEach(r=>{
        const color=COLORS[r.cell_type] || "#B9C1D1";
        const active = r.cell_type === currentCellType ? " font-weight:700;" : "";
        box.insertAdjacentHTML("beforeend",
          `<div class="celltype-row"><i class="celltype-dot" style="background:${esc(color)}"></i><div class="celltype-name" style="${active}">${esc(r.cell_type)}</div><div class="celltype-count"><span>S <b>${fmtInt(r.sender)}</b></span><span>R <b>${fmtInt(r.receiver)}</b></span></div></div>`
        );
      });
      if(currentCellType !== "All cell types"){
        box.insertAdjacentHTML("beforeend", `<div class="celltype-hint">Filtered to events where ${esc(currentCellType)} is sender or receiver.</div>`);
      }
    }
    function renderPairs(rows){
      const box=document.getElementById("pair-list"); box.innerHTML="";
      if(!rows.length){ box.innerHTML='<div class="detail-text">No annotated cell-type interaction in the embedded pool.</div>'; return; }
      const max=Math.max(...rows.map(r=>Number(r.n_edges)||0),1);
      rows.slice(0,10).forEach(r=>{
        box.insertAdjacentHTML("beforeend", `<div class="pair-row"><div><div class="pair-name">${esc(r.pair)}</div><div class="pair-bar"><i style="width:${100*(Number(r.n_edges)||0)/max}%"></i></div></div><b>${fmtInt(r.n_edges)}</b></div>`);
      });
    }
    function row(k,v){ return `<div class="kv"><span>${esc(k)}</span><b>${esc(v)}</b></div>`; }
    function detailHtml(d){
      if(!d) return '<div class="detail-title">No event selected</div><div class="detail-text">Click a sender, receiver, or line midpoint to inspect one ScaleComm communication event.</div>';
      const kind=d[0];
      if(kind==="edge") return `<div class="detail-title">${esc(d[1])} -> ${esc(d[3])}</div>` + row("sender",`${d[1]} (${d[2]})`) + row("receiver",`${d[3]} (${d[4]})`) + row("distance",d[5]) + row("sender niche",d[6]) + row("receiver niche",d[7]);
      if(kind==="sender") return `<div class="detail-title">Sender: ${esc(d[1])}</div>` + row("cell type",d[2]) + row("niche",d[3]) + row("receiver",`${d[4]} (${d[5]})`) + row("distance",d[6]);
      if(kind==="receiver") return `<div class="detail-title">Receiver: ${esc(d[1])}</div>` + row("cell type",d[2]) + row("niche",d[3]) + row("sender",`${d[4]} (${d[5]})`) + row("distance",d[6]);
      return `<div class="detail-title">${esc(d[1]||"cell")}</div>` + row("cell type",d[2]) + row("niche",d[3]);
    }
    function updateDetail(d){
      selected=d;
      detailBox.innerHTML=detailHtml(d);
      popover.innerHTML = `<div class="popover-head"><div class="popover-title">Selected communication event</div><button class="popover-close" id="close-popover">x</button></div>${detailHtml(d)}`;
      popover.classList.add("on");
      document.getElementById("close-popover").onclick=()=>{ selected=null; popover.classList.remove("on"); draw(); };
      currentMode="inspect"; modeButtons.forEach(b=>b.classList.toggle("active", b.dataset.mode===currentMode));
      draw();
    }
    function resetDetail(){
      selected=null; popover.classList.remove("on"); detailBox.innerHTML=detailHtml(null);
    }
    function setMode(mode){
      currentMode=mode; modeButtons.forEach(b=>b.classList.toggle("active", b.dataset.mode===mode)); draw();
    }
    function setScreenshotMode(mode){
      pageRoot.classList.toggle("paper-view", mode==="paper");
      pageRoot.classList.toggle("presentation-view", mode==="paper");
      toolViewButton.classList.toggle("active", mode==="tool");
      paperViewButton.classList.toggle("active", mode==="paper");
      setTimeout(()=>Plotly.Plots.resize(plot).then(draw), 50);
    }
    function exportPng(){
      Plotly.downloadImage(plot, {format:"png", width:1800, height:1000, scale:2, filename:`ScaleComm_Explorer_${safeName(currentLR)}_${safeName(currentPair)}_top${currentTop}`});
    }
    function downloadCsv(){
      const e=lrData().edges;
      const header=["LR","sender_id","receiver_id","sender_cell_type","receiver_cell_type","sender_niche","receiver_niche","distance","sender_x","sender_y","receiver_x","receiver_y"];
      const rows=currentShowEdges.map(i=>[currentLR,e.sender_id[i],e.receiver_id[i],e.sender_type[i],e.receiver_type[i],e.sender_niche[i],e.receiver_niche[i],e.distance[i],e.sx[i],e.sy[i],e.rx[i],e.ry[i]]);
      const csv=[header].concat(rows).map(row=>row.map(v=>`"${String(v).replaceAll('"','""')}"`).join(",")).join("\\n");
      const blob=new Blob([csv], {type:"text/csv;charset=utf-8"});
      const url=URL.createObjectURL(blob);
      const a=document.createElement("a"); a.href=url; a.download=`ScaleComm_${safeName(currentLR)}_${safeName(currentPair)}_top${currentTop}_edges.csv`; document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
    }
    lrSearch.oninput=()=>{ refreshLROptions(); resetDetail(); draw(); };
    lrSelect.onchange=()=>{ setLR(lrSelect.value); };
    pairSelect.onchange=()=>{ currentPair=pairSelect.value; currentCellType="All cell types"; refreshCellTypeOptions(); resetDetail(); draw(); };
    celltypeSelect.onchange=()=>{ currentCellType=celltypeSelect.value; resetDetail(); draw(); };
    topRange.oninput=()=>{ currentTop=Number(topRange.value); topLabel.textContent=`top ${currentTop}%`; resetDetail(); draw(); };
    contextSelect.onchange=()=>{ currentContext=contextSelect.value; draw(); };
    modeButtons.forEach(b=>b.onclick=()=>setMode(b.dataset.mode));
    toolViewButton.onclick=()=>setScreenshotMode("tool");
    paperViewButton.onclick=()=>setScreenshotMode("paper");
    document.getElementById("reset-camera").onclick=()=>{ const b=plotBounds(currentShowEdges); Plotly.relayout(plot, {"xaxis.range":b.x, "yaxis.range":b.y}); };
    document.getElementById("export-png").onclick=exportPng;
    document.getElementById("download-csv").onclick=downloadCsv;

    refreshLROptions();
    topRange.value=currentTop; topLabel.textContent=`top ${currentTop}%`;
    contextSelect.value=currentContext;
    resetDetail();
    currentShowEdges=edgeIndicesForPair(true);
    Plotly.newPlot(plot, eventTraces(currentShowEdges), layout(), {responsive:true, displayModeBar:false, displaylogo:false, scrollZoom:true}).then(()=>{
      updateInspector();
      const params = new URLSearchParams(window.location.search);
      if(params.get("view")==="paper") setScreenshotMode("paper");
      plot.on("plotly_click", ev=>{ if(ev?.points?.length && ev.points[0].customdata) updateDetail(ev.points[0].customdata); });
    });
    </script>
    """

    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>ScaleComm Explorer</title>"
        + css
        + "</head><body>"
        + body
        + f"<script>{plotly_js}</script>"
        + js.replace("__PAYLOAD__", payload_json)
        + "</body></html>"
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    html = build_html()
    product = OUT_DIR / "spatialself_explorer_product_ui.html"
    paper = OUT_DIR / "spatialself_explorer_paper_screenshot.html"
    current = OUT_DIR / "spatialself_explorer_static.html"
    product.write_text(html, encoding="utf-8")
    paper_html = (
        html.replace("<title>ScaleComm Explorer</title>", "<title>ScaleComm Explorer Paper View</title>")
        .replace('<div class="page">', '<div class="page paper-view presentation-view">', 1)
        .replace('<button id="tool-view" class="active">Tool</button><button id="paper-view">Figure</button>', '<button id="tool-view">Tool</button><button id="paper-view" class="active">Figure</button>')
    )
    paper.write_text(paper_html, encoding="utf-8")
    current.write_text(html, encoding="utf-8")
    print(product)
    print(paper)
    print(current)


if __name__ == "__main__":
    main()
