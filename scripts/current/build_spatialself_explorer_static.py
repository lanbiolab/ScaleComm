#!/usr/bin/env python
"""Build a standalone interactive ScaleComm Explorer HTML page.

The static page is intended for supplement/demo use. It embeds a compact
subset of the Slide-seqV2 ScaleComm output so the browser can switch LR
pairs, cell-type pairs, and edge thresholds without a Python server.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("PANDAS_USE_NUMEXPR", "0")
os.environ.setdefault("PANDAS_USE_BOTTLENECK", "0")

import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs

from apps.spatialself_viewer.app_explorer import CELL_COLORS, load_slideseq_data


OUT_DIR = ROOT / "outputs/current_web_ui"
DEFAULT_LR = "Cxcl12-Cxcr4"
TOP_LR_N = 14
BACKGROUND_MAX = 18000
EDGE_MAX_PER_LR = 3000
FULL_EDGE_PATH = ROOT / "outputs/current_slideseqv2_hpc_full/spatialself_scale_full_e80/merged_predictions.csv"
FULL_EDGE_CHUNK_SIZE = 250_000


def _round_list(values: pd.Series | np.ndarray, digits: int = 5) -> list[float]:
    return np.round(np.asarray(values, dtype=float), digits).tolist()


def _scale(values: pd.Series, vmin: float, vmax: float) -> np.ndarray:
    span = max(float(vmax - vmin), 1e-9)
    return (pd.to_numeric(values, errors="coerce").to_numpy(float) - vmin) / span


def _stable_hash_mod(value: str, modulus: int = 10_000) -> int:
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulus


def _pair_summary(lr_edges: pd.DataFrame) -> pd.DataFrame:
    annotated = lr_edges[
        ~lr_edges["sender_cell_type"].astype(str).eq("unannotated")
        & ~lr_edges["receiver_cell_type"].astype(str).eq("unannotated")
    ].copy()
    if annotated.empty:
        return pd.DataFrame(columns=["pair", "n_edges", "mean_score", "median_distance"])
    annotated["pair"] = annotated["sender_cell_type"].astype(str) + " -> " + annotated["receiver_cell_type"].astype(str)
    return (
        annotated.groupby("pair", as_index=False)
        .agg(n_edges=("score", "size"), mean_score=("score", "mean"), median_distance=("distance", "median"))
        .sort_values(["n_edges", "mean_score"], ascending=False)
        .reset_index(drop=True)
    )


def _attach_edge_positions(edges: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    pos = cells[["cell_id", "x_plot", "y_plot", "cell_type", "niche"]].drop_duplicates("cell_id")
    sender = pos.rename(
        columns={
            "cell_id": "sender_id",
            "x_plot": "sender_x_plot",
            "y_plot": "sender_y_plot",
            "cell_type": "sender_cell_type_meta",
            "niche": "sender_niche_meta",
        }
    )
    receiver = pos.rename(
        columns={
            "cell_id": "receiver_id",
            "x_plot": "receiver_x_plot",
            "y_plot": "receiver_y_plot",
            "cell_type": "receiver_cell_type_meta",
            "niche": "receiver_niche_meta",
        }
    )
    out = edges.merge(sender, on="sender_id", how="left").merge(receiver, on="receiver_id", how="left")
    out["sender_cell_type"] = out["sender_cell_type"].where(out["sender_cell_type"].ne("unknown"), out["sender_cell_type_meta"])
    out["receiver_cell_type"] = out["receiver_cell_type"].where(out["receiver_cell_type"].ne("unknown"), out["receiver_cell_type_meta"])
    out["sender_niche"] = out["sender_niche"].where(out["sender_niche"].ne("unknown"), out["sender_niche_meta"])
    out["receiver_niche"] = out["receiver_niche"].where(out["receiver_niche"].ne("unknown"), out["receiver_niche_meta"])
    return out.dropna(subset=["sender_x_plot", "sender_y_plot", "receiver_x_plot", "receiver_y_plot", "score"]).copy()


def _load_full_ranked_edges(cells: pd.DataFrame, ranked_lrs: list[str]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load a larger per-LR UI edge pool from full predictions.

    The standalone explorer should remain responsive, so it embeds the top
    annotated events per LR rather than all 1.38M scored candidates.
    """
    if not FULL_EDGE_PATH.exists():
        return pd.DataFrame(), {}

    wanted = set(ranked_lrs)
    usecols = [
        "candidate_id",
        "sender_id",
        "receiver_id",
        "ligand",
        "receptor",
        "mechanism",
        "spatial_range",
        "active_niche",
        "distance",
        "sender_cell_type",
        "receiver_cell_type",
        "sender_niche",
        "receiver_niche",
        "ligand_expr",
        "receptor_expr",
        "score",
    ]
    chunks: list[pd.DataFrame] = []
    annotated_counts: dict[str, int] = {}
    for chunk in pd.read_csv(FULL_EDGE_PATH, usecols=usecols, chunksize=FULL_EDGE_CHUNK_SIZE):
        chunk["sender_id"] = chunk["sender_id"].astype(str)
        chunk["receiver_id"] = chunk["receiver_id"].astype(str)
        chunk["ligand"] = chunk["ligand"].astype(str)
        chunk["receptor"] = chunk["receptor"].astype(str)
        chunk["LR"] = chunk["ligand"] + "-" + chunk["receptor"]
        chunk = chunk[chunk["LR"].isin(wanted)].copy()
        if chunk.empty:
            continue
        for col in ["score", "distance", "ligand_expr", "receptor_expr"]:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        for col in ["sender_cell_type", "receiver_cell_type", "sender_niche", "receiver_niche", "mechanism"]:
            chunk[col] = chunk[col].fillna("unknown").astype(str)
        chunk = _attach_edge_positions(chunk, cells)
        chunk = chunk[
            ~chunk["sender_cell_type"].astype(str).eq("unannotated")
            & ~chunk["receiver_cell_type"].astype(str).eq("unannotated")
        ].copy()
        if chunk.empty:
            continue
        for lr, n in chunk["LR"].value_counts().items():
            annotated_counts[str(lr)] = annotated_counts.get(str(lr), 0) + int(n)
        chunks.append(chunk)

    if not chunks:
        return pd.DataFrame(), annotated_counts

    edges = pd.concat(chunks, ignore_index=True)
    edges = (
        edges.sort_values(["LR", "score"], ascending=[True, False])
        .groupby("LR", group_keys=False, sort=False)
        .head(EDGE_MAX_PER_LR)
        .reset_index(drop=True)
    )
    return edges, annotated_counts


def build_payload() -> dict:
    tables = load_slideseq_data()
    cells = tables["cells"].copy()
    fallback_edges = tables["edges"].copy()
    ranking = tables["ranking"].copy()
    scale = tables["scale"].copy()

    annotated = cells[~cells["cell_type"].astype(str).eq("unannotated")].copy()
    ranked_lrs = (
        ranking
        .sort_values("rank_q95", ascending=True)["LR"]
        .head(TOP_LR_N)
        .tolist()
    )
    if DEFAULT_LR in set(ranking["LR"].astype(str)) and DEFAULT_LR not in ranked_lrs:
        ranked_lrs = [DEFAULT_LR] + ranked_lrs[: TOP_LR_N - 1]

    full_edges, annotated_counts = _load_full_ranked_edges(cells, ranked_lrs)
    if full_edges.empty:
        edges = fallback_edges[
            ~fallback_edges["sender_cell_type"].astype(str).eq("unannotated")
            & ~fallback_edges["receiver_cell_type"].astype(str).eq("unannotated")
        ].copy()
        annotated_counts = edges["LR"].value_counts().astype(int).to_dict()
    else:
        edges = full_edges.copy()

    lr_edges_all = edges[edges["LR"].isin(ranked_lrs)].copy()
    xmin = float(min(annotated["x_plot"].min(), lr_edges_all["sender_x_plot"].min(), lr_edges_all["receiver_x_plot"].min()))
    xmax = float(max(annotated["x_plot"].max(), lr_edges_all["sender_x_plot"].max(), lr_edges_all["receiver_x_plot"].max()))
    ymin = float(min(annotated["y_plot"].min(), lr_edges_all["sender_y_plot"].min(), lr_edges_all["receiver_y_plot"].min()))
    ymax = float(max(annotated["y_plot"].max(), lr_edges_all["sender_y_plot"].max(), lr_edges_all["receiver_y_plot"].max()))

    # The Explorer is a cell-type-resolved browser. Keep only annotated beads so
    # the view focuses on the central region used for biological interpretation.
    bg = annotated.copy()
    bg["_sample_key"] = bg["cell_id"].astype(str).map(_stable_hash_mod)
    if len(bg) > BACKGROUND_MAX:
        bg = bg.sort_values("_sample_key").head(BACKGROUND_MAX)
    bg = bg.drop(columns=["_sample_key"], errors="ignore")

    background = {
        "x": _round_list(_scale(bg["x_plot"], xmin, xmax), 5),
        "y": _round_list(_scale(bg["y_plot"], ymin, ymax), 5),
        "cell_type": bg["cell_type"].astype(str).tolist(),
        "niche": bg["niche"].astype(str).tolist(),
        "cell_id": bg["cell_id"].astype(str).tolist(),
    }

    lr_payload: dict[str, dict] = {}
    for lr in ranked_lrs:
        lr_edges = edges[edges["LR"].eq(lr)].sort_values("score", ascending=False).copy()
        lr_edges = lr_edges.head(min(len(lr_edges), EDGE_MAX_PER_LR)).copy()
        pairs = _pair_summary(lr_edges)
        all_annotated = lr_edges
        pair_names = ["All interactions"] + pairs.head(24)["pair"].astype(str).tolist()
        scale_row = scale[scale["LR"].eq(lr)]
        scale_vals = []
        if not scale_row.empty:
            for i in range(6):
                col = f"scale_{i}"
                scale_vals.append(float(pd.to_numeric(scale_row[col], errors="coerce").iloc[0]) if col in scale_row else 0.0)
        rank_row = ranking[ranking["LR"].eq(lr)]
        mechanism = "ScaleComm-ranked LR"
        rank_q95 = None
        median_distance = None
        if not rank_row.empty and "mechanism" in rank_row:
            mechanism = str(rank_row["mechanism"].iloc[0]).replace("_", " ")
            rank_q95 = int(rank_row["rank_q95"].iloc[0]) if "rank_q95" in rank_row else None
            median_distance = float(rank_row["median_distance"].iloc[0]) if "median_distance" in rank_row else None

        lr_payload[lr] = {
            "mechanism": mechanism,
            "rank_q95": rank_q95,
            "median_distance_all": round(median_distance, 1) if median_distance is not None else None,
            "all_lr_edges": int(annotated_counts.get(lr, len(lr_edges))),
            "embedded_edges": int(len(lr_edges)),
            "annotated_edges": int(annotated_counts.get(lr, len(all_annotated))),
            "pair_names": pair_names,
            "pair_summary": [
                {
                    "pair": str(row.pair),
                    "n_edges": int(row.n_edges),
                    "median_distance": round(float(row.median_distance), 1) if pd.notna(row.median_distance) else None,
                }
                for row in pairs.head(12).itertuples(index=False)
            ],
            "scale": [round(v, 4) for v in scale_vals],
            "edges": {
                "sender_id": lr_edges["sender_id"].astype(str).tolist(),
                "receiver_id": lr_edges["receiver_id"].astype(str).tolist(),
                "sender_type": lr_edges["sender_cell_type"].astype(str).tolist(),
                "receiver_type": lr_edges["receiver_cell_type"].astype(str).tolist(),
                "sender_niche": lr_edges["sender_niche"].astype(str).tolist(),
                "receiver_niche": lr_edges["receiver_niche"].astype(str).tolist(),
                "sx": _round_list(_scale(lr_edges["sender_x_plot"], xmin, xmax), 6),
                "sy": _round_list(_scale(lr_edges["sender_y_plot"], ymin, ymax), 6),
                "rx": _round_list(_scale(lr_edges["receiver_x_plot"], xmin, xmax), 6),
                "ry": _round_list(_scale(lr_edges["receiver_y_plot"], ymin, ymax), 6),
                "score": _round_list(lr_edges["score"], 6),
                "distance": _round_list(lr_edges["distance"], 2),
            },
            "annotated_pool": int(len(all_annotated)),
        }

    return {
        "dataset": "Slide-seqV2 hippocampus",
        "default_lr": DEFAULT_LR if DEFAULT_LR in lr_payload else ranked_lrs[0],
        "total_cells": int(len(cells)),
        "annotated_cells": int(len(annotated)),
        "background": background,
        "cell_colors": CELL_COLORS,
        "lrs": lr_payload,
    }


def build_html() -> str:
    payload = build_payload()
    plotly_js = get_plotlyjs()
    payload_json = json.dumps(payload, separators=(",", ":"))
    css = """
    <style>
    html, body { margin:0; padding:0; width:100vw; height:100vh; overflow:hidden; background:#fff; font-family:Arial,Helvetica,sans-serif; color:#20272C; }
    .page { width:100vw; height:100vh; box-sizing:border-box; padding:14px 24px; overflow:hidden; background:#fff; }
    .topbar { display:flex; align-items:center; justify-content:space-between; gap:1rem; border:1px solid #E3E8EC; padding:8px 12px; margin-bottom:10px; }
    .title { font-size:20px; font-weight:700; line-height:1; }
    .subtitle { font-size:12px; color:#66727A; margin-top:5px; }
    .badges { display:flex; gap:6px; flex-wrap:wrap; justify-content:flex-end; }
    .badge { border:1px solid #E3E8EC; background:#FBFCFD; padding:4px 7px; font-size:11px; color:#20272C; }
    .controls { display:grid; grid-template-columns:1.00fr 1.12fr 1.52fr .78fr; gap:14px; margin-bottom:8px; }
    .control { border:1px solid #E3E8EC; background:#fff; padding:8px 10px; min-height:48px; box-sizing:border-box; }
    .control label { display:block; font-size:11px; font-weight:700; margin-bottom:5px; color:#20272C; }
    .control select, .control input[type=range] { width:100%; }
    .control select { appearance:none; border:0; outline:0; padding:0; background:transparent; font-size:15px; color:#20272C; }
    .control .static-value { font-size:15px; color:#20272C; }
    .range-line { display:grid; grid-template-columns:1fr 36px; gap:8px; align-items:center; }
    input[type=range] { accent-color:#78B66B; }
    #top-label { font-size:15px; color:#20272C; }
    .caption { color:#66727A; font-size:11px; margin:0 0 4px 0; }
    #shell { height:704px; display:grid; grid-template-columns:minmax(0,1fr) 324px; gap:12px; }
    #canvas { position:relative; border:1px solid #E3E8EC; background:#fff; overflow:hidden; }
    #plot { width:100%; height:100%; }
    .toolbar { position:absolute; top:12px; left:12px; z-index:10; display:flex; gap:6px; }
    .mode { border:1px solid rgba(211,218,223,.95); background:rgba(255,255,255,.86); color:#4D5961; font:400 11px/1 Arial,Helvetica,sans-serif; padding:6px 8px; cursor:pointer; }
    .mode.active { background:#20272C; border-color:#20272C; color:#fff; }
    .tool-action { border:1px solid rgba(211,218,223,.95); background:rgba(255,255,255,.86); color:#4D5961; font:400 11px/1 Arial,Helvetica,sans-serif; padding:6px 8px; cursor:pointer; }
    .tool-action:hover, .mode:hover { background:#fff; color:#20272C; }
    .layer-key { position:absolute; right:13px; bottom:12px; z-index:10; display:grid; gap:5px; border:1px solid rgba(227,232,236,.78); background:rgba(255,255,255,.72); padding:7px 8px; color:#526069; font-size:10.5px; line-height:1.1; }
    .layer-key div { display:flex; align-items:center; gap:6px; white-space:nowrap; }
    .dot { width:8px; height:8px; border-radius:999px; display:inline-block; box-shadow:0 0 0 1px rgba(255,255,255,.9); }
    .sender-dot { background:#78B66B; } .receiver-dot { background:#536FA8; }
    .line-key { display:inline-block; width:18px; height:2px; background:#2D3A40; }
    .canvas-note { position:absolute; left:13px; bottom:12px; z-index:10; border:1px solid rgba(227,232,236,.82); background:rgba(255,255,255,.82); color:#66727A; font-size:10.5px; padding:6px 8px; }
    #inspector { border:1px solid #E3E8EC; background:#FBFCFD; padding:13px; overflow:hidden; }
    .eyebrow { color:#66727A; font-size:10px; letter-spacing:.055em; text-transform:uppercase; line-height:1; }
    .lr-title { margin-top:6px; font-size:22px; font-weight:700; line-height:1.04; color:#20272C; }
    .pair-title { margin-top:5px; color:#3E4A52; font-size:12px; line-height:1.25; }
    .pill-row { display:flex; gap:5px; flex-wrap:wrap; margin:10px 0 10px; }
    .pill-row span { border:1px solid #E3E8EC; background:#fff; color:#66727A; font-size:10px; padding:4px 6px; line-height:1; }
    .metric-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:5px; margin-bottom:9px; }
    .metric { background:#fff; border:1px solid #E3E8EC; padding:5px 7px; min-height:35px; }
    .metric span { display:block; color:#66727A; font-size:10px; line-height:1.05; }
    .metric b { display:block; color:#20272C; font-size:15px; line-height:1.12; margin-top:4px; }
    .section { border-top:1px solid #E3E8EC; padding-top:8px; margin-top:8px; }
    .section-head { font-size:12px; font-weight:700; color:#20272C; margin-bottom:6px; }
    .detail-box { background:#fff; border:1px solid #E3E8EC; padding:7px 8px; min-height:50px; }
    .detail-title { font-size:13px; font-weight:700; line-height:1.2; color:#20272C; margin-bottom:6px; }
    .detail-text, .detail-row { color:#4C5860; font-size:11px; line-height:1.42; }
    .detail-row { display:grid; grid-template-columns:70px minmax(0,1fr); gap:7px; border-top:1px solid #EDF1F4; padding:5px 0; }
    .detail-row:first-of-type { border-top:0; } .detail-row span { color:#66727A; } .detail-row b { color:#20272C; font-weight:600; word-break:break-word; }
    .scale-box { display:grid; gap:5px; }
    .scale-row { display:grid; grid-template-columns:24px minmax(0,1fr) 34px; gap:7px; align-items:center; color:#66727A; font-size:10px; }
    .scale-row div { height:7px; background:#E9EEF1; overflow:hidden; }
    .scale-row i { display:block; height:100%; background:linear-gradient(90deg,#78B66B,#536FA8); }
    .scale-row b { color:#20272C; font-weight:600; text-align:right; }
    </style>
    """
    js = """
    <script>
    const DATA = __PAYLOAD__;
    const CELL_COLORS = DATA.cell_colors;
    const SENDER = "#78B66B", RECEIVER = "#536FA8";
    let currentLR = DATA.default_lr;
    let currentPair = "All interactions";
    let currentTop = 20;
    let currentMode = "overview";
    let selected = null;
    let currentShowEdges = [];
    const bg = DATA.background;
    const plot = document.getElementById("plot");
    const lrSelect = document.getElementById("lr-select");
    const pairSelect = document.getElementById("pair-select");
    const topRange = document.getElementById("top-range");
    const topLabel = document.getElementById("top-label");
    const detail = document.getElementById("detail-box");
    const modeButtons = Array.from(document.querySelectorAll(".mode[data-mode]"));
    const defaultCamera = {eye:{x:0.72,y:-0.98,z:0.54}, center:{x:0.02,y:0,z:-0.10}, projection:{type:"perspective"}};

    function esc(v){ if(v===undefined||v===null||String(v)==="nan") return ""; const d=document.createElement("div"); d.textContent=String(v); return d.innerHTML; }
    function row(k,v){ if(v===undefined||v===null||v==="") return ""; return `<div class="detail-row"><span>${esc(k)}</span><b>${esc(v)}</b></div>`; }
    function uniq(arr){ return Array.from(new Set(arr)); }
    function finite(v){ const n=Number(v); return Number.isFinite(n)?n:null; }
    function lrData(){ return DATA.lrs[currentLR]; }
    function formatPair(p){ return p.replaceAll("_"," "); }
    function edgeIndicesForState(){
      const e = lrData().edges;
      let idx = [];
      for(let i=0;i<e.score.length;i++){
        if(currentPair === "All interactions"){
          idx.push(i);
        } else {
          const parts = currentPair.split(" -> ");
          if(e.sender_type[i] === parts[0] && e.receiver_type[i] === parts[1]) idx.push(i);
        }
      }
      idx.sort((a,b)=>e.score[b]-e.score[a]);
      const n = Math.min(idx.length, Math.max(20, Math.ceil(idx.length * currentTop / 100)));
      return idx.slice(0,n);
    }
    function curved(x0,y0,x1,y1,idx){
      const dx=x1-x0, dy=y1-y0, len=Math.max(Math.sqrt(dx*dx+dy*dy), 1e-6);
      const px=-dy/len, py=dx/len, sign=(idx%2===0)?1:-1;
      let bend=sign*(0.012+0.018*((idx%11)/10)); if(len<0.030) bend*=1.65;
      const ts=[0,.22,.5,.78,1], xs=[], ys=[], zs=[];
      ts.forEach(t=>{ const arc=Math.sin(Math.PI*t); xs.push((1-t)*x0+t*x1+bend*px*arc); ys.push((1-t)*y0+t*y1+bend*py*arc); zs.push(1-t); });
      xs.push(null); ys.push(null); zs.push(null); return [xs,ys,zs];
    }
    function baseTraces(showIdx){
      const e = lrData().edges;
      const contextTypes = uniq(showIdx.flatMap(i=>[e.sender_type[i], e.receiver_type[i]])).filter(x=>!["unannotated","unknown","nan"].includes(x)).slice(0,6);
      const traces = [];
      [[1,"#F4F8F2","rgba(120,182,107,0.22)"],[0,"#F4F6FB","rgba(83,111,168,0.22)"]].forEach(layer=>{
        traces.push({type:"surface", x:[[0,1],[0,1]], y:[[0,0],[1,1]], z:[[layer[0],layer[0]],[layer[0],layer[0]]], surfacecolor:[[0,0],[0,0]], colorscale:[[0,layer[1]],[1,layer[1]]], opacity:.18, showscale:false, hoverinfo:"skip", showlegend:false});
        traces.push({type:"scatter3d", mode:"lines", x:[0,1,1,0,0], y:[0,0,1,1,0], z:[layer[0],layer[0],layer[0],layer[0],layer[0]], line:{color:layer[2], width:1.4}, hoverinfo:"skip", showlegend:false});
      });
      traces.push({type:"scatter3d", mode:"markers", name:"annotated tissue", x:bg.x, y:bg.y, z:bg.x.map(()=>-.025), marker:{size:1.45,color:"#CAD1D6",opacity:.18,line:{width:0}}, customdata:bg.cell_id.map((id,i)=>["cell",id,bg.cell_type[i],bg.niche[i]]), hovertemplate:"cell=%{customdata[1]}<br>type=%{customdata[2]}<br>niche=%{customdata[3]}<extra></extra>", showlegend:false});
      contextTypes.forEach(ct=>{
        const x=[], y=[], z=[], cd=[];
        for(let i=0;i<bg.x.length;i++){ if(bg.cell_type[i]===ct){ x.push(bg.x[i]); y.push(bg.y[i]); z.push(-.015); cd.push(["cell",bg.cell_id[i],ct,bg.niche[i]]); } }
        traces.push({type:"scatter3d", mode:"markers", name:ct, x, y, z, marker:{size:2.35,color:CELL_COLORS[ct]||"#B9C1D1",opacity:.54,line:{width:.1,color:"white"}}, customdata:cd, hovertemplate:"cell=%{customdata[1]}<br>type=%{customdata[2]}<br>niche=%{customdata[3]}<extra></extra>", showlegend:false});
      });
      const scores = showIdx.map(i=>Number(e.score[i]));
      const sorted = scores.map((s,i)=>[s,i]).sort((a,b)=>a[0]-b[0]);
      const rankPct = new Array(scores.length); sorted.forEach((x,rank)=>rankPct[x[1]]=(rank+1)/scores.length);
      const buckets = [
        {mask:(q)=>q<.70, width:.62, opacity:.115, color:"#9EADB2"},
        {mask:(q)=>q>=.70&&q<.90, width:1.02, opacity:.29, color:"#64777C"},
        {mask:(q)=>q>=.90, width:1.62, opacity:.60, color:"#2D3A40"}
      ];
      buckets.forEach((b,bi)=>{
        const lx=[],ly=[],lz=[];
        showIdx.forEach((edgeIdx,j)=>{ if(!b.mask(rankPct[j])) return; const c=curved(e.sx[edgeIdx],e.sy[edgeIdx],e.rx[edgeIdx],e.ry[edgeIdx],j); lx.push(...c[0]); ly.push(...c[1]); lz.push(...c[2]); });
        if(lx.length) traces.push({type:"scatter3d", mode:"lines", name:bi===2?"predicted communication":undefined, x:lx, y:ly, z:lz, line:{color:b.color,width:b.width}, opacity:b.opacity, hoverinfo:"skip", showlegend:false});
      });
      const minS = Math.min(...scores), maxS=Math.max(...scores), span=Math.max(maxS-minS,1e-9);
      const sx=[],sy=[],rx=[],ry=[],senderCd=[],receiverCd=[],edgeCd=[],midX=[],midY=[];
      showIdx.forEach((edgeIdx,j)=>{
        sx.push(e.sx[edgeIdx]); sy.push(e.sy[edgeIdx]); rx.push(e.rx[edgeIdx]); ry.push(e.ry[edgeIdx]); midX.push((e.sx[edgeIdx]+e.rx[edgeIdx])/2); midY.push((e.sy[edgeIdx]+e.ry[edgeIdx])/2);
        edgeCd.push(["edge",e.sender_id[edgeIdx],e.sender_type[edgeIdx],e.receiver_id[edgeIdx],e.receiver_type[edgeIdx],Number(e.distance[edgeIdx]).toFixed(1),e.sender_niche[edgeIdx],e.receiver_niche[edgeIdx],e.sx[edgeIdx],e.sy[edgeIdx],e.rx[edgeIdx],e.ry[edgeIdx]]);
        senderCd.push(["sender",e.sender_id[edgeIdx],e.sender_type[edgeIdx],e.sender_niche[edgeIdx],e.receiver_id[edgeIdx],e.receiver_type[edgeIdx],Number(e.distance[edgeIdx]).toFixed(1),e.sx[edgeIdx],e.sy[edgeIdx],e.rx[edgeIdx],e.ry[edgeIdx]]);
        receiverCd.push(["receiver",e.receiver_id[edgeIdx],e.receiver_type[edgeIdx],e.receiver_niche[edgeIdx],e.sender_id[edgeIdx],e.sender_type[edgeIdx],Number(e.distance[edgeIdx]).toFixed(1),e.rx[edgeIdx],e.ry[edgeIdx],e.sx[edgeIdx],e.sy[edgeIdx]]);
      });
      const sizes = scores.map(s=>5+2.8*((s-minS)/span));
      const rSizes = scores.map(s=>5.2+3*((s-minS)/span));
      traces.push({type:"scatter3d", mode:"markers", name:"edge selector", x:midX, y:midY, z:midX.map(()=>.5), marker:{size:8,color:"rgba(0,0,0,0.01)",opacity:.03,line:{width:0}}, customdata:edgeCd, hovertemplate:"sender=%{customdata[1]} (%{customdata[2]})<br>receiver=%{customdata[3]} (%{customdata[4]})<br>distance=%{customdata[5]}<extra></extra>", showlegend:false});
      traces.push({type:"scatter3d", mode:"markers", name:"sending cells", x:sx, y:sy, z:sx.map(()=>1), marker:{size:sizes,color:SENDER,opacity:.94,line:{width:.58,color:"white"}}, customdata:senderCd, hovertemplate:"sender=%{customdata[1]}<br>type=%{customdata[2]}<br>niche=%{customdata[3]}<br>receiver=%{customdata[4]} (%{customdata[5]})<br>distance=%{customdata[6]}<extra></extra>"});
      traces.push({type:"scatter3d", mode:"markers", name:"receiving cells", x:rx, y:ry, z:rx.map(()=>0), marker:{size:rSizes,color:RECEIVER,opacity:.95,line:{width:.58,color:"white"}}, customdata:receiverCd, hovertemplate:"receiver=%{customdata[1]}<br>type=%{customdata[2]}<br>niche=%{customdata[3]}<br>sender=%{customdata[4]} (%{customdata[5]})<br>distance=%{customdata[6]}<extra></extra>"});
      return traces;
    }
    function selectionTraces(d){
      if(!d) return []; let sx,sy,rx,ry;
      if(d[0]==="edge"){ sx=finite(d[8]); sy=finite(d[9]); rx=finite(d[10]); ry=finite(d[11]); }
      else if(d[0]==="sender"){ sx=finite(d[7]); sy=finite(d[8]); rx=finite(d[9]); ry=finite(d[10]); }
      else if(d[0]==="receiver"){ rx=finite(d[7]); ry=finite(d[8]); sx=finite(d[9]); sy=finite(d[10]); }
      else return [];
      if([sx,sy,rx,ry].some(v=>v===null)) return [];
      const c=curved(sx,sy,rx,ry,1);
      return [
        {type:"scatter3d", mode:"lines", x:c[0], y:c[1], z:c[2], line:{color:"#11181D",width:5}, opacity:.78, hoverinfo:"skip", showlegend:false},
        {type:"scatter3d", mode:"markers", x:[sx], y:[sy], z:[1], marker:{size:9,color:SENDER,line:{width:1.3,color:"#fff"}}, hoverinfo:"skip", showlegend:false},
        {type:"scatter3d", mode:"markers", x:[rx], y:[ry], z:[0], marker:{size:9,color:RECEIVER,line:{width:1.3,color:"#fff"}}, hoverinfo:"skip", showlegend:false}
      ];
    }
    function layout(){
      return {height:704, margin:{l:0,r:0,t:0,b:0}, paper_bgcolor:"#fff", plot_bgcolor:"#fff", showlegend:false,
        scene:{domain:{x:[0,1],y:[0,1]}, xaxis:{visible:false,showgrid:false,zeroline:false,showbackground:false}, yaxis:{visible:false,showgrid:false,zeroline:false,showbackground:false}, zaxis:{visible:false,showgrid:false,zeroline:false,showbackground:false,range:[-.08,1.08]}, aspectmode:"manual", aspectratio:{x:2.55,y:.92,z:.56}, camera:defaultCamera},
        hovermode:"closest", uirevision:`${currentLR}-${currentPair}-${currentTop}`};
    }
    function draw(){
      currentShowEdges = edgeIndicesForState();
      let traces = baseTraces(currentShowEdges);
      if(currentMode==="focus"){
        traces.forEach(t=>{ if(t.mode==="lines") t.opacity=.055; if(t.name==="sending cells"||t.name==="receiving cells") t.marker.opacity=selected?.length ? .24 : .82; });
      } else if(currentMode==="network"){
        traces.forEach(t=>{ if(t.mode==="lines") t.opacity=Math.min((t.opacity||.2)*1.75,.65); });
      }
      traces = traces.concat(selectionTraces(selected));
      Plotly.react(plot, traces, layout(), {responsive:true, displayModeBar:false, displaylogo:false, scrollZoom:true});
      updateInspector();
    }
    function updatePairOptions(){
      const d=lrData(); pairSelect.innerHTML="";
      d.pair_names.forEach(p=>{ const opt=document.createElement("option"); opt.value=p; opt.textContent=formatPair(p); pairSelect.appendChild(opt); });
      currentPair=d.pair_names.includes(currentPair)?currentPair:d.pair_names[0];
      pairSelect.value=currentPair;
    }
    function updateControls(){
      Object.keys(DATA.lrs).forEach(lr=>{ const opt=document.createElement("option"); opt.value=lr; opt.textContent=lr; lrSelect.appendChild(opt); });
      lrSelect.value=currentLR; topRange.value=currentTop; topLabel.textContent=`top ${currentTop}%`; updatePairOptions();
    }
    function updateInspector(){
      const d=lrData(), e=d.edges, idx=currentShowEdges;
      const dist=idx.map(i=>Number(e.distance[i])).sort((a,b)=>a-b);
      const selectedPool=edgeIndicesForPair(false).length;
      document.getElementById("ins-lr").textContent=currentLR;
      document.getElementById("ins-pair").textContent=formatPair(currentPair);
      document.getElementById("pill-top").textContent=`top ${currentTop}%`;
      document.getElementById("pill-mech").textContent=d.mechanism;
      setMetric("m-display", idx.length.toLocaleString());
      setMetric("m-pool", selectedPool.toLocaleString());
      setMetric("m-all", d.all_lr_edges.toLocaleString());
      setMetric("m-annotated", DATA.annotated_cells.toLocaleString());
      setMetric("m-dist", dist.length?dist[Math.floor(dist.length/2)].toFixed(1):"NA");
      renderScale(d.scale);
    }
    function edgeIndicesForPair(applyTop=true){
      const e=lrData().edges; let idx=[];
      for(let i=0;i<e.score.length;i++){
        if(currentPair==="All interactions") idx.push(i);
        else { const p=currentPair.split(" -> "); if(e.sender_type[i]===p[0] && e.receiver_type[i]===p[1]) idx.push(i); }
      }
      idx.sort((a,b)=>e.score[b]-e.score[a]); if(!applyTop) return idx;
      return idx.slice(0, Math.min(idx.length, Math.max(20, Math.ceil(idx.length*currentTop/100))));
    }
    function setMetric(id,val){ document.getElementById(id).textContent=val; }
    function renderScale(vals){
      const box=document.getElementById("scale-box"); box.innerHTML=""; const max=Math.max(...vals,1e-9);
      vals.forEach((v,i)=>{ box.insertAdjacentHTML("beforeend", `<div class="scale-row"><span>S${i}</span><div><i style="width:${100*v/max}%"></i></div><b>${Number(v).toFixed(2)}</b></div>`); });
    }
    function updateDetail(d){
      selected=d; const kind=d?.[0]||"cell";
      if(kind==="edge") detail.innerHTML=`<div class="detail-title">${esc(d[1])} -> ${esc(d[3])}</div>`+row("sender",`${d[1]} (${d[2]})`)+row("receiver",`${d[3]} (${d[4]})`)+row("distance",d[5])+row("sender niche",d[6])+row("receiver niche",d[7]);
      else if(kind==="sender") detail.innerHTML=`<div class="detail-title">Sender: ${esc(d[1])}</div>`+row("cell type",d[2])+row("niche",d[3])+row("receiver",`${d[4]} (${d[5]})`)+row("distance",d[6]);
      else if(kind==="receiver") detail.innerHTML=`<div class="detail-title">Receiver: ${esc(d[1])}</div>`+row("cell type",d[2])+row("niche",d[3])+row("sender",`${d[4]} (${d[5]})`)+row("distance",d[6]);
      else detail.innerHTML=`<div class="detail-title">${esc(d[1]||"cell")}</div>`+row("cell type",d[2])+row("niche",d[3]);
      if(currentMode==="overview") setMode("focus"); else draw();
    }
    function setMode(mode){ currentMode=mode; modeButtons.forEach(b=>b.classList.toggle("active", b.dataset.mode===mode)); draw(); }
    function safeName(s){ return String(s).replace(/[^A-Za-z0-9]+/g,"_").replace(/^_+|_+$/g,""); }
    function exportPng(){
      Plotly.downloadImage(plot, {format:"png", width:1800, height:1000, scale:2, filename:`ScaleComm_${safeName(currentLR)}_${safeName(currentPair)}_top${currentTop}`});
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
    lrSelect.onchange=()=>{ currentLR=lrSelect.value; selected=null; currentPair="All interactions"; updatePairOptions(); resetDetail(); draw(); };
    pairSelect.onchange=()=>{ currentPair=pairSelect.value; selected=null; resetDetail(); draw(); };
    topRange.oninput=()=>{ currentTop=Number(topRange.value); topLabel.textContent=`top ${currentTop}%`; selected=null; resetDetail(); draw(); };
    modeButtons.forEach(b=>b.onclick=()=>setMode(b.dataset.mode));
    document.getElementById("reset-camera").onclick=()=>Plotly.relayout(plot, {"scene.camera":defaultCamera});
    document.getElementById("export-png").onclick=exportPng;
    document.getElementById("download-csv").onclick=downloadCsv;
    function resetDetail(){ detail.innerHTML='<div class="detail-title">No event selected</div><div class="detail-text">Click a sender, receiver, or line midpoint to inspect one ScaleComm communication event.</div>'; }
    updateControls();
    resetDetail();
    currentShowEdges = edgeIndicesForState();
    Plotly.newPlot(plot, baseTraces(currentShowEdges), layout(), {responsive:true, displayModeBar:false, displaylogo:false, scrollZoom:true}).then(()=>{
      updateInspector();
      plot.on("plotly_click", ev=>{ if(ev?.points?.length && ev.points[0].customdata) updateDetail(ev.points[0].customdata); });
    });
    </script>
    """
    body = """
    <div class="page">
      <div class="topbar">
        <div><div class="title">ScaleComm Explorer</div><div class="subtitle">Interactive single-cell communication browser for ScaleComm outputs.</div></div>
        <div class="badges"><span class="badge">ScaleComm only</span><span class="badge">LR filtering</span><span class="badge">cell-type pairs</span><span class="badge">clickable edges</span></div>
      </div>
      <div class="controls">
        <div class="control"><label>Dataset</label><div class="static-value">Slide-seqV2 hippocampus</div></div>
        <div class="control"><label>Ligand-receptor</label><select id="lr-select"></select></div>
        <div class="control"><label>Cell-type interaction</label><select id="pair-select"></select></div>
        <div class="control"><label>Edge threshold</label><div class="range-line"><input id="top-range" type="range" min="5" max="50" step="5" value="20"><span id="top-label">top 20%</span></div></div>
      </div>
      <p class="caption">Standalone explorer embeds top ScaleComm events for quick LR and cell-type-pair filtering; drag to rotate, scroll to zoom, and click events to inspect one communication edge.</p>
      <div id="shell">
        <div id="canvas">
          <div id="plot"></div>
          <div class="toolbar"><button class="mode active" data-mode="overview">Overview</button><button class="mode" data-mode="focus">Focus</button><button class="mode" data-mode="network">Network</button><button id="reset-camera" class="mode">Reset</button><button id="export-png" class="tool-action">Export PNG</button><button id="download-csv" class="tool-action">Download CSV</button></div>
          <div class="layer-key"><div><span class="dot sender-dot"></span>sender cells</div><div><span class="dot receiver-dot"></span>receiver cells</div><div><span class="line-key"></span>predicted communication</div></div>
          <div class="canvas-note">Drag to rotate · scroll to zoom · click an edge or endpoint</div>
        </div>
        <aside id="inspector">
          <div class="eyebrow">ScaleComm Explorer</div><div id="ins-lr" class="lr-title"></div><div id="ins-pair" class="pair-title"></div>
          <div class="pill-row"><span>Slide-seqV2</span><span id="pill-top">top 20%</span><span id="pill-mech"></span></div>
          <div class="metric-grid">
            <div class="metric"><span>displayed edges</span><b id="m-display"></b></div><div class="metric"><span>embedded pool</span><b id="m-pool"></b></div>
            <div class="metric"><span>all LR edges</span><b id="m-all"></b></div><div class="metric"><span>annotated beads</span><b id="m-annotated"></b></div>
            <div class="metric"><span>median distance</span><b id="m-dist"></b></div>
          </div>
          <div class="section"><div class="section-head">Selected event</div><div id="detail-box" class="detail-box"></div></div>
          <div class="section"><div class="section-head">Learned scale profile</div><div id="scale-box" class="scale-box"></div></div>
        </aside>
      </div>
    </div>
    """
    return (
        "<!doctype html><html><head><meta charset='utf-8'><title>ScaleComm Explorer</title>"
        + css
        + "</head><body>"
        + body
        + f"<script>{plotly_js}</script>"
        + js.replace("__PAYLOAD__", payload_json)
        + "</body></html>"
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "spatialself_explorer_static.html"
    out.write_text(build_html(), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
