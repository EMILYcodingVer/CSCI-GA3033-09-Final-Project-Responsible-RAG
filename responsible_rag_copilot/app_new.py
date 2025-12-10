"""
app_new.py

Premium UI + three modes: BASELINE | RAG | RESPONSIBLE

This version adds:
- Stable critic (see critic.py: temperature=0)
- Safe rewrite flow in RESPONSIBLE mode
- Frontend shows only:
    * Reliability snapshot
    * Retrieved docs
    * Safe final answer
    * Safety critic (after the final answer)
- Planner is optional and shown as a collapsible bubble (not forced)
- All modes keep only the latest run (previous output is cleared on each ASK)
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from openai import OpenAI
import os
import json

from rag import SimpleRAG
from planner import generate_plan
from critic import evaluate_answer
from config import OPENAI_API_KEY, CHAT_MODEL

app = FastAPI(title="Responsible RAG — Premium UI")

client = OpenAI(api_key=OPENAI_API_KEY)

rag = SimpleRAG(corpus_path="data")


# ----------------------------
# Helper functions
# ----------------------------
def get_doc_text(doc):
    """
    Safely extract text from a returned retrieval item.

    Accepts:
      - dict-like objects with "content" or "text"
      - tuples/lists where first element is text (e.g., (text, score, meta))
      - string objects
      - objects with attribute `page_content`
    """
    try:
        if isinstance(doc, dict):
            return doc.get("content") or doc.get("text") or str(doc)
    except Exception:
        pass

    if hasattr(doc, "page_content"):
        try:
            return doc.page_content
        except Exception:
            pass

    if isinstance(doc, (list, tuple)):
        if len(doc) > 0:
            return str(doc[0])
        return ""

    return str(doc)


def parse_critic_json(text: str):
    """
    Try to parse critic output as JSON.

    Returns a dict if possible, otherwise {}.
    """
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try to recover a JSON object inside the string
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            return json.loads(snippet)
        except Exception:
            return {}

    return {}


def compute_evidence_score(docs):
    """
    Compute a simple evidence score from retrieved documents based on similarity.

    Returns a float in [0,1] or None if unavailable.
    """
    sims = []
    if not docs:
        return None
    for d in docs:
        try:
            if isinstance(d, dict) and "similarity" in d:
                sims.append(float(d["similarity"]))
        except Exception:
            continue
    if not sims:
        return None
    max_sim = max(sims)
    return max(0.0, min(1.0, max_sim))


# ----------------------------
# Premium HTML UI
# ----------------------------
HTML_PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Responsible RAG — Copilot</title>
<style>
  :root{
    --bg-900:#07070b; --bg-800:#0f1116; --panel:#0f1724;
    --glass: rgba(255,255,255,0.04);
    --accent1:#6fb3ff; --accent2:#b58dff;
    --muted:#9aa6c4; --text:#e9f0ff;
    --card-shadow: 0 6px 30px rgba(2,6,23,0.6);
  }
  [data-theme="light"]{
    --bg-900:#f6f7fb; --bg-800:#eef1f7; --panel:#ffffff;
    --glass: rgba(3,8,18,0.04);
    --accent1:#1967d2; --accent2:#7b3cff;
    --muted:#54607a; --text:#071129;
    --card-shadow: 0 8px 30px rgba(10,20,40,0.06);
  }
  *{box-sizing:border-box}
  html,body{
    height:100%; margin:0;
    font-family:-apple-system,BlinkMacSystemFont,"SF Pro Text","SF Pro Display","Segoe UI",system-ui,Roboto,"Helvetica Neue",Arial,sans-serif;
    background: radial-gradient(1200px 600px at 10% 10%, rgba(18,24,40,0.5), transparent), var(--bg-900);
    color:var(--text);
  }
  header{
    display:flex;align-items:center;gap:20px;
    padding:18px 28px;
    border-bottom:1px solid rgba(255,255,255,0.02);
    position:sticky;top:0;
    backdrop-filter: blur(6px);
    background: linear-gradient(180deg, rgba(0,0,0,0.14), transparent);
    z-index:80;
  }
  .logo{display:flex;align-items:center;gap:12px;cursor:pointer;}
  .logo svg{height:36px;width:36px;filter:drop-shadow(0 6px 16px rgba(0,0,0,0.5));}
  .logo h2{
    margin:0;font-size:21px;letter-spacing:0.7px;
    color:var(--accent1); text-shadow:0 4px 28px rgba(103,141,255,0.16);
  }
  nav.topnav{margin-left:auto;display:flex;gap:18px;align-items:center;}
  nav.topnav a{
    color:var(--muted);text-decoration:none;font-weight:600;
    font-size:14px;letter-spacing:0.06em;text-transform:uppercase;
  }
  nav.topnav a:hover{color:var(--text);}
  .layout{
    display:grid;grid-template-columns: 250px 1fr;
    gap:30px;padding:26px 28px 26px 28px;align-items:start;
  }
  aside.sidebar{
    background: linear-gradient(180deg, rgba(255,255,255,0.02), transparent);
    border-radius:14px;padding:18px;
    height:calc(100vh - 120px);position:sticky;top:76px;
    box-shadow:var(--card-shadow);
    border:1px solid rgba(255,255,255,0.04);
  }
  .sb-label{
    margin-bottom:10px;font-weight:650;
    font-size:13px;color:var(--muted);letter-spacing:0.18em;text-transform:uppercase;
  }
  .sb-item{
    display:flex;align-items:center;gap:12px;
    padding:10px 11px;border-radius:12px;
    color:var(--muted);cursor:pointer;margin-bottom:8px;
    font-size:14px;font-weight:540;
    transition: all .18s ease-out;
  }
  .sb-item.active{
    background: linear-gradient(90deg, rgba(107,150,255,0.12), rgba(181,141,255,0.06));
    color:var(--accent1);
    font-weight:650;
    box-shadow: 0 8px 26px rgba(40,60,120,0.25);
    transform:translateY(-1px);
  }
  .sb-item svg{opacity:0.9}

  .sidebar-copy{
    font-size:14px;
    color:var(--muted);
    line-height:1.7;
  }
  .sidebar-copy strong{
    display:block;
    font-size:13.5px;
    text-transform:uppercase;
    letter-spacing:0.12em;
    margin-bottom:6px;
  }
  .sidebar-chip{
    display:inline-flex;
    align-items:center;
    gap:6px;
    padding:4px 8px;
    border-radius:999px;
    border:1px solid rgba(148,163,184,0.45);
    font-size:11px;
    text-transform:uppercase;
    letter-spacing:0.12em;
    margin-top:8px;
  }

  .card{
    background: linear-gradient(180deg, rgba(255,255,255,0.018), rgba(15,23,42,0.86));
    border-radius:16px;padding:20px 20px 18px 20px;
    box-shadow:var(--card-shadow);
    border:1px solid rgba(255,255,255,0.04);
  }
  .hero-label{
    font-size:13px;
    color:var(--muted);
    letter-spacing:0.14em;
    text-transform:uppercase;
    margin-bottom:10px;
  }
  .hero-shell{
    position:relative;
    border-radius:18px;
    padding:18px 18px 16px 18px;
    background:radial-gradient(circle at 0% 0%, rgba(111,179,255,0.18), transparent 55%), rgba(15,23,42,0.9);
    border:1px solid rgba(148,163,184,0.50);
    box-shadow:0 18px 55px rgba(15,23,42,0.85);
  }
  .hero-shell:before{
    content:"";
    position:absolute;inset:-1px;
    border-radius:inherit;
    background: radial-gradient(circle at 0% 0%, rgba(111,179,255,0.28), transparent 60%);
    opacity:0.35;pointer-events:none;
  }
  .hero-shell-inner{position:relative;z-index:1;}

  .hero-input-row{
    display:flex;gap:14px;align-items:flex-start;
  }
  .hero-bubble-label{
    font-size:11px;
    text-transform:uppercase;
    letter-spacing:0.18em;
    color:var(--muted);
    margin-bottom:6px;
  }
  .hero-input-bubble{
    flex:1;
    border-radius:16px;
    padding:14px 14px 12px 14px;
    background:rgba(15,23,42,0.95);
    border:1px solid rgba(148,163,184,0.65);
    box-shadow:0 14px 38px rgba(15,23,42,0.9);
  }
  .hero-input{
    width:100%;
    border:none;
    outline:none;
    resize:vertical;
    background:transparent;
    color:var(--text);
    font-family:-apple-system,BlinkMacSystemFont,"SF Mono","Menlo",monospace;
    font-size:15px;
    line-height:1.6;
  }
  .hero-input::placeholder{
    color:rgba(148,163,184,0.7);
  }
  .hero-foot{
    display:flex;
    gap:10px;
    align-items:center;
    margin-top:14px;
  }
  .hero-foot small{
    font-size:11px;
    color:var(--muted);
  }
  .controls{
    display:flex;gap:12px;margin-top:4px;
  }
  .btn{
    padding:10px 18px;border-radius:999px;border:none;
    cursor:pointer;font-weight:600;
    background: linear-gradient(180deg, rgba(20,30,60,0.9), rgba(8,12,24,0.95));
    color:var(--text);
    box-shadow: 0 10px 32px rgba(8,10,20,0.7);
    transition: transform .16s ease, box-shadow .16s ease, background .16s;
    font-size:13px;
  }
  .btn.accent{
    background: linear-gradient(90deg, var(--accent1), var(--accent2));
    color:#fff;
    box-shadow: 0 12px 34px rgba(99,102,241,0.45);
  }
  .btn:hover{
    transform:translateY(-2px);
    box-shadow:0 16px 40px rgba(15,23,42,0.9);
  }
  .btn.ghost{
    background: transparent;
    border:1px solid rgba(148,163,184,0.55);
    color:var(--muted);
  }
  .btn:disabled{
    opacity:0.6;
    cursor:default;
    box-shadow:none;
    transform:none;
  }
  .mode-pill{
    padding:6px 12px;
    border-radius:999px;
    background:rgba(15,23,42,0.9);
    border:1px solid rgba(148,163,184,0.55);
    color:var(--muted);
    font-size:11px;
    letter-spacing:0.16em;
    text-transform:uppercase;
  }

  .out-grid{
    display:grid;
    grid-template-columns:1fr;
    gap:16px;
    margin-top:18px;
  }
  .answer-shell{
    position:relative;
    border-radius:16px;
    padding:18px 18px 16px 18px;
    background:rgba(15,23,42,0.98);
    border:1px solid rgba(148,163,184,0.75);
    overflow:hidden;
  }
  .answer-pulse{
    position:absolute;inset:-1px;border-radius:inherit;
    pointer-events:none;
    background:radial-gradient(circle at 0% 0%, rgba(111,179,255,0.35), transparent 60%);
    opacity:0;animation:answerGlow 2.8s ease-out 1;
  }
  .answer{
    white-space:pre-wrap;
    font-size:15px;
    color:var(--text);
    line-height:1.65;
    position:relative;z-index:1;
  }
  @keyframes answerGlow{
    0%{border-color:rgba(111,179,255,0.70); box-shadow:0 0 0 0 rgba(111,179,255,0.65);}
    80%{border-color:rgba(111,179,255,0.10); box-shadow:0 0 0 12px rgba(111,179,255,0);}
    100%{border-color:rgba(111,179,255,0.0); box-shadow:none;}
  }

  .critic{
    background: linear-gradient(180deg, rgba(255,59,48,0.10), rgba(255,59,48,0.03));
    border-left:4px solid #ff6161;
    padding:14px 16px;
    border-radius:12px;
    color:#ffdfe0;
    font-size:13.5px;
  }

  .risk-chip{
    padding:4px 9px;
    border-radius:999px;
    font-size:11px;
    font-weight:600;
    letter-spacing:0.08em;
    text-transform:uppercase;
  }
  .risk-low{
    background:rgba(22,163,74,0.18);
    color:#4ade80;
  }
  .risk-med{
    background:rgba(234,179,8,0.18);
    color:#facc15;
  }
  .risk-high{
    background:rgba(248,113,113,0.18);
    color:#fecaca;
  }
  .risk-unk{
    background:rgba(148,163,184,0.14);
    color:#e5e7eb;
  }

  .retrieved-item{
    margin-top:8px;
    padding:8px 9px;
    border-radius:10px;
    background:rgba(15,23,42,0.85);
    border:1px solid rgba(148,163,184,0.25);
  }
  .retrieved-header{
    display:flex;
    gap:8px;
    align-items:center;
    margin-bottom:4px;
    font-size:11px;
    color:var(--muted);
  }
  .retrieved-index{
    font-weight:600;
    color:var(--text);
  }
  .retrieved-source{
    padding:2px 6px;
    border-radius:999px;
    background:rgba(15,23,42,0.9);
  }
  .retrieved-score{
    margin-left:auto;
    font-variant-numeric:tabular-nums;
  }
  .retrieved-text{
    font-size:13px;
    color:var(--muted);
  }

  .meta-label{
    font-size:12px;
    letter-spacing:0.18em;
    text-transform:uppercase;
    color:var(--muted);
    margin-bottom:10px;
  }
  .section-title{
    font-size:17px;
    margin:0 0 6px 0;
  }

  .plan-toggle-chip{
    display:inline-flex;
    align-items:center;
    gap:6px;
    padding:6px 10px;
    border-radius:999px;
    border:1px solid rgba(148,163,184,0.55);
    background:rgba(15,23,42,0.9);
    font-size:11px;
    text-transform:uppercase;
    letter-spacing:0.12em;
    cursor:pointer;
  }

  footer.small{
    text-align:center;
    color:var(--muted);
    padding:22px;
    font-size:12px;
  }

  @media (max-width:900px){
    .layout{grid-template-columns:1fr;padding:16px;}
    aside.sidebar{display:none;}
  }
</style>
</head>
<body>
<header>
  <div class="logo" onclick="window.scrollTo({top:0, behavior:'smooth'})">
    <svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden>
      <defs>
        <linearGradient id="g1" x1="0" x2="1">
          <stop offset="0" stop-color="#6fb3ff"/>
          <stop offset="1" stop-color="#b58dff"/>
        </linearGradient>
      </defs>
      <rect x="4" y="4" width="56" height="56" rx="14" fill="url(#g1)" opacity="0.16"/>
      <g transform="translate(12,12)">
        <path d="M20 2 L26 10 L18 10 Z" fill="#b8d8ff" opacity="0.95"/>
        <circle cx="14" cy="20" r="8" fill="#b58dff" opacity="0.9"/>
        <path d="M6 18 L14 6 L22 18" stroke="#eaf2ff" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
      </g>
    </svg>
    <h2>Copilot • Responsible RAG</h2>
  </div>

  <nav class="topnav" role="navigation" aria-label="main">
    <a href="#" onclick="document.getElementById('query').focus(); return false">ASK</a>
    <a href="#how">HOW IT WORKS</a>
    <a href="#about">ABOUT</a>
    <button id="themeToggle" class="btn ghost" title="Toggle dark / light">LIGHT</button>
  </nav>
</header>

<main class="layout">
  <aside class="sidebar">
    <div class="sb-label">MODES</div>
    <div class="sb-item active" data-mode="baseline" onclick="selectMode(this)">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M3 12h18" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>
      BASELINE
    </div>
    <div class="sb-item" data-mode="rag" onclick="selectMode(this)">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M3 6h18M3 12h18M3 18h18" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>
      RAG
    </div>
    <div class="sb-item" data-mode="responsible" onclick="selectMode(this)">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
        <path d="M12 2v20" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>
        <circle cx="12" cy="8" r="2" fill="currentColor"/>
      </svg>
      RESPONSIBLE
    </div>

    <hr style="margin:16px 0;border:none;border-top:1px solid rgba(148,163,184,0.25)"/>

    <div class="sidebar-copy">
      <strong>HUMAN-CENTERED</strong>
      See which sources were used and how confident the system is before you trust an answer.
      <span class="sidebar-chip">SOURCE-AWARE</span>
    </div>
    <div style="height:10px;"></div>
    <div class="sidebar-copy">
      <strong>SAFETY LAYER</strong>
      Responsible mode adds a planner, a safety critic, and a safe rewrite so the final answer is
      explicitly checked for grounding and risk.
      <span class="sidebar-chip">SAFETY-FIRST</span>
    </div>
  </aside>

  <section>
    <div class="card">
      <div class="hero-label">ASK ANYTHING — WE CHECK SOURCES FOR YOU</div>
      <div class="hero-shell">
        <div class="hero-shell-inner">
          <div class="hero-input-row">
            <div class="hero-input-bubble">
              <div class="hero-bubble-label">QUESTION</div>
              <textarea id="query" class="hero-input" rows="4"
                placeholder="Ask about a company, a claim, or a document — we will show you what it is based on.">Is SpaceX a reliable company?</textarea>
            </div>
          </div>

          <div class="hero-foot">
            <div class="controls">
              <button class="btn accent" id="runBtn" onclick="runCurrentMode()">ASK</button>
              <button class="btn ghost" onclick="clearOutput()">CLEAR</button>
            </div>
            <div style="margin-left:auto;display:flex;gap:8px;align-items:center;">
              <small>MODE</small>
              <div id="modeBadge" class="mode-pill">BASELINE</div>
            </div>
          </div>
        </div>
      </div>

      <div class="out-grid" id="outGrid"></div>
    </div>

    <div class="card" id="how" style="margin-top:18px;">
      <h3 class="section-title">How it works</h3>
      <p style="margin:0;font-size:14px;color:var(--muted);line-height:1.7;">
        <strong>BASELINE</strong> uses the model alone — fast, but not explicitly grounded.<br/>
        <strong>RAG</strong> retrieves top-matching passages from your corpus and answers strictly from them.<br/>
        <strong>RESPONSIBLE</strong> adds a planner, a safety critic, and a safe rewrite on top of RAG to highlight
        uncertainty, missing evidence, and potential risks before you see the final answer.
      </p>
    </div>

    <div class="card" id="about" style="margin-top:14px;">
      <h3 class="section-title">About this copilot</h3>
      <p style="margin:0;font-size:14px;color:var(--muted);line-height:1.7;">
        This interface is a research prototype for <em>trustworthy RAG</em>. Instead of only
        showing a final answer, it exposes sources, a reliability snapshot, an optional reasoning
        plan, a safety critic, and a safe final rewrite. The goal is to help you quickly decide:
        which parts are strongly supported, where the model is extrapolating, and whether there are
        safety or fairness concerns you should be aware of.
      </p>
    </div>
  </section>
</main>

<footer class="small">Built for research • Responsible RAG Copilot</footer>

<script>
  // -------------------------
  // Theme handling
  // -------------------------
  const themeToggle = document.getElementById('themeToggle');
  const runBtn = document.getElementById('runBtn');

  function applyTheme(t){
    if(t==='light'){
      document.documentElement.setAttribute('data-theme','light');
      localStorage.setItem('theme','light');
      themeToggle.innerText='DARK';
    } else {
      document.documentElement.removeAttribute('data-theme');
      localStorage.setItem('theme','dark');
      themeToggle.innerText='LIGHT';
    }
  }
  (function(){
    const t = localStorage.getItem('theme') || 'dark';
    applyTheme(t);
  })();
  themeToggle.onclick = () => {
    const cur = localStorage.getItem('theme') || 'dark';
    applyTheme(cur==='dark' ? 'light' : 'dark');
  };

  // -------------------------
  // Mode selection
  // -------------------------
  let currentMode = 'baseline';
  let isRunning = false;

  function selectMode(el){
    document.querySelectorAll('.sb-item').forEach(x => x.classList.remove('active'));
    el.classList.add('active');
    currentMode = el.dataset.mode;
    document.getElementById('modeBadge').innerText = currentMode.toUpperCase();
  }

  // -------------------------
  // Typing animation for final answer
  // -------------------------
  function typeWriter(el, text, speed=18){
    el.innerText = '';
    let i = 0;
    return new Promise(resolve => {
      function step(){
        if(i < text.length){
          el.innerText += text[i++];
          setTimeout(step, speed + Math.random()*12);
        } else {
          resolve();
        }
      }
      step();
    });
  }

  // -------------------------
  // Helpers for reliability snapshot
  // -------------------------
  function formatPercent(score){
    if(score === null || score === undefined || isNaN(score)) return '—';
    return Math.round(score * 100) + '%';
  }
  function riskChip(risk){
    const r = (risk || 'unknown').toLowerCase();
    if(r === 'low')   return '<span class="risk-chip risk-low">LOW RISK</span>';
    if(r === 'medium')return '<span class="risk-chip risk-med">MEDIUM RISK</span>';
    if(r === 'high')  return '<span class="risk-chip risk-high">HIGH RISK</span>';
    return '<span class="risk-chip risk-unk">RISK UNKNOWN</span>';
  }

  function buildMetaHTML(meta){
    const mode = (meta.mode || '').toString().toUpperCase() || '—';
    const evidenceScore = typeof meta.evidence_score === 'number' ? meta.evidence_score : null;
    const overall  = typeof meta.overall_score === 'number' ? meta.overall_score : null;
    const grounding= typeof meta.grounding_score === 'number' ? meta.grounding_score : null;
    const safety   = typeof meta.safety_score === 'number' ? meta.safety_score : null;
    const risk     = meta.hallucination_risk || null;
    const numSources = (meta.num_sources !== undefined && meta.num_sources !== null)
      ? meta.num_sources
      : null;

    let summaryLine = '';
    if(meta.critic_summary){
      summaryLine = escapeHTML(meta.critic_summary);
    } else if(mode === 'BASELINE'){
      summaryLine = 'No retrieval or safety pass — treat this as a normal LLM answer.';
    } else if(mode === 'RAG'){
      summaryLine = 'Answer grounded on retrieved documents; no safety critic was run.';
    } else {
      summaryLine = 'Planner, safety critic, and safe rewrite were applied to this answer.';
    }

    let evidenceBadge = '';
    if(evidenceScore !== null){
      evidenceBadge = `<div><strong>${formatPercent(evidenceScore)}</strong><br/>Evidence</div>`;
    }

    return `
      <div class="meta-label">Reliability Snapshot</div>
      <div style="display:flex; flex-wrap:wrap; gap:12px; align-items:flex-start; margin-bottom:10px;">
        <div style="font-size:12px; text-transform:uppercase; letter-spacing:0.14em; color:var(--muted);">MODE</div>
        <div style="padding:4px 10px; border-radius:999px; background:rgba(15,23,42,0.9); font-size:12px; font-weight:600;">${mode}</div>
        ${riskChip(risk)}
      </div>
      <div style="display:flex; flex-wrap:wrap; gap:18px; font-size:13px; color:var(--muted); margin-bottom:6px;">
        <div><strong>${formatPercent(overall)}</strong><br/>Overall</div>
        <div><strong>${formatPercent(grounding)}</strong><br/>Grounding</div>
        <div><strong>${formatPercent(safety)}</strong><br/>Safety</div>
        ${evidenceBadge}
        <div><strong>${numSources !== null ? numSources : '—'}</strong><br/>Sources</div>
      </div>
      <p style="font-size:13.5px; color:var(--muted); margin-top:8px;">${summaryLine}</p>
    `;
  }

  // -------------------------
  // Normalize retrieved docs
  // -------------------------
  function normalizeRetrieved(docs){
    try {
      if(!docs) return [];
      return docs.map(item => {
        if(Array.isArray(item) && item.length>0){
          return { text: String(item[0]), score: item[1] ?? null, source: null };
        }
        if(typeof item === 'object' && item !== null){
          const text = item.text || item.content || item.page_content || JSON.stringify(item);
          const score = (item.similarity !== undefined && item.similarity !== null)
            ? item.similarity
            : (item.score ?? null);
          const source = item.source ?? null;
          return { text: String(text), score: score, source: source };
        }
        return { text: String(item), score: null, source: null };
      });
    } catch(e){
      return [];
    }
  }

  // -------------------------
  // Main API call flow
  // -------------------------
  async function runCurrentMode(){
    const q = document.getElementById('query').value.trim();
    if(!q) return alert('Please enter a question.');
    if(isRunning) return; // prevent double-click while running

    isRunning = true;
    runBtn.disabled = true;

    const outGrid = document.getElementById('outGrid');
    clearOutput();  // keep only the latest run for all modes

    const processingCard = document.createElement('div');
    processingCard.className = 'card';
    processingCard.innerHTML = '<div style="color:var(--muted); font-size:14px;">Processing your request…</div>';
    outGrid.appendChild(processingCard);

    try {
      const res = await fetch('/api/run', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ query: q, mode: currentMode })
      });
      const payload = await res.json();
      processingCard.remove();

      // Reliability snapshot
      if(payload.meta){
        const metaCard = document.createElement('div');
        metaCard.className = 'card';
        metaCard.innerHTML = buildMetaHTML(payload.meta);
        outGrid.appendChild(metaCard);
      }

      // Retrieved docs
      if(payload.retrieved){
        const retrieved = normalizeRetrieved(payload.retrieved);
        const docCard = document.createElement('div');
        docCard.className = 'card';
        let html = '<div class="meta-label">Retrieved Docs</div>';
        retrieved.forEach((d,i) => {
          html += `
            <div class="retrieved-item">
              <div class="retrieved-header">
                <span class="retrieved-index">[${i}]</span>
                ${d.source ? `<span class="retrieved-source">${escapeHTML(d.source)}</span>` : ''}
                ${d.score !== null && d.score !== undefined ? `<span class="retrieved-score">${(d.score * 100).toFixed(1)}%</span>` : ''}
              </div>
              <div class="retrieved-text">${escapeHTML(d.text)}</div>
            </div>
          `;
        });
        docCard.innerHTML = html;
        outGrid.appendChild(docCard);
      }

      // Optional planner: collapsible bubble (only when available)
      let planCard = null;
      if(payload.plan){
        planCard = document.createElement('div');
        planCard.className = 'card';
        planCard.style.display = 'none';
        planCard.innerHTML = `<div class="meta-label">Planner</div>
          <pre style="margin:0;font-size:13.5px;color:var(--muted);white-space:pre-wrap;">${escapeHTML(payload.plan)}</pre>`;
        outGrid.appendChild(planCard);

        const toggleCard = document.createElement('div');
        toggleCard.className = 'card';
        toggleCard.style.padding = '10px 14px';
        toggleCard.style.marginTop = '0px';
        toggleCard.innerHTML = `
          <div class="plan-toggle-chip" id="planToggleChip">
            <span>Show reasoning plan</span>
          </div>
        `;
        const chip = toggleCard.querySelector('#planToggleChip');
        chip.onclick = () => {
          const visible = planCard.style.display !== 'none';
          planCard.style.display = visible ? 'none' : 'block';
          chip.querySelector('span').textContent = visible ? 'Show reasoning plan' : 'Hide reasoning plan';
        };
        // Insert the toggle just before the hidden plan card
        outGrid.insertBefore(toggleCard, planCard);
      }

      // Safe final answer (or fallback to "answer")
      const finalText = payload.safe_answer || payload.answer;
      if(finalText){
        const ansCard = document.createElement('div');
        ansCard.className = 'card';
        ansCard.innerHTML = `<div class="section-title" style="font-size:16px;margin-bottom:10px;">Safe Final Answer</div>`;
        const shell = document.createElement('div');
        shell.className = 'answer-shell';
        const pulse = document.createElement('div');
        pulse.className = 'answer-pulse';
        const ansBody = document.createElement('div');
        ansBody.className = 'answer';
        shell.appendChild(pulse);
        shell.appendChild(ansBody);
        ansCard.appendChild(shell);
        outGrid.appendChild(ansCard);
        await typeWriter(ansBody, finalText);
      }

      // Safety critic (shown after the final answer)
      if(payload.critic){
        const criticCard = document.createElement('div');
        criticCard.className = 'card';
        criticCard.innerHTML = `<div class="meta-label">Safety Critic</div>`;
        const c = payload.critic;
        if(typeof c === 'string'){
          criticCard.innerHTML += `<div class="critic"><pre style="white-space:pre-wrap;">${escapeHTML(c)}</pre></div>`;
        } else {
          const issues = Array.isArray(c.issues) ? c.issues : [];
          const suggestions = Array.isArray(c.suggestions) ? c.suggestions : [];
          let html = '<div class="critic">';
          if(c.summary){
            html += `<p style="margin-top:0;margin-bottom:8px;">${escapeHTML(c.summary)}</p>`;
          }
          if(issues.length){
            html += '<p style="margin:6px 0 2px 0; font-weight:600;">Issues</p><ul style="margin:0 0 4px 18px; padding:0;">';
            issues.forEach(item => { html += `<li>${escapeHTML(item)}</li>`; });
            html += '</ul>';
          }
          if(suggestions.length){
            html += '<p style="margin:6px 0 2px 0; font-weight:600;">Suggestions</p><ul style="margin:0 0 0 18px; padding:0;">';
            suggestions.forEach(item => { html += `<li>${escapeHTML(item)}</li>`; });
            html += '</ul>';
          }
          html += '</div>';
          criticCard.innerHTML += html;
        }
        outGrid.appendChild(criticCard);
      }

    } catch(err){
      processingCard.remove();
      const errCard = document.createElement('div');
      errCard.className = 'card';
      errCard.innerHTML = '<div style="color:#ffb4b4">Request failed — open the browser console for details.</div>';
      outGrid.appendChild(errCard);
      console.error(err);
    } finally {
      isRunning = false;
      runBtn.disabled = false;
    }
  }

  function clearOutput(){
    const outGrid = document.getElementById('outGrid');
    outGrid.innerHTML = '';
  }

  function escapeHTML(s){
    return String(s || '').replace(/[&<>"']/g, c => (
      {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]
    ));
  }
</script>
</body>
</html>
"""


# ----------------------------
# Routes
# ----------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_PAGE


@app.post("/api/run")
async def run_query(request: Request):
    """
    POST body: { "query": "<text>", "mode": "baseline"|"rag"|"responsible" }
    Returns JSON:
      - meta: dict with reliability info
      - retrieved: list of docs
      - plan: str (optional, only in responsible)
      - critic: dict/str (optional, only in responsible)
      - safe_answer: str (optional, only in responsible)
      - answer: str (baseline / rag answer or safe_answer)
    """
    data = await request.json()
    query = data.get("query", "")
    mode = data.get("mode", "baseline")

    if not query:
        return JSONResponse({"error": "Empty query"}, status_code=400)

    # -------------------------
    # 1) Baseline: direct model call
    # -------------------------
    if mode == "baseline":
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query},
            ],
        )
        answer = resp.choices[0].message.content
        meta = {"mode": "baseline"}
        return JSONResponse({"answer": answer, "meta": meta, "retrieved": []})

    # -------------------------
    # 2) RAG: retrieve -> context -> model (answer using docs)
    # -------------------------
    if mode == "rag":
        docs = rag.retrieve(query)
        texts = [get_doc_text(d) for d in docs]
        context = "\n\n".join(texts)

        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Answer ONLY using the retrieved documents. If unsure, say you don't know.",
                },
                {
                    "role": "user",
                    "content": context + "\n\nQuestion: " + query,
                },
            ],
        )
        answer = resp.choices[0].message.content
        meta = {
            "mode": "rag",
            "evidence_score": compute_evidence_score(docs),
            "num_sources": len(texts),
        }
        return JSONResponse({"retrieved": docs, "answer": answer, "meta": meta})

    # -------------------------
    # 3) Responsible: planner + draft + critic + safe rewrite
    # -------------------------
    if mode == "responsible":
        docs = rag.retrieve(query)
        texts = [get_doc_text(d) for d in docs]

        # Planner: high-level reasoning steps
        plan = generate_plan(query, texts)

        # Draft answer using plan and retrieved context
        draft_resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Follow the plan and use the retrieved context "
                        "when relevant. Do not invent facts beyond the context."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Plan:\n"
                        + plan
                        + "\n\nUser question:\n"
                        + query
                        + "\n\nRetrieved context:\n"
                        + "\n\n".join(texts)
                    ),
                },
            ],
        )
        draft_text = draft_resp.choices[0].message.content

        # Safety critic
        critic_text = evaluate_answer(query, texts, draft_text)
        critic_parsed = parse_critic_json(critic_text)

        # Safe rewrite based on critic feedback
        critic_json_str = json.dumps(critic_parsed, ensure_ascii=False) if critic_parsed else critic_text
        safe_resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful assistant. Rewrite the answer to be safe, factual, and grounded "
                        "ONLY in the retrieved documents. Use the critic feedback to fix issues, remove or "
                        "explicitly flag speculative claims, and clearly acknowledge uncertainties. Preserve "
                        "useful content when it is supported by the documents. Never add new unsupported facts."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "User question:\n"
                        + query
                        + "\n\nRetrieved documents:\n"
                        + "\n\n".join(texts)
                        + "\n\nOriginal draft answer:\n"
                        + draft_text
                        + "\n\nCritic feedback (JSON or text):\n"
                        + critic_json_str
                    ),
                },
            ],
        )
        safe_answer = safe_resp.choices[0].message.content

        # Build meta for reliability snapshot
        meta = {
            "mode": "responsible",
            "evidence_score": compute_evidence_score(docs),
            "num_sources": len(texts),
        }
        if critic_parsed:
            meta.update(
                {
                    "overall_score": critic_parsed.get("overall_score"),
                    "grounding_score": critic_parsed.get("grounding_score"),
                    "safety_score": critic_parsed.get("safety_score"),
                    "hallucination_risk": critic_parsed.get("hallucination_risk"),
                    "issues": critic_parsed.get("issues"),
                    "suggestions": critic_parsed.get("suggestions"),
                    "critic_summary": critic_parsed.get("summary"),
                }
            )

        return JSONResponse(
            {
                "retrieved": docs,
                "plan": plan,
                "critic": critic_parsed or critic_text,
                "critic_raw": critic_text,
                "safe_answer": safe_answer,
                "answer": safe_answer,  # for compatibility with front-end
                "meta": meta,
            }
        )

    # Invalid mode
    return JSONResponse({"error": "Invalid mode specified"}, status_code=400)
