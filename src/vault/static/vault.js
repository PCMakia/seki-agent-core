const $ = (id) => document.getElementById(id);

const state = {
  nodes: [],
  edges: [],
  pos: new Map(),
  vel: new Map(),
  selected: null,
  drag: null,
  running: false,
};

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(opts.headers || {}) },
    ...opts,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || JSON.stringify(body);
    } catch (_err) {
      /* ignore */
    }
    throw new Error(detail);
  }
  if (res.status === 204) return null;
  return res.json();
}

function hours() {
  const n = Number($("hours").value);
  return Number.isFinite(n) ? n : 24;
}

function sessionId() {
  return ($("session").value || "default").trim() || "default";
}

function maxAct() {
  return Math.max(0.01, ...state.nodes.map((n) => n.activation_weight || 0));
}

function seedLayout(canvas) {
  const w = canvas.clientWidth || 640;
  const h = canvas.clientHeight || 480;
  state.pos.clear();
  state.vel.clear();
  state.nodes.forEach((n, i) => {
    const ang = (i / Math.max(1, state.nodes.length)) * Math.PI * 2;
    const r = 80 + (i % 7) * 18;
    state.pos.set(n.id, { x: w / 2 + Math.cos(ang) * r, y: h / 2 + Math.sin(ang) * r });
    state.vel.set(n.id, { x: 0, y: 0 });
  });
}

function stepForces(canvas) {
  const w = canvas.clientWidth || 640;
  const h = canvas.clientHeight || 480;
  const ids = state.nodes.map((n) => n.id);
  const kRep = 2200;
  const kSpring = 0.012;
  const rest = 110;

  for (const a of ids) {
    const pa = state.pos.get(a);
    const va = state.vel.get(a);
    va.x += (w / 2 - pa.x) * 0.004;
    va.y += (h / 2 - pa.y) * 0.004;
    for (const b of ids) {
      if (a >= b) continue;
      const pb = state.pos.get(b);
      const vb = state.vel.get(b);
      let dx = pa.x - pb.x;
      let dy = pa.y - pb.y;
      let dist = Math.hypot(dx, dy) || 0.01;
      const f = kRep / (dist * dist);
      dx /= dist;
      dy /= dist;
      va.x += dx * f;
      va.y += dy * f;
      vb.x -= dx * f;
      vb.y -= dy * f;
    }
  }
  for (const e of state.edges) {
    const pa = state.pos.get(e.src_id);
    const pb = state.pos.get(e.dst_id);
    const va = state.vel.get(e.src_id);
    const vb = state.vel.get(e.dst_id);
    if (!pa || !pb || !va || !vb) continue;
    const dx = pb.x - pa.x;
    const dy = pb.y - pa.y;
    const dist = Math.hypot(dx, dy) || 0.01;
    const t = (dist - rest) * kSpring * Math.min(2, e.weight || 1);
    const fx = (dx / dist) * t;
    const fy = (dy / dist) * t;
    va.x += fx;
    va.y += fy;
    vb.x -= fx;
    vb.y -= fy;
  }
  for (const id of ids) {
    if (state.drag && state.drag === id) {
      state.vel.get(id).x = 0;
      state.vel.get(id).y = 0;
      continue;
    }
    const p = state.pos.get(id);
    const v = state.vel.get(id);
    v.x *= 0.82;
    v.y *= 0.82;
    p.x = Math.min(w - 24, Math.max(24, p.x + v.x));
    p.y = Math.min(h - 24, Math.max(24, p.y + v.y));
  }
}

function draw(canvas) {
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
    canvas.width = w * dpr;
    canvas.height = h * dpr;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const peak = maxAct();

  ctx.lineWidth = 1;
  for (const e of state.edges) {
    const a = state.pos.get(e.src_id);
    const b = state.pos.get(e.dst_id);
    if (!a || !b) continue;
    ctx.strokeStyle = e.id && state.selectedEdge === e.id ? "#f59e0b" : "rgba(167,139,250,0.35)";
    ctx.lineWidth = Math.max(1, Math.min(4, (e.weight || 1) * 0.35));
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }

  for (const n of state.nodes) {
    const p = state.pos.get(n.id);
    if (!p) continue;
    const r = 7 + 10 * ((n.activation_weight || 0) / peak);
    const hot = (n.activation_weight || 0) / peak > 0.55;
    ctx.beginPath();
    ctx.fillStyle = n.type === "base" ? "#38bdf8" : hot ? "#f59e0b" : "#a78bfa";
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    ctx.fill();
    if (state.selected === n.id) {
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 2;
      ctx.stroke();
    }
    ctx.fillStyle = "#dcdcdc";
    ctx.font = "12px Segoe UI, sans-serif";
    ctx.fillText(n.name, p.x + r + 4, p.y + 4);
  }
}

function hitTest(x, y) {
  let best = null;
  let bestD = 16;
  for (const n of state.nodes) {
    const p = state.pos.get(n.id);
    if (!p) continue;
    const d = Math.hypot(p.x - x, p.y - y);
    if (d < bestD) {
      best = n.id;
      bestD = d;
    }
  }
  return best;
}

function loop(canvas) {
  if (!state.running) return;
  stepForces(canvas);
  draw(canvas);
  requestAnimationFrame(() => loop(canvas));
}

function renderList(el, items, kind) {
  el.innerHTML = "";
  if (!items.length) {
    el.innerHTML = '<li class="muted">None in this window</li>';
    return;
  }
  for (const item of items) {
    const li = document.createElement("li");
    if (kind === "node") {
      li.textContent = item.name;
      const meta = document.createElement("div");
      meta.className = "meta";
      meta.textContent = `${item.type} · act ${item.activation_weight.toFixed(2)}`;
      li.appendChild(meta);
      li.onclick = () => selectNode(item.id, true);
    } else {
      li.innerHTML = `<span>${item.src_name} → ${item.dst_name}</span>`;
      const meta = document.createElement("div");
      meta.className = "meta";
      meta.textContent = `${item.relation_type} · w ${item.weight.toFixed(2)}`;
      li.appendChild(meta);
      li.onclick = () => selectNode(item.src_id, true);
    }
    el.appendChild(li);
  }
}

async function selectNode(id, recenter) {
  state.selected = id;
  try {
    const [node, neighbors, episodes] = await Promise.all([
      api(`/agent/vault/nodes/${id}`),
      api(`/agent/vault/nodes/${id}/neighbors`),
      api(`/agent/vault/nodes/${id}/episodes`),
    ]);
    renderInspector(node, neighbors.items || [], episodes.items || []);
    if (recenter) loadGraph({ centerId: id });
  } catch (err) {
    $("inspector").innerHTML = `<p class="muted">${err.message}</p>`;
  }
}

function renderInspector(node, neighbors, episodes) {
  const pane = $("inspector");
  pane.innerHTML = `
    <h2>Note</h2>
    <h3>${escapeHtml(node.name)}</h3>
    <p class="muted">${escapeHtml(node.type)} · degree ${node.degree ?? "?"} · act ${node.activation_weight.toFixed(2)}</p>
    <p class="muted">${node.last_activation_ts ? escapeHtml(node.last_activation_ts) : "never activated"}</p>
    <textarea id="summary" class="summary">${escapeHtml(node.summary)}</textarea>
    <div class="actions">
      <button type="button" id="save-summary">Save summary</button>
    </div>
    <h2>Backlinks</h2>
    <div id="backlinks"></div>
    <h2>New link</h2>
    <input id="link-target" placeholder="Target note name" />
    <input id="link-rel" placeholder="relation" value="co_occurs" />
    <div class="actions"><button type="button" id="add-edge">Link</button></div>
    <h2>Episodes</h2>
    <div id="episodes"></div>
  `;
  $("save-summary").onclick = async () => {
    await api(`/agent/vault/nodes/${node.id}`, {
      method: "PATCH",
      body: JSON.stringify({ summary: $("summary").value }),
    });
  };
  $("add-edge").onclick = async () => {
    const name = $("link-target").value.trim();
    if (!name) return;
    const found = await api(`/agent/vault/nodes?q=${encodeURIComponent(name)}&limit=8`);
    const target = (found.items || []).find((n) => n.name.toLowerCase() === name.toLowerCase()) || (found.items || [])[0];
    if (!target) {
      alert("No matching note");
      return;
    }
    await api("/agent/vault/edges", {
      method: "POST",
      body: JSON.stringify({
        src_id: node.id,
        dst_id: target.id,
        relation_type: $("link-rel").value.trim() || "co_occurs",
      }),
    });
    selectNode(node.id, true);
  };
  const bl = $("backlinks");
  if (!neighbors.length) bl.innerHTML = '<p class="muted">No links yet</p>';
  for (const e of neighbors) {
    const otherId = e.src_id === node.id ? e.dst_id : e.src_id;
    const otherName = e.src_id === node.id ? e.dst_name : e.src_name;
    const row = document.createElement("div");
    row.className = "backlink";
    row.innerHTML = `<a href="#">${escapeHtml(otherName)}</a> <span class="muted">${escapeHtml(e.relation_type)} · ${e.direction}</span>`;
    row.querySelector("a").onclick = (ev) => {
      ev.preventDefault();
      selectNode(otherId, true);
    };
    const del = document.createElement("button");
    del.type = "button";
    del.textContent = "Unlink";
    del.onclick = async () => {
      await api(`/agent/vault/edges/${e.id}`, { method: "DELETE" });
      selectNode(node.id, true);
    };
    row.appendChild(del);
    bl.appendChild(row);
  }
  const epEl = $("episodes");
  if (!episodes.length) epEl.innerHTML = '<p class="muted">No linked episodes</p>';
  for (const ep of episodes) {
    const div = document.createElement("div");
    div.className = "ep";
    div.textContent = `${ep.ts} · ${ep.session_id} · ${(ep.user_text || "").slice(0, 80)}`;
    div.onclick = () => openEpisode(ep.id);
    epEl.appendChild(div);
  }
}

async function openEpisode(id) {
  const ep = await api(`/agent/vault/episodes/${id}`);
  $("ep-title").textContent = `Episode ${ep.id} · ${ep.session_id}`;
  $("ep-body").textContent = `${ep.ts}\n\nUser:\n${ep.user_text}\n\nSeki:\n${ep.assistant_text}`;
  $("episode-dialog").showModal();
}

function escapeHtml(s) {
  return String(s || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

async function loadGraph(opts = {}) {
  const canvas = $("graph");
  const q = ($("search").value || "").trim();
  const params = new URLSearchParams({ hours: String(hours()), limit: "50" });
  if (q) params.set("q", q);
  if (opts.centerId) params.set("center_id", String(opts.centerId));
  const data = await api(`/agent/vault/graph?${params.toString()}`);
  state.nodes = data.nodes || [];
  state.edges = data.edges || [];
  seedLayout(canvas);
  if (!state.running) {
    state.running = true;
    loop(canvas);
  }
  $("hint").textContent = state.nodes.length
    ? `${state.nodes.length} notes · ${state.edges.length} links`
    : "No notes in this slice yet — chat first, or widen Hours to 0";
}

async function loadSide() {
  const hot = await api(`/agent/vault/hottest?hours=${hours()}&limit=12`);
  renderList($("hot-nodes"), hot.nodes || [], "node");
  renderList($("hot-edges"), hot.edges || [], "edge");
  const stats = await api("/agent/vault/stats");
  $("stats").textContent = `${stats.nodes} notes · ${stats.edges} links · ${stats.episodes} episodes`;
  const chain = await api(`/agent/vault/chain?session_id=${encodeURIComponent(sessionId())}`);
  if (!chain.chain) {
    $("chain").textContent = `No chain stored for session ${sessionId()}.`;
    return;
  }
  const c = chain.chain;
  const steps = (c.steps || []).map((s) => `${s.step_type}:${s.name}`).join(" → ");
  $("chain").textContent = `input: ${c.input_text || ""}\n${steps || "(no steps)"}\ncache_hits=${c.cache_hits ?? 0}`;
}

async function reload() {
  await Promise.all([loadGraph(), loadSide()]);
}

function bindCanvas() {
  const canvas = $("graph");
  const loc = (ev) => {
    const rect = canvas.getBoundingClientRect();
    return { x: ev.clientX - rect.left, y: ev.clientY - rect.top };
  };
  canvas.addEventListener("pointerdown", (ev) => {
    const { x, y } = loc(ev);
    const id = hitTest(x, y);
    if (id != null) {
      state.drag = id;
      canvas.setPointerCapture(ev.pointerId);
      selectNode(id, false);
    }
  });
  canvas.addEventListener("pointermove", (ev) => {
    if (state.drag == null) return;
    const { x, y } = loc(ev);
    const p = state.pos.get(state.drag);
    if (p) {
      p.x = x;
      p.y = y;
    }
  });
  canvas.addEventListener("pointerup", () => {
    state.drag = null;
  });
}

let searchTimer = null;
$("search").addEventListener("input", () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => loadGraph(), 250);
});
$("reload").onclick = reload;
$("hours").onchange = reload;
$("session").onchange = loadSide;
window.addEventListener("resize", () => draw($("graph")));

bindCanvas();
reload().catch((err) => {
  $("hint").textContent = err.message;
});
