const $ = (id) => document.getElementById(id);

const els = {
  dropzone: $("dropzone"), file: $("file"), thumbs: $("thumbs"), dropHint: $("dropHint"),
  projPanel: $("projPanel"),
  removeBg: $("removeBg"), autoTex: $("autoTex"), albedoOnly: $("albedoOnly"),
  steps: $("steps"), stepsOut: $("stepsOut"),
  guidance: $("guidance"), guidanceOut: $("guidanceOut"),
  octree: $("octree"), octreeOut: $("octreeOut"),
  views: $("views"), viewsOut: $("viewsOut"), texRes: $("texRes"),
  seed: $("seed"), faceCount: $("faceCount"),
  genBtn: $("genBtn"), texBtn: $("texBtn"),
  progressWrap: $("progressWrap"), statusText: $("statusText"), progressPct: $("progressPct"), bar: $("bar"),
  error: $("error"), health: $("health"),
  mv: $("mv"), placeholder: $("placeholder"),
  tabShape: $("tabShape"), tabTex: $("tabTex"),
  exportBar: $("exportBar"), fmt: $("fmt"), download: $("download"),
  galleryStrip: $("galleryStrip"), refreshGallery: $("refreshGallery"),
  retexBanner: $("retexBanner"), retexText: $("retexText"), retexCancel: $("retexCancel"),
};
let retextureSource = null;

let selectedFiles = [];
let currentJob = null;
let pollTimer = null;
let shapeUrl = null;
let texUrl = null;

// ── Inputs ────────────────────────────────────────────────
const bindRange = (input, out, fmt = (v) => v) => {
  const upd = () => (out.textContent = fmt(input.value));
  input.addEventListener("input", upd); upd();
};
bindRange(els.steps, els.stepsOut);
bindRange(els.guidance, els.guidanceOut, (v) => Number(v).toFixed(1));
bindRange(els.octree, els.octreeOut);
bindRange(els.views, els.viewsOut);

// ── Upload (multiple) ─────────────────────────────────────
els.dropzone.addEventListener("click", () => els.file.click());
els.file.addEventListener("change", (e) => setImages([...e.target.files]));
["dragenter", "dragover"].forEach((ev) =>
  els.dropzone.addEventListener(ev, (e) => { e.preventDefault(); els.dropzone.classList.add("drag"); })
);
["dragleave", "drop"].forEach((ev) =>
  els.dropzone.addEventListener(ev, (e) => { e.preventDefault(); els.dropzone.classList.remove("drag"); })
);
els.dropzone.addEventListener("drop", (e) => {
  if (e.dataTransfer.files.length) setImages([...e.dataTransfer.files]);
});

function setImages(files) {
  files = files.filter((f) => f.type.startsWith("image/"));
  if (!files.length) return;
  selectedFiles = files;
  els.thumbs.innerHTML = "";
  files.forEach((f, i) => {
    const wrap = document.createElement("div");
    wrap.className = "thumb-item" + (i === 0 ? " primary" : "");
    const img = document.createElement("img");
    img.src = URL.createObjectURL(f);
    wrap.appendChild(img);
    if (i === 0) {
      const tag = document.createElement("span");
      tag.className = "thumb-tag"; tag.textContent = "shape";
      wrap.appendChild(tag);
    }
    els.thumbs.appendChild(wrap);
  });
  els.thumbs.hidden = false;
  els.dropHint.hidden = true;
  els.genBtn.disabled = false;
  if (retextureSource) els.texBtn.disabled = false;
}

// ── Texture mode + projection view slots ──────────────────
let textureMode = "hunyuan";
let viewFiles = {};          // angle -> uploaded File
let aiAngles = new Set();    // angles to AI-fill (empty slots with ✨ active)
let openaiAvailable = false;

const texmodeSel = document.getElementById("texmode");
function applyTextureMode() {
  textureMode = texmodeSel ? texmodeSel.value : textureMode;
  els.projPanel.hidden = textureMode !== "projection";
  const usesRef = ["gptproject", "mvadapter", "mvgpt"].includes(textureMode);
  const gptPanel = document.getElementById("gptPanel");
  if (gptPanel) gptPanel.hidden = !usesRef;
  const gptModeHint = document.getElementById("gptModeHint");
  if (gptModeHint) gptModeHint.hidden = textureMode !== "gptproject";
  const mvHint = document.getElementById("mvHint");
  if (mvHint) mvHint.hidden = textureMode !== "mvadapter";
  const mvgptHint = document.getElementById("mvgptHint");
  if (mvgptHint) mvgptHint.hidden = textureMode !== "mvgpt";
  const mvViewsetWrap = document.getElementById("mvViewsetWrap");
  if (mvViewsetWrap) mvViewsetWrap.hidden = !["mvadapter", "mvgpt"].includes(textureMode);
}
if (texmodeSel) {
  texmodeSel.addEventListener("change", applyTextureMode);
  applyTextureMode();  // initialize panels/hints on load to match the default mode
}

// ── GPT/MV-Adapter: optional style reference image(s) ─────
let gptReferenceFiles = [];
let gptReferenceSides = [];   // side tag per reference file (parallel array)
const REF_SIDES = ["any", "front", "back", "left", "right", "top", "bottom"];
const gptRefTags = document.getElementById("gptRefTags");

function renderRefTags() {
  if (!gptRefTags) return;
  gptRefTags.innerHTML = "";
  gptReferenceFiles.forEach((f, i) => {
    const row = document.createElement("div");
    row.className = "ref-tag-row";
    const thumb = document.createElement("img");
    thumb.className = "ref-tag-thumb";
    thumb.src = URL.createObjectURL(f);
    const name = document.createElement("span");
    name.className = "ref-tag-name";
    name.textContent = f.name || `Reference ${i + 1}`;
    const sel = document.createElement("select");
    sel.className = "ref-tag-side fmt";
    REF_SIDES.forEach((s) => {
      const o = document.createElement("option");
      o.value = s;
      o.textContent = s === "any" ? "Any view" : s.charAt(0).toUpperCase() + s.slice(1);
      if (s === gptReferenceSides[i]) o.selected = true;
      sel.appendChild(o);
    });
    sel.addEventListener("change", () => { gptReferenceSides[i] = sel.value; });
    row.append(thumb, name, sel);
    gptRefTags.appendChild(row);
  });
}

const gptRefSlot = document.getElementById("gptRefSlot");
if (gptRefSlot) {
  const refInput = gptRefSlot.querySelector("input");
  const refRm = gptRefSlot.querySelector(".vslot-rm");
  const refLabel = gptRefSlot.querySelector(".vslot-label");
  gptRefSlot.addEventListener("click", (e) => {
    if (e.target.closest(".vslot-rm")) return;
    refInput.click();
  });
  refInput.addEventListener("change", (e) => {
    const files = [...e.target.files].filter((f) => f.type.startsWith("image/"));
    if (!files.length) return;
    gptReferenceFiles = files;
    // default: first reference = front, the rest = any (user can retag each).
    gptReferenceSides = files.map((_, i) => (i === 0 ? "front" : "any"));
    gptRefSlot.style.backgroundImage = `url(${URL.createObjectURL(files[0])})`;
    gptRefSlot.classList.add("filled");
    if (refLabel) refLabel.textContent = files.length > 1 ? `Reference ×${files.length}` : "Reference";
    refRm.hidden = false;
    renderRefTags();
  });
  refRm.addEventListener("click", (e) => {
    e.stopPropagation();
    gptReferenceFiles = [];
    gptReferenceSides = [];
    refInput.value = "";
    gptRefSlot.style.backgroundImage = "";
    gptRefSlot.classList.remove("filled");
    if (refLabel) refLabel.textContent = "Reference";
    refRm.hidden = true;
    renderRefTags();
  });
}

// Append reference files + their side tags (parallel) to a FormData.
function appendReferences(fd) {
  gptReferenceFiles.forEach((f, i) => {
    fd.append("reference", f);
    fd.append("reference_side", gptReferenceSides[i] || "any");
  });
}

function refreshSlot(slot) {
  const angle = slot.dataset.angle;
  const hasPhoto = !!viewFiles[angle];
  const aiBtn = slot.querySelector(".vslot-ai");
  slot.querySelector(".vslot-rm").hidden = !hasPhoto;
  aiBtn.hidden = hasPhoto;                       // a real photo overrides AI
  aiBtn.disabled = !openaiAvailable;
  aiBtn.classList.toggle("on", !hasPhoto && openaiAvailable && aiAngles.has(angle));
  slot.classList.toggle("filled", hasPhoto);
}

document.querySelectorAll("#projPanel .vslot").forEach((slot) => {
  const angle = slot.dataset.angle;
  const input = slot.querySelector("input");
  aiAngles.add(angle); // default: AI-fill on for empty slots

  slot.addEventListener("click", (e) => {
    if (e.target.closest(".vslot-ai") || e.target.closest(".vslot-rm")) return;
    input.click();
  });
  input.addEventListener("change", (e) => {
    const f = e.target.files[0];
    if (!f || !f.type.startsWith("image/")) return;
    viewFiles[angle] = f;
    slot.style.backgroundImage = `url(${URL.createObjectURL(f)})`;
    refreshSlot(slot);
  });
  slot.querySelector(".vslot-ai").addEventListener("click", (e) => {
    e.stopPropagation();
    aiAngles.has(angle) ? aiAngles.delete(angle) : aiAngles.add(angle);
    refreshSlot(slot);
  });
  slot.querySelector(".vslot-rm").addEventListener("click", (e) => {
    e.stopPropagation();
    delete viewFiles[angle];
    input.value = "";
    slot.style.backgroundImage = "";
    refreshSlot(slot);
  });
  refreshSlot(slot);
});

// ── Generate shape ────────────────────────────────────────
els.genBtn.addEventListener("click", async () => {
  if (!selectedFiles.length) return;
  exitRetexture();   // generating a new shape cancels retexture mode
  resetOutputs();
  const fd = new FormData();
  selectedFiles.forEach((f) => fd.append("images", f));
  fd.append("remove_background", els.removeBg.checked);
  fd.append("auto_texture", els.autoTex.checked);
  fd.append("steps", els.steps.value);
  fd.append("guidance_scale", els.guidance.value);
  fd.append("octree_resolution", els.octree.value);
  fd.append("seed", els.seed.value);
  fd.append("face_count", els.faceCount.value);
  fd.append("views", els.views.value);
  fd.append("tex_resolution", els.texRes.value);
  fd.append("albedo_only", els.albedoOnly.checked);
  fd.append("texture_mode", textureMode);
  if (textureMode === "projection") {
    for (const [angle, f] of Object.entries(viewFiles)) fd.append(angle, f);
    const fill = [...aiAngles].filter((a) => !viewFiles[a]); // only empty slots marked ✨
    fd.append("ai_fill_angles", fill.join(","));
  }
  if (["gptproject", "mvadapter", "mvgpt"].includes(textureMode)) appendReferences(fd);
  if (["mvadapter", "mvgpt"].includes(textureMode)) { const mv = document.getElementById("mvViewset"); if (mv) fd.append("mv_viewset", mv.value); }

  setBusy(true);
  showProgress("Queued", 5);
  try {
    const res = await fetch("/api/generate", { method: "POST", body: fd });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    currentJob = (await res.json()).id;
    poll();
  } catch (err) {
    showError(err.message); setBusy(false);
  }
});

// ── Generate texture (step 2) / retexture ─────────────────
function appendTextureFields(fd) {
  fd.append("texture_mode", textureMode);
  fd.append("remove_background", els.removeBg.checked);
  fd.append("face_count", els.faceCount.value);
  fd.append("views", els.views.value);
  fd.append("tex_resolution", els.texRes.value);
  fd.append("albedo_only", els.albedoOnly.checked);
  if (textureMode === "projection") {
    for (const [angle, f] of Object.entries(viewFiles)) fd.append(angle, f);
    fd.append("ai_fill_angles", [...aiAngles].filter((a) => !viewFiles[a]).join(","));
  }
  if (["gptproject", "mvadapter", "mvgpt"].includes(textureMode)) appendReferences(fd);
  if (["mvadapter", "mvgpt"].includes(textureMode)) { const mv = document.getElementById("mvViewset"); if (mv) fd.append("mv_viewset", mv.value); }
}
els.texBtn.addEventListener("click", async () => {
  hideError();
  try {
    if (retextureSource) {
      if (!selectedFiles.length) { showError("Upload a front image to retexture"); return; }
      setBusy(true); showProgress("Queued for texturing", 60);
      const fd = new FormData();
      fd.append("source_id", retextureSource);
      selectedFiles.forEach((f) => fd.append("images", f));
      appendTextureFields(fd);
      const res = await fetch("/api/retexture", { method: "POST", body: fd });
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      resetOutputs();
      currentJob = (await res.json()).id;
      exitRetexture();
      poll();
    } else {
      if (!currentJob) return;
      setBusy(true); showProgress("Queued for texturing", 60);
      // Send the CURRENT mode + references so the texture step honors the latest UI
      // selection (not whatever mode was chosen when the shape was generated).
      const fd = new FormData();
      appendTextureFields(fd);
      const res = await fetch(`/api/jobs/${currentJob}/texture`, { method: "POST", body: fd });
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      poll();
    }
  } catch (err) {
    showError(err.message); setBusy(false);
  }
});

// ── Polling ───────────────────────────────────────────────
function poll() {
  clearTimeout(pollTimer);
  pollTimer = setTimeout(async () => {
    try {
      const res = await fetch(`/api/jobs/${currentJob}`);
      if (!res.ok) throw new Error("Lost job");
      const job = await res.json();
      render(job);
      if (job.status === "completed" || job.status === "failed") {
        setBusy(false);
        if (job.status === "failed") showError(job.error || "Generation failed");
        else loadGallery();
        return;
      }
      poll();
    } catch (err) {
      showError(err.message); setBusy(false);
    }
  }, 1500);
}

const LABELS = {
  queued: "Queued", loading_model: "Loading model",
  processing_shape: "Generating shape", shape_ready: "Shape ready",
  queued_texture: "Queued for texturing", processing_texture: "Texturing",
  completed: "Completed", failed: "Failed",
};

function render(job) {
  showProgress(LABELS[job.status] || job.status, job.progress || 0);
  if (job.shape_url && job.shape_url !== shapeUrl) {
    shapeUrl = job.shape_url;
    els.tabShape.disabled = false;
    els.texBtn.disabled = false;
    loadModel(shapeUrl, "shape");
  }
  if (job.textured_url && job.textured_url !== texUrl) {
    texUrl = job.textured_url;
    els.tabTex.disabled = false;
    loadModel(texUrl, "tex");
  }
}

// ── Viewer ────────────────────────────────────────────────
function loadModel(url, which) {
  els.mv.src = url;
  els.mv.hidden = false;
  els.placeholder.hidden = true;
  els.exportBar.hidden = false;
  els.tabShape.classList.toggle("active", which === "shape");
  els.tabTex.classList.toggle("active", which === "tex");
}
els.tabShape.addEventListener("click", () => shapeUrl && loadModel(shapeUrl, "shape"));
els.tabTex.addEventListener("click", () => texUrl && loadModel(texUrl, "tex"));

// ── Download / export ─────────────────────────────────────
els.download.addEventListener("click", async () => {
  if (!currentJob) return;
  const fmt = els.fmt.value;
  const orig = els.download.textContent;
  els.download.textContent = fmt === "glb" ? "Downloading…" : "Converting…";
  els.download.disabled = true;
  try {
    const res = await fetch(`/api/jobs/${currentJob}/download/${fmt}`);
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
    const blob = await res.blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${currentJob}.${fmt}`;
    a.click();
    URL.revokeObjectURL(a.href);
  } catch (err) {
    showError("Export failed: " + err.message);
  } finally {
    els.download.textContent = orig;
    els.download.disabled = false;
  }
});

// ── Gallery ───────────────────────────────────────────────
async function loadGallery() {
  try {
    const items = await (await fetch("/api/gallery")).json();
    if (!items.length) {
      els.galleryStrip.innerHTML = '<p class="gallery-empty">No models yet — generate one above.</p>';
      return;
    }
    els.galleryStrip.innerHTML = "";
    items.slice(0, 18).forEach((it) => {
      const card = document.createElement("div");
      card.className = "gcard";
      card.innerHTML = `
        <model-viewer src="${it.preview_url}" auto-rotate rotation-per-second="30deg"
          interaction-prompt="none" disable-zoom camera-controls="false" shadow-intensity="0.6"></model-viewer>
        <div class="gcard-foot">
          <span class="badge ${it.textured ? "tex" : "shp"}">${it.textured ? "textured" : "shape"}</span>
          <span class="gcard-actions">
            ${it.shape_url ? '<button class="gbtn" data-act="tex" title="Texture this model">🎨</button>' : ""}
            <button class="gbtn" data-act="del" title="Delete">🗑</button>
          </span>
        </div>`;
      card.addEventListener("click", () => {
        currentJob = it.id;
        shapeUrl = it.shape_url; texUrl = it.textured_url;
        els.tabShape.disabled = !shapeUrl;
        els.tabTex.disabled = !texUrl;
        loadModel(it.textured_url || it.shape_url, it.textured_url ? "tex" : "shape");
        window.scrollTo({ top: 0, behavior: "smooth" });
      });
      const texBtnEl = card.querySelector('[data-act="tex"]');
      if (texBtnEl) texBtnEl.addEventListener("click", (e) => { e.stopPropagation(); enterRetexture(it.id); });
      card.querySelector('[data-act="del"]').addEventListener("click", (e) => { e.stopPropagation(); deleteModel(it.id, card); });
      els.galleryStrip.appendChild(card);
    });
  } catch { /* ignore */ }
}
els.refreshGallery.addEventListener("click", loadGallery);

// ── Retexture / delete ────────────────────────────────────
function enterRetexture(id) {
  retextureSource = id;
  els.retexBanner.hidden = false;
  els.retexText.textContent = `Re-texturing ${id.slice(0, 8)} — upload a front image, set mode, then Generate Texture.`;
  els.texBtn.disabled = !selectedFiles.length;
  window.scrollTo({ top: 0, behavior: "smooth" });
}
function exitRetexture() {
  retextureSource = null;
  els.retexBanner.hidden = true;
}
els.retexCancel.addEventListener("click", exitRetexture);
async function deleteModel(id, card) {
  try {
    const res = await fetch(`/api/jobs/${id}`, { method: "DELETE" });
    if (!res.ok) throw new Error(res.statusText);
    card.remove();
  } catch (e) {
    showError("Delete failed: " + e.message);
  }
}

// ── UI helpers ────────────────────────────────────────────
function setBusy(b) {
  els.genBtn.disabled = b || !selectedFiles.length;
  els.texBtn.disabled = b || (!shapeUrl && !(retextureSource && selectedFiles.length));
}
function showProgress(text, pct) {
  els.progressWrap.hidden = false;
  els.statusText.textContent = text;
  els.progressPct.textContent = `${Math.round(pct)}%`;
  els.bar.style.width = `${pct}%`;
}
function showError(msg) { els.error.hidden = false; els.error.textContent = msg; }
function hideError() { els.error.hidden = true; }
function resetOutputs() {
  hideError();
  shapeUrl = null; texUrl = null;
  els.tabShape.disabled = true; els.tabTex.disabled = true;
  els.tabShape.classList.remove("active"); els.tabTex.classList.remove("active");
  els.exportBar.hidden = true;
}

// ── Health poll ───────────────────────────────────────────
async function health() {
  try {
    const h = await (await fetch("/api/health")).json();
    if (openaiAvailable !== !!h.openai) {
      openaiAvailable = !!h.openai;
      document.querySelectorAll("#projPanel .vslot").forEach(refreshSlot);
    }
    if (h.model_error) { els.health.className = "health pill down"; els.health.textContent = "model error"; }
    else if (h.model_ready) { els.health.className = "health pill ok"; els.health.textContent = h.queue ? `busy · ${h.queue} queued` : "ready"; }
    else { els.health.className = "health pill busy"; els.health.textContent = "warming up"; }
  } catch {
    els.health.className = "health pill down"; els.health.textContent = "offline";
  }
}
health();
setInterval(health, 4000);
loadGallery();
