"use strict";

const field = (name) => document.querySelector(`[data-testid="${name}"]`);
const canvas = field("placement-canvas");
const context = canvas.getContext("2d");
const RUN_KEY = "gen-reality-run-id";
let csrfToken = null;
let selection = null;
let draft = {u: 0.5, v: 0.5, heightFrac: 0.25, rotationDeg: 0};
let sourceImage = null;
let assetImage = null;
let savedPreviewReady = false;
let busy = true;
let generation = 0;
let drag = null;

function status(message) {
  field("run-status").textContent = message;
}

function validDraft() {
  return Number.isFinite(draft.u) && draft.u >= 0 && draft.u <= 1 &&
    Number.isFinite(draft.v) && draft.v >= 0 && draft.v <= 1 &&
    Number.isFinite(draft.heightFrac) && draft.heightFrac > 0 && draft.heightFrac <= 1;
}

function draftIsSaved() {
  const saved = selection?.placement;
  return saved && validDraft() && draft.u === saved.u && draft.v === saved.v &&
    draft.heightFrac === saved.heightFrac && saved.rotationDeg === 0;
}

function updateControls() {
  const active = csrfToken !== null && !busy;
  const hasInputs = Boolean(selection?.source && selection?.asset && sourceImage && assetImage);
  field("start-run").disabled = !active;
  field("refresh-run").disabled = !active || !selection;
  field("source-controls").disabled = !active || !selection;
  field("asset-controls").disabled = !active || !selection;
  field("placement-controls").disabled = !active || !hasInputs;
  field("save-placement").disabled = !active || !hasInputs || !validDraft();
  field("create-guides").disabled = !active || !hasInputs || !draftIsSaved() || !savedPreviewReady;
  field("guide-history").disabled = !active || !selection?.history.some((item) => item.status === "PREPARED");
  field("draft-status").textContent = !validDraft() ? "Enter valid position and height values." :
    draftIsSaved() ? "This draft matches the saved placement." : "Unsaved draft. Save placement before creating guides.";
}

function writeDraftFields() {
  field("position-u").value = draft.u;
  field("position-v").value = draft.v;
  field("height-number").value = draft.heightFrac;
  // A range input rounds its thumb. The precise number remains in the draft.
  field("height-range").value = draft.heightFrac;
}

function draw() {
  context.clearRect(0, 0, 1280, 720);
  if (sourceImage) context.drawImage(sourceImage, 0, 0, 1280, 720);
  if (assetImage && selection?.asset && validDraft()) {
    const height = draft.heightFrac * 720;
    const width = height * selection.asset.width / selection.asset.height;
    context.drawImage(assetImage, draft.u * 1280 - width / 2,
      draft.v * 720 - height / 2, width, height);
  }
}

function updateDraftFromNumbers() {
  draft = {u: field("position-u").valueAsNumber, v: field("position-v").valueAsNumber,
    heightFrac: field("height-number").valueAsNumber, rotationDeg: 0};
  if (validDraft()) field("height-range").value = draft.heightFrac;
  draw();
  updateControls();
}

for (const name of ["position-u", "position-v", "height-number"]) {
  field(name).addEventListener("input", updateDraftFromNumbers);
}
field("height-range").addEventListener("input", () => {
  draft.heightFrac = field("height-range").valueAsNumber;
  field("height-number").value = draft.heightFrac;
  draw();
  updateControls();
});

function pointerPosition(event) {
  const box = canvas.getBoundingClientRect();
  return {u: (event.clientX - box.left) / box.width, v: (event.clientY - box.top) / box.height};
}

canvas.addEventListener("pointerdown", (event) => {
  if (busy || !csrfToken || !assetImage || !sourceImage || !validDraft() || event.button !== 0) return;
  const point = pointerPosition(event);
  const halfHeight = draft.heightFrac / 2;
  const halfWidth = draft.heightFrac * 720 * selection.asset.width / selection.asset.height / 1280 / 2;
  if (Math.abs(point.u - draft.u) > halfWidth || Math.abs(point.v - draft.v) > halfHeight) return;
  drag = {id: event.pointerId, u: point.u - draft.u, v: point.v - draft.v};
  canvas.setPointerCapture(event.pointerId);
  event.preventDefault();
});

canvas.addEventListener("pointermove", (event) => {
  if (!drag || drag.id !== event.pointerId) return;
  const point = pointerPosition(event);
  draft.u = Math.max(0, Math.min(1, point.u - drag.u));
  draft.v = Math.max(0, Math.min(1, point.v - drag.v));
  writeDraftFields();
  draw();
  updateControls();
});

function endDrag(event) {
  if (!drag || drag.id !== event.pointerId) return;
  if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
  drag = null;
}
canvas.addEventListener("pointerup", endDrag);
canvas.addEventListener("pointercancel", endDrag);

function expireSession() {
  csrfToken = null;
  field("sign-in-again").hidden = false;
  status("Your session ended. Sign in again.");
  updateControls();
}

async function request(path, options = {}) {
  const headers = {...options.headers};
  if (options.method && options.method !== "GET") headers["X-CSRF-Token"] = csrfToken;
  const response = await fetch(path, {...options, headers, credentials: "same-origin", cache: "no-store"});
  if (!response.ok) {
    if (response.status === 401) expireSession();
    const error = new Error("Local request failed.");
    error.status = response.status;
    throw error;
  }
  return response.json();
}

function jsonRequest(path, method, body) {
  return request(path, {method, headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
}

function mediaPath(path, kind) {
  const base = `/runs/${selection.runId}`;
  const pattern = kind === "preview" ?
    new RegExp(`^${base}/preview/[0-9]{1,10}/(source|asset|placement)$`) :
    new RegExp(`^${base}/guides/att-[0-9a-f]{32}/(keyframe-a|guide-b)$`);
  if (typeof path !== "string" || !pattern.test(path)) throw new Error("Invalid local media path.");
  return path;
}

function loadImage(path, image) {
  return new Promise((resolve, reject) => {
    image.onload = () => { image.onload = image.onerror = null; resolve(image); };
    image.onerror = () => { image.onload = image.onerror = null; reject(new Error("Local image could not load.")); };
    image.src = path;
  });
}

function clearGuides() {
  const video = field("video-guide");
  video.pause();
  video.removeAttribute("src");
  video.load();
  video.hidden = true;
  field("keyframe-guide").removeAttribute("src");
  field("keyframe-guide").hidden = true;
}

function loadVideo(path) {
  const video = field("video-guide");
  return new Promise((resolve, reject) => {
    video.onloadedmetadata = () => { video.onloadedmetadata = video.onerror = null; resolve(); };
    video.onerror = () => { video.onloadedmetadata = video.onerror = null; reject(new Error("Local video could not load.")); };
    video.src = path;
    video.load();
  });
}

async function showGuides(attempt, currentGeneration) {
  clearGuides();
  await loadImage(mediaPath(attempt.keyframeAUrl, "guide"), field("keyframe-guide"));
  if (currentGeneration !== generation) return;
  field("keyframe-guide").hidden = false;
  await loadVideo(mediaPath(attempt.guideBUrl, "guide"));
  if (currentGeneration !== generation) return;
  field("video-guide").hidden = false;
}

function showHistory() {
  const history = field("guide-history");
  history.replaceChildren();
  const blank = document.createElement("option");
  blank.value = "";
  blank.textContent = "Choose a saved guide";
  history.append(blank);
  selection.history.forEach((attempt, index) => {
    const option = document.createElement("option");
    option.value = attempt.attemptId;
    const names = {PREPARED: "Prepared", BUILDING: "Building", FAILED: "Failed", INTERRUPTED: "Interrupted"};
    option.textContent = `Guide ${index + 1} · Input revision ${attempt.inputRevision} · ${names[attempt.status] || "Unavailable"}`;
    option.disabled = attempt.status !== "PREPARED";
    history.append(option);
  });
  history.value = selection.prepared?.attemptId || "";
}

async function acceptSelection(value) {
  if (!/^run-[0-9a-f]{32}$/.test(value.runId)) throw new Error("Invalid local run.");
  selection = value;
  const currentGeneration = ++generation;
  try { localStorage.setItem(RUN_KEY, selection.runId); } catch { /* The current run still works without storage. */ }
  draft = selection.placement ? {...selection.placement} : {u: 0.5, v: 0.5, heightFrac: 0.25, rotationDeg: 0};
  sourceImage = assetImage = null;
  savedPreviewReady = false;
  writeDraftFields();
  draw();
  showHistory();
  clearGuides();
  field("saved-preview").hidden = true;
  field("saved-preview").removeAttribute("src");
  field("saved-preview-status").textContent = selection.placementPreviewUrl ? "Loading the exact saved preview…" : "No placement has been saved.";
  field("run-id").textContent = `Run ${selection.runId} · Revision ${selection.revision}`;
  field("selected-source").textContent = selection.source ? "Selected source: normalized to five seconds." : "No source selected.";
  field("selected-asset").textContent = selection.asset ? `Selected PNG: ${selection.asset.width} × ${selection.asset.height}.` : "No PNG selected.";
  updateControls();
  if (selection.source) {
    const image = await loadImage(mediaPath(selection.source.previewUrl, "preview"), new Image());
    if (currentGeneration !== generation) return;
    sourceImage = image;
  }
  if (selection.asset) {
    const image = await loadImage(mediaPath(selection.asset.previewUrl, "preview"), new Image());
    if (currentGeneration !== generation) return;
    assetImage = image;
  }
  draw();
  if (selection.placementPreviewUrl) {
    await loadImage(mediaPath(selection.placementPreviewUrl, "preview"), field("saved-preview"));
    if (currentGeneration !== generation) return;
    savedPreviewReady = true;
    field("saved-preview").hidden = false;
    field("saved-preview-status").textContent = "Exact saved placement from the local server.";
  }
  if (selection.prepared) await showGuides(selection.prepared, currentGeneration);
}

async function refreshSelection() {
  await acceptSelection(await request(`/runs/${selection.runId}`));
}

async function perform(action, recoverGuide = false) {
  if (busy || !csrfToken) return;
  busy = true;
  updateControls();
  try {
    await action();
  } catch (error) {
    if (error.status === 401) return;
    if ((error.status === 409 || recoverGuide) && selection) {
      try { await refreshSelection(); } catch (refreshError) {
        if (refreshError.status === 401) return;
        status("The request failed. Use Refresh run before trying again.");
        return;
      }
      status("The run was refreshed. Check its current state before trying again.");
    } else if (error.status === 408) {
      status("The upload took too long. Select the file and upload it again.");
    } else {
      status("The local request or media load failed. Use Refresh run before trying again.");
    }
  } finally {
    busy = false;
    updateControls();
  }
}

field("start-run").addEventListener("click", () => perform(async () => {
  status("Starting a local run…");
  await acceptSelection(await jsonRequest("/runs", "POST", {}));
  status("Local run started. Choose its source and PNG.");
}));
field("refresh-run").addEventListener("click", () => perform(async () => {
  status("Refreshing the run…");
  await refreshSelection();
  status("Current run restored.");
}));

function sourceMime(file) {
  const extension = file.name.toLowerCase().match(/\.(mp4|mov)$/)?.[1];
  if (!extension) return null;
  const expected = extension === "mp4" ? "video/mp4" : "video/quicktime";
  return !file.type || file.type === expected ? expected : null;
}

field("upload-source").addEventListener("click", () => perform(async () => {
  const file = field("source-file").files[0];
  const kind = field("source-kind").value;
  if (!file || !sourceMime(file) || file.size === 0 || file.size > 256 * 1024 * 1024 ||
      !["clean_camera", "synthetic_fixture"].includes(kind) || !field("clean-source").checked) {
    status("Choose an MP4 or MOV up to 256 MiB, a source type, and the clean-source declaration.");
    return;
  }
  status("Uploading and preparing the source…");
  const result = await request(`/runs/${selection.runId}/source`, {method: "POST", body: file, headers: {
    "Content-Type": sourceMime(file), "X-Expected-Revision": String(selection.revision),
    "X-Clean-Source": "true", "X-Source-Kind": kind,
  }});
  await acceptSelection(result);
  field("source-file").value = "";
  field("clean-source").checked = false;
  status("Source selected.");
}));

field("upload-asset").addEventListener("click", () => perform(async () => {
  const file = field("asset-file").files[0];
  if (!file || !/\.png$/i.test(file.name) || (file.type && file.type !== "image/png") ||
      file.size === 0 || file.size > 32 * 1024 * 1024 || !field("non-human").checked) {
    status("Choose a PNG up to 32 MiB and confirm the non-human declaration.");
    return;
  }
  status("Uploading and checking the PNG…");
  const result = await request(`/runs/${selection.runId}/asset`, {method: "POST", body: file, headers: {
    "Content-Type": "image/png", "X-Expected-Revision": String(selection.revision), "X-Non-Human": "true",
  }});
  await acceptSelection(result);
  field("asset-file").value = "";
  field("non-human").checked = false;
  status("PNG selected. Adjust the placement draft, then save it.");
}));

field("save-placement").addEventListener("click", () => perform(async () => {
  if (!validDraft()) return;
  status("Saving placement and loading its exact preview…");
  await acceptSelection(await jsonRequest(`/runs/${selection.runId}/placement`, "PUT",
    {expectedRevision: selection.revision, ...draft}));
  status("Placement saved. The server preview is ready.");
}));
field("create-guides").addEventListener("click", () => perform(async () => {
  if (!draftIsSaved() || !savedPreviewReady) return;
  status("Creating the two local guides…");
  await acceptSelection(await jsonRequest(`/runs/${selection.runId}/guides`, "POST",
    {expectedRevision: selection.revision}));
  status("Local guides prepared. Earlier guides remain in history.");
}, true));

field("guide-history").addEventListener("change", () => perform(async () => {
  const attempt = selection.history.find((item) => item.attemptId === field("guide-history").value && item.status === "PREPARED");
  if (!attempt) { clearGuides(); return; }
  status("Loading the selected saved guides…");
  await showGuides(attempt, ++generation);
  status(`Showing saved guides from input revision ${attempt.inputRevision}.`);
}));

async function initialize() {
  try {
    const session = await request("/session");
    csrfToken = session.csrfToken;
    let savedRun = null;
    try {
      savedRun = localStorage.getItem(RUN_KEY);
      if (savedRun && !/^run-[0-9a-f]{32}$/.test(savedRun)) {
        localStorage.removeItem(RUN_KEY);
        savedRun = null;
      }
    } catch { /* Storage is optional for the current session. */ }
    if (savedRun) {
      try {
        await acceptSelection(await request(`/runs/${savedRun}`));
        status("Current run restored.");
      } catch (error) {
        if (error.status === 401) return;
        status(error.status === 404 ? "The saved run is unavailable. Start a local run when ready." :
          "The saved run could not load. Use Refresh run or start a local run.");
      }
    } else {
      status("Start a local run when you are ready.");
    }
  } catch {
    if (csrfToken === null) status("Sign in again to prepare local guides.");
  } finally {
    busy = false;
    updateControls();
  }
}

initialize();
