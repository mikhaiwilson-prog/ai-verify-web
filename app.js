// ai-verify-web — fully client-side AI image verification.
//
// Pipeline per image:
//   1. Read C2PA manifest via c2pa-js (returns AI-assertion + generator info)
//   2. Detect faces via MediaPipe Tasks Vision (BlazeFace short-range)
//   3. Mask each detected face on a canvas (privacy guardrail — face never
//      leaves the browser; canvas output is what the user downloads)
//   4. Surface a 3-branch verdict card identical to the Python plugin's logic
//
// PRIVACY: this file does not perform any network I/O on user images.
// CDN-hosted WASM and model files are pulled once at boot time. The user's
// image data never leaves the browser tab.

import { createC2pa } from "https://cdn.jsdelivr.net/npm/c2pa@0.27.0/+esm";
import {
  FilesetResolver,
  FaceDetector,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm";

// ── Config ─────────────────────────────────────────────────────────────

const C2PA_VERSION = "0.27.0";
const VISION_VERSION = "0.10.14";

const AI_GENERATOR_KEYWORDS = [
  "chatgpt", "openai", "dall-e", "dalle", "gpt-image",
  "firefly", "adobe generative", "midjourney", "stable diffusion",
  "imagen", "gemini", "ideogram", "leonardo", "flux",
  "comfyui", "automatic1111", "runway", "pika", "sora",
];

const AI_DST_SUBSTRINGS = [
  "trainedAlgorithmicMedia",
  "compositeWithTrainedAlgorithmicMedia",
];

const FACE_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_detector/" +
  "blaze_face_short_range/float16/1/blaze_face_short_range.tflite";

const EXPAND_PCT = 0.15;
const MIN_FACE_CONFIDENCE = 0.5;

// ── Engine bootstrap ───────────────────────────────────────────────────

let c2pa = null;
let faceDetector = null;

async function initEngines() {
  setBoot("warn", "Loading C2PA reader…");
  c2pa = await createC2pa({
    wasmSrc: `https://cdn.jsdelivr.net/npm/c2pa@${C2PA_VERSION}/dist/assets/wasm/toolkit_bg.wasm`,
    workerSrc: `https://cdn.jsdelivr.net/npm/c2pa@${C2PA_VERSION}/dist/c2pa.worker.min.js`,
  });

  setBoot("warn", "Loading face detector (one-time, ~230 KB)…");
  const vision = await FilesetResolver.forVisionTasks(
    `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VISION_VERSION}/wasm`,
  );
  faceDetector = await FaceDetector.createFromOptions(vision, {
    baseOptions: { modelAssetPath: FACE_MODEL_URL },
    runningMode: "IMAGE",
    minDetectionConfidence: MIN_FACE_CONFIDENCE,
  });

  setBoot("ready", "Ready. Drop or choose image(s) to verify.");
}

function setBoot(state, text) {
  const root = document.getElementById("bootStatus");
  const t = document.getElementById("bootText");
  root.classList.remove("ready", "error");
  if (state === "ready") root.classList.add("ready");
  if (state === "error") root.classList.add("error");
  t.textContent = text;
}

// ── C2PA inspection ────────────────────────────────────────────────────

async function readC2pa(file) {
  try {
    const result = await c2pa.read(file);
    const manifestStore = result?.manifestStore;
    if (!manifestStore || !manifestStore.activeManifest) {
      return emptyC2pa();
    }
    const m = manifestStore.activeManifest;

    const claimGenName =
      m.claimGenerator?.name ??
      m.claimGenerator?.value ??
      (typeof m.claimGenerator === "string" ? m.claimGenerator : null);

    const signatureIssuer = m.signatureInfo?.issuer ?? null;
    const validationStatus = m.validationStatus ?? [];
    const c2paValid = validationStatus.length === 0;

    let hasAiAssertion = false;
    let aiSourceType = null;
    let aiSoftwareAgent = null;

    const assertions = m.assertions ?? [];
    const assertionList = Array.isArray(assertions)
      ? assertions
      : Object.values(assertions);

    for (const a of assertionList) {
      const label = a.label ?? a.uri ?? "";
      const data = a.data ?? a;
      if (label.includes("c2pa.actions")) {
        const actions = data?.actions ?? [];
        for (const action of actions) {
          const dst = action.digitalSourceType ?? "";
          if (AI_DST_SUBSTRINGS.some((sub) => dst.includes(sub))) {
            hasAiAssertion = true;
            aiSourceType = dst;
            const sw = action.softwareAgent;
            aiSoftwareAgent =
              typeof sw === "string" ? sw : sw?.name ?? null;
            break;
          }
        }
      }
    }

    if (!hasAiAssertion && claimGenName) {
      const lower = claimGenName.toLowerCase();
      if (AI_GENERATOR_KEYWORDS.some((kw) => lower.includes(kw))) {
        hasAiAssertion = true;
      }
    }

    return {
      has_c2pa: true,
      c2pa_valid: c2paValid,
      validation_state: c2paValid ? "valid" : JSON.stringify(validationStatus),
      claim_generator_name: claimGenName,
      signature_issuer: signatureIssuer,
      has_ai_assertion: hasAiAssertion,
      ai_source_type: aiSourceType,
      ai_software_agent: aiSoftwareAgent,
      raw_manifest: m,
      error: null,
    };
  } catch (err) {
    const msg = String(err?.message ?? err).toLowerCase();
    const noManifestPhrases = [
      "manifest not found",
      "no claim",
      "jumbf not found",
      "no manifest",
    ];
    if (noManifestPhrases.some((p) => msg.includes(p))) {
      return emptyC2pa();
    }
    return { ...emptyC2pa(), error: String(err?.message ?? err) };
  }
}

function emptyC2pa() {
  return {
    has_c2pa: false,
    c2pa_valid: null,
    validation_state: null,
    claim_generator_name: null,
    signature_issuer: null,
    has_ai_assertion: false,
    ai_source_type: null,
    ai_software_agent: null,
    raw_manifest: null,
    error: null,
  };
}

// ── Face masking ───────────────────────────────────────────────────────

async function loadImageFromFile(file) {
  const url = URL.createObjectURL(file);
  try {
    const img = new Image();
    img.crossOrigin = "anonymous";
    await new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = reject;
      img.src = url;
    });
    return img;
  } finally {
    // Revoke after a tick so detection has the bitmap loaded
    setTimeout(() => URL.revokeObjectURL(url), 0);
  }
}

async function maskFaces(imgElement) {
  const result = faceDetector.detect(imgElement);
  const detections = result?.detections ?? [];
  if (detections.length === 0) {
    const err = new Error("No face detected");
    err.code = "FACE_DETECTION_FAILED";
    throw err;
  }

  const canvas = document.createElement("canvas");
  canvas.width = imgElement.naturalWidth;
  canvas.height = imgElement.naturalHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(imgElement, 0, 0);

  for (const det of detections) {
    const bb = det.boundingBox;
    const x = Math.max(0, bb.originX - bb.width * EXPAND_PCT);
    const y = Math.max(0, bb.originY - bb.height * EXPAND_PCT);
    const w = Math.min(canvas.width - x, bb.width * (1 + 2 * EXPAND_PCT));
    const h = Math.min(canvas.height - y, bb.height * (1 + 2 * EXPAND_PCT));
    ctx.fillStyle = "rgb(128, 128, 128)";
    ctx.fillRect(x, y, w, h);
  }

  return { canvas, faces: detections.length };
}

function canvasToBlob(canvas) {
  return new Promise((resolve) =>
    canvas.toBlob(resolve, "image/jpeg", 0.95),
  );
}

// ── Verdict ────────────────────────────────────────────────────────────

function buildVerdict(c2pa) {
  if (c2pa.has_c2pa && c2pa.has_ai_assertion) {
    return {
      kind: "ai-confirmed",
      iconClass: "ph-shield-warning",
      title: "AI confirmed by C2PA provenance",
      rows: [
        ["Generator", c2pa.claim_generator_name ?? "—"],
        ["Source type", c2pa.ai_source_type ?? "—"],
        ["Signed by", c2pa.signature_issuer ?? "—"],
      ],
      note: "High confidence — no manual cross-check needed. OpenAI Verify would read the same C2PA chunk this just parsed.",
      showLinks: false,
    };
  }
  if (c2pa.has_c2pa && !c2pa.has_ai_assertion) {
    return {
      kind: "ai-clean",
      iconClass: "ph-shield-check",
      title: "Local C2PA is clean — no AI signal",
      rows: [["Signed by", c2pa.signature_issuer ?? "—"]],
      note: "Strong evidence this is not AI. If you want an extra cross-check, download the masked image and upload it to OpenAI Verify or Gemini below.",
      showLinks: true,
    };
  }
  return {
    kind: "inconclusive",
    iconClass: "ph-clock-countdown",
    title: "Can't verify locally — needs manual cross-check",
    rows: [],
    note: "The image has no C2PA Content Credentials. This is normal for many cameras and is NOT evidence of AI. Download the masked image below and upload it to OpenAI Verify or Gemini SynthID to check.",
    showLinks: true,
  };
}

// ── UI ─────────────────────────────────────────────────────────────────

const resultsEl = document.getElementById("results");
const cardTemplate = document.getElementById("resultCardTemplate");

function appendCard(filename) {
  const card = cardTemplate.content.firstElementChild.cloneNode(true);
  card.querySelector(".result-filename").textContent = filename;
  resultsEl.prepend(card);
  return card;
}

function setCardState(card, state, statusText) {
  card.dataset.state = state;
  card.querySelector(".result-status").textContent = statusText;
}

function renderVerdict(card, verdict) {
  card.dataset.kind = verdict.kind;
  const v = card.querySelector(".verdict-card");
  v.innerHTML =
    `<div class="verdict-title"><i class="ph ${escapeHtml(verdict.iconClass)}" aria-hidden="true"></i>${escapeHtml(verdict.title)}</div>` +
    verdict.rows
      .map(
        ([k, val]) =>
          `<div class="verdict-row"><div class="verdict-key">${escapeHtml(k)}</div><div class="verdict-value">${escapeHtml(val)}</div></div>`,
      )
      .join("") +
    `<div class="verdict-note">${escapeHtml(verdict.note)}</div>`;
}

function renderError(card, message) {
  card.dataset.kind = "error";
  const v = card.querySelector(".verdict-card");
  v.innerHTML = `<div class="verdict-title"><i class="ph ph-warning-circle" aria-hidden="true"></i>Could not process</div><div class="verdict-note">${escapeHtml(message)}</div>`;
}

function escapeHtml(s) {
  if (s == null) return "";
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

async function processFile(file) {
  const card = appendCard(file.name);
  setCardState(card, "processing", "Processing");

  try {
    // 1. C2PA (does not need pixels in DOM — reads the file directly)
    const c2paResult = await readC2pa(file);

    // 2. Load image for face detection
    const img = await loadImageFromFile(file);

    // 3. Face mask. Privacy guardrail: refuse if no face.
    let maskedCanvas = null;
    let maskError = null;
    let facesDetected = 0;
    try {
      const m = await maskFaces(img);
      maskedCanvas = m.canvas;
      facesDetected = m.faces;
    } catch (err) {
      maskError = err.message || String(err);
    }

    // 4. Render verdict
    const verdict = buildVerdict(c2paResult);
    renderVerdict(card, verdict);

    // 5. C2PA detail (collapsed by default)
    const compactC2pa = { ...c2paResult };
    if (compactC2pa.raw_manifest) {
      compactC2pa.raw_manifest = "<full manifest available; omitted from preview>";
    }
    card.querySelector(".result-c2pa-json").textContent = JSON.stringify(
      compactC2pa,
      null,
      2,
    );

    // 6. Manual cross-check links
    if (verdict.showLinks && maskedCanvas) {
      card.querySelector(".manual-links").classList.remove("hidden");
    }

    // 7. Masked preview + download (only if face was found)
    if (maskedCanvas) {
      const blob = await canvasToBlob(maskedCanvas);
      const url = URL.createObjectURL(blob);
      const previewImg = card.querySelector(".result-preview-img");
      previewImg.src = url;
      const downloadBtn = card.querySelector(".result-download");
      downloadBtn.disabled = false;
      const label = downloadBtn.querySelector(".result-download-label");
      label.textContent = `Download masked (${facesDetected} face${facesDetected === 1 ? "" : "s"})`;
      downloadBtn.addEventListener("click", () => {
        const a = document.createElement("a");
        a.href = url;
        a.download = `ai-verify-masked-${stripExt(file.name)}.jpg`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
      });
    } else {
      // No face detected — surface that prominently, hide manual links + download
      const v = card.querySelector(".verdict-card");
      v.insertAdjacentHTML(
        "beforeend",
        `<div class="verdict-note"><strong>Face masking failed:</strong> ${escapeHtml(maskError ?? "unknown")}. No masked image produced — cannot proceed to manual cross-check.</div>`,
      );
      card.querySelector(".manual-links")?.classList.add("hidden");
      card.querySelector(".result-download").remove();
    }

    setCardState(card, "ok", STATUS_LABELS[verdict.kind] ?? "Done");
  } catch (err) {
    console.error(err);
    renderError(card, err?.message ?? String(err));
    setCardState(card, "error", "Error");
  }
}

const STATUS_LABELS = {
  "ai-confirmed": "AI confirmed",
  "ai-clean": "Clean",
  "inconclusive": "Inconclusive",
  "error": "Error",
};

function stripExt(filename) {
  const dot = filename.lastIndexOf(".");
  return dot === -1 ? filename : filename.slice(0, dot);
}

async function processFiles(fileList) {
  const files = Array.from(fileList).filter((f) => f.type.startsWith("image/"));
  if (files.length === 0) return;
  for (const file of files) {
    await processFile(file);
  }
}

// ── Wire up input + drag-drop ──────────────────────────────────────────

const fileInput = document.getElementById("fileInput");
const dropZone = document.getElementById("dropZone");

fileInput.addEventListener("change", (e) => {
  processFiles(e.target.files);
  e.target.value = ""; // allow re-selecting the same file
});

["dragenter", "dragover"].forEach((evt) =>
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropZone.classList.add("dragging");
  }),
);
["dragleave", "drop"].forEach((evt) =>
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    if (evt === "dragleave" && dropZone.contains(e.relatedTarget)) return;
    dropZone.classList.remove("dragging");
  }),
);
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  if (e.dataTransfer?.files?.length) processFiles(e.dataTransfer.files);
});

// ── Go ─────────────────────────────────────────────────────────────────

initEngines().catch((err) => {
  console.error(err);
  setBoot("error", `Failed to load engines: ${err?.message ?? err}`);
});
