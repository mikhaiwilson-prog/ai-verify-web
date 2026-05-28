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

import { createC2pa } from "https://cdn.jsdelivr.net/npm/@contentauth/c2pa-web@0.8.1/+esm";
import {
  FilesetResolver,
  FaceDetector,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm";

// ── Config ─────────────────────────────────────────────────────────────

const C2PA_VERSION = "0.8.1";
const VISION_VERSION = "0.10.14";

// Mirrors src/ai_verify/config.py — keep in sync with the Python plugin.
const AI_GENERATOR_KEYWORDS = [
  "chatgpt", "openai", "dall-e", "dalle", "gpt-image",
  "firefly", "adobe generative", "midjourney", "stable diffusion",
  "imagen", "gemini", "ideogram", "leonardo", "flux",
  "comfyui", "automatic1111", "runway", "pika", "sora",
  "media service", "designer", "aurora", "recraft",
  "nightcafe", "freepik", "magnific",
];

// IPTC digitalSourceType vocabulary — AI-related entries.
// Source: https://cv.iptc.org/newscodes/digitalsourcetype/
const AI_DST_SUBSTRINGS = [
  "trainedAlgorithmicMedia",
  "compositeWithTrainedAlgorithmicMedia",
  "compositeSynthetic",
  "virtualRecording",
];

// Issuers known to sign C2PA manifests on AI-generated content.
const AI_SIGNATURE_ISSUERS = [
  "openai opco",
  "adobe inc",
  "google llc",
  "stability ai",
  "black forest labs",
  "midjourney",
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
    wasmSrc: `https://cdn.jsdelivr.net/npm/@contentauth/c2pa-web@${C2PA_VERSION}/dist/resources/c2pa_bg.wasm`,
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
  root.classList.remove("ready", "error", "hidden");
  if (state === "ready") {
    // Once ready, hide the boot pill — keep the UI minimal.
    root.classList.add("hidden");
    return;
  }
  if (state === "error") root.classList.add("error");
  t.textContent = text;
}

// ── C2PA inspection ────────────────────────────────────────────────────

// Action helper: read digitalSourceType from action or action.parameters.
function actionDst(action) {
  const direct = action?.digitalSourceType;
  if (direct) return String(direct);
  const params = action?.parameters || {};
  return String(params.digitalSourceType || "");
}

// Action helper: read softwareAgent name (string or {name} object form).
function actionSoftwareAgent(action) {
  let sw = action?.softwareAgent;
  if (sw == null) sw = (action?.parameters || {}).softwareAgent;
  if (sw && typeof sw === "object") return sw.name ?? null;
  if (typeof sw === "string") return sw;
  return null;
}

function isAiDst(dst) {
  if (!dst) return false;
  const lower = String(dst).toLowerCase();
  return AI_DST_SUBSTRINGS.some((sub) => lower.includes(sub.toLowerCase()));
}

function matchesAiIssuer(issuer) {
  if (!issuer) return false;
  const lower = String(issuer).toLowerCase();
  return AI_SIGNATURE_ISSUERS.some((kw) => lower.includes(kw));
}

async function readC2pa(file) {
  let reader = null;
  try {
    // @contentauth/c2pa-web@0.8.x API: reader.fromBlob returns null when
    // there is no C2PA metadata. Use blob.type as the asset format.
    reader = await c2pa.reader.fromBlob(file.type, file);
    if (reader === null) {
      return emptyC2pa();
    }

    // manifestStore() returns the raw c2pa-rs JSON shape (snake_case):
    // { active_manifest, manifests: { [label]: Manifest }, validation_status, ... }
    // Same shape the Python c2pa-python library returns — keep both in lockstep.
    const manifestStore = await reader.manifestStore();

    const activeLabel = manifestStore?.active_manifest;
    if (!activeLabel) return emptyC2pa();

    const manifest = manifestStore?.manifests?.[activeLabel];
    if (!manifest) return emptyC2pa();

    const validationStatus = manifest.validation_status || [];
    const c2paValid = validationStatus.length === 0;

    const genInfo = manifest.claim_generator_info || [];
    const claimGenName = genInfo[0]?.name ?? manifest.claim_generator ?? null;

    const sigInfo = manifest.signature_info || {};
    const signatureIssuer = sigInfo.issuer ?? null;

    let hasAiAssertion = false;
    let aiSourceType = null;
    let aiSoftwareAgent = null;

    // C2PA 2.2 spec: action labels include `c2pa.actions` and `c2pa.actions.v2`.
    // Match by prefix to catch any version.
    for (const assertion of manifest.assertions || []) {
      const label = assertion?.label || "";
      const data = assertion?.data || {};
      if (label.startsWith("c2pa.actions")) {
        for (const action of data.actions || []) {
          const dst = actionDst(action);
          if (isAiDst(dst)) {
            hasAiAssertion = true;
            aiSourceType = dst;
            aiSoftwareAgent = actionSoftwareAgent(action);
            break;
          }
        }
      }
      if (hasAiAssertion) break;
    }

    // Fallback 1: claim generator name matches a known AI tool (case-insensitive)
    if (!hasAiAssertion && claimGenName) {
      const lower = String(claimGenName).toLowerCase();
      if (AI_GENERATOR_KEYWORDS.some((kw) => lower.includes(kw))) {
        hasAiAssertion = true;
      }
    }

    // Fallback 2: signature issuer matches a known AI signer
    if (!hasAiAssertion && matchesAiIssuer(signatureIssuer)) {
      hasAiAssertion = true;
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
      raw_manifest: manifestStore,
      error: null,
    };
  } catch (err) {
    return { ...emptyC2pa(), error: String(err?.message ?? err) };
  } finally {
    if (reader && typeof reader.free === "function") {
      try {
        await reader.free();
      } catch {
        /* ignore */
      }
    }
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
      note: "High confidence — no further check needed.",
      showLinks: false,
    };
  }
  if (c2pa.has_c2pa && !c2pa.has_ai_assertion) {
    return {
      kind: "ai-clean",
      iconClass: "ph-shield-check",
      title: "Local C2PA is clean — no AI signal",
      rows: [["Signed by", c2pa.signature_issuer ?? "—"]],
      note: "Strong evidence this is not AI, but C2PA alone is not definitive. If you want full confidence, download the masked image and check it on BOTH OpenAI Verify AND Gemini below.",
      showLinks: true,
    };
  }
  return {
    kind: "inconclusive",
    iconClass: "ph-clock-countdown",
    title: "Can't verify locally — manual cross-check required",
    rows: [],
    note: "No C2PA Content Credentials found. This is NOT evidence the image is real — most cameras and edited images also lack C2PA. AI generators can also strip it. To know whether this image is AI, you MUST check the masked version on BOTH OpenAI Verify AND Gemini below. Checking only one service is not enough — they detect different AI providers.",
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

function setCardState(card, state, _statusText) {
  card.dataset.state = state;
  // Status pill removed for minimalism — the verdict-card title now carries
  // the entire state signal. We still flip card.dataset.state so the
  // accent bar + verdict-card border color can react.
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
    // 1. C2PA (cheap, deterministic — no pixels needed)
    const c2paResult = await readC2pa(file);

    // 2. Render verdict + C2PA details
    const verdict = buildVerdict(c2paResult);
    renderVerdict(card, verdict);

    const compactC2pa = { ...c2paResult };
    if (compactC2pa.raw_manifest) {
      compactC2pa.raw_manifest = "<full manifest available; omitted from preview>";
    }
    card.querySelector(".result-c2pa-json").textContent = JSON.stringify(
      compactC2pa,
      null,
      2,
    );

    // 3. If AI is confirmed by C2PA, no need to mask — analyst won't be doing
    //    any remote cross-check, so the masked artifact serves no purpose.
    if (verdict.kind === "ai-confirmed") {
      setCardState(card, "ok", STATUS_LABELS[verdict.kind] ?? "Done");
      return;
    }

    // 4. Otherwise: load image and mask the face for the analyst's
    //    optional manual upload to Gemini / OpenAI Verify.
    const img = await loadImageFromFile(file);

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

    // 5. Manual cross-check links — shown only when masked image is available
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
