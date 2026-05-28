# ai-verify-web

A static, zero-backend web version of the ai-verify tool for trust-and-safety
analysts. Same pipeline as the Claude Code plugin (C2PA inspection + face
masking + verdict card) but runs entirely in the browser.

## Privacy

- **No upload.** Images never leave the browser tab.
- **No backend.** Nothing to compromise server-side because there is no server.
- **No telemetry.** No analytics, no logging, no external calls on user data.
- The only network requests on load are pulling the C2PA reader and BlazeFace
  model from public CDNs (cached after first load).

## Deploy

Three files, no build step required:

```
web/
├── index.html
├── app.js
└── style.css
```

Drop them on any static HTTP host (S3 + CloudFront, internal nginx, GitHub
Pages, an internal Vercel project, etc.). They must be served over HTTP(S),
not `file://`, because the C2PA WASM and MediaPipe model are loaded via ES
module imports that the browser blocks under `file://`.

Internal hosting example (nginx):

```nginx
location /ai-verify/ {
  alias /var/www/ai-verify-web/;
  index index.html;
  add_header Content-Security-Policy
    "default-src 'self'; script-src 'self' https://cdn.jsdelivr.net; connect-src https://cdn.jsdelivr.net https://storage.googleapis.com; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline';";
}
```

## Usage

1. Open the hosted URL.
2. Drop one or more selfies (PNG/JPEG/WEBP) into the upload area, or click
   to choose files.
3. Each image processes locally in sequence:
   - C2PA manifest read
   - Face detected and masked on a canvas
   - Verdict card rendered with one of three states
4. If C2PA was inconclusive, click the Gemini / OpenAI links and upload the
   downloaded masked image manually.

## Verdict states

| State | Meaning |
|---|---|
| 🤖 AI confirmed | C2PA manifest contains an AI assertion (e.g. DALL-E, Firefly). No manual cross-check needed. |
| ✅ No AI signal | Valid C2PA provenance with no AI assertion. Optional manual cross-check available. |
| ⏸️ Inconclusive | No C2PA manifest. Recommend manual cross-check at Gemini SynthID and/or OpenAI Verify. |
| ❌ Error | Couldn't process the file. Most common: no face detected (image not a selfie). |

## Privacy guardrails (browser-side)

The page deliberately mirrors the Python plugin's privacy contract:

- If face detection fails, **no masked image is produced and no download
  button is shown.** The analyst is told why and cannot proceed to manual
  cross-check from the UI.
- The original image is read into memory for processing only; nothing
  persists beyond the browser tab.
- No external services are contacted with the image data.

## Dependencies (CDN, loaded once)

- [`c2pa@0.27.0`](https://www.npmjs.com/package/c2pa) — Adobe's C2PA JS reader (WASM)
- [`@mediapipe/tasks-vision@0.10.14`](https://www.npmjs.com/package/@mediapipe/tasks-vision) — face detection
- BlazeFace short-range tflite model from `storage.googleapis.com`

If your internal network blocks `cdn.jsdelivr.net` or `storage.googleapis.com`,
mirror these assets to your CDN and update the URLs at the top of `app.js`.

## Browser support

Tested on Chrome / Edge / Safari 17+. Requires:
- ES modules
- Canvas 2D
- File API
- WASM
