# AgriSahayak AI — research frontend

## Run locally

Use Node.js 20.19+ or 22.12+. In this folder run `npm install`, then `npm run dev`.
Start the existing FastAPI backend separately on port 8000. Vite forwards `/api` requests to `http://127.0.0.1:8000`. Adjust vite.config.js if your backend uses another address.
For production use `npm run build` and configure your host to proxy `/api` to your backend; the development proxy is not included in the built files. Never put Gemini secrets in the frontend.

The uploaded archive did not contain package.json or Vite configuration. These files supply a minimal setup; if merging into an existing working project, keep your existing configuration and merge only what is needed.

## Changes

- New ResearchUI.jsx and research.css provide a responsive research workspace, question composer, Hindi/Kannada example selection, retrieved evidence cards, loading/error/retry states and architecture page.
- main.jsx loads ResearchUI. Original App.jsx remains for the existing retrieval and model comparison pages. Only those two functions were exported.
- Existing API client and request payloads are unchanged. Main assistant uses `model_choice: "Gemini API"`.
- Language buttons change example questions only. No unsupported language parameter is added. Answer language handling remains the backend's responsibility.
- Retrieved evidence is displayed as context, not fabricated sentence-level citations. The new assistant labels numeric scores neutrally because score semantics must be confirmed in the backend.
- Static benchmark values in the original research pages remain as supplied, with an explicit unverified-results notice above them. Do not present these as independently validated results without evaluation files. No performance values were invented.
- No fake backend status, progress stages, confidence scores or answer fixtures are shipped.

## Backend contract preserved

POST /api/chat: {question, model_choice}; response {answer: string, retrieved: [{rank, chunk_id, title, text, score, source, url}]}.
POST /api/retrieve: {question, top_k}.
POST /api/compare: {question, top_k}.
POST /api/labeled-answer: {question}.

Only the frontend was supplied, so live backend functionality and Hindi/Kannada retrieval quality cannot be verified here. The original comparison UI can request a generated labeled answer when ground truth exists; it may use your configured generation service.

## Design

White and muted green workspace, dark green typography, readable evidence cards, no fabricated commercial claims. System fonts include script-aware fallbacks. No external image assets or font requests are needed for the new interface.

## Verification performed

- `npm run build` passed.
- DOM interaction checks passed with a mocked API: initial rendering, Kannada example selection, unchanged chat payload, failed request and retry, answer/evidence display, unsafe source URL suppression, reset, no-evidence warning, architecture navigation and original research page navigation.
- Browser screenshot/layout inspection could not be completed because the browser downloads timed out. Responsive styles are implemented but need a visual check on your phone and laptop.
- No real backend requests or model generations were run during testing.
