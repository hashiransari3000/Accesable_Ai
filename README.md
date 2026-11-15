# AccessAble AI

An offline-first accessibility assistant **prototype**: understand a scene
through object detection, read visible text with OCR, hear concise spoken
summaries, and explore indoor route planning on a sample map.

The interface uses the **Atkinson Hyperlegible** typeface (Braille
Institute), a contrast-checked calm palette, a dark sidebar shell, and
3px focus rings — designed against WCAG 2.2 AA principles with HCI rules
applied throughout (see `docs/UI_UX_GUIDELINES.txt`).

> **Educational prototype only. AI may miss or misidentify objects.
> Do not rely on this app for safe mobility or obstacle avoidance.**

## Features

- **Scene Assistant** — choose **Image upload** or **Live camera** mode;
  detect common objects with a small pretrained YOLO model (CPU), see
  confidence values and an annotated preview, and get cautious
  relative-position descriptions ("appears in the lower center area of the
  image").
- **Read Text** — extract printed text from images with local OCR
  (English), with faithful copy/download and a speak-aloud option.
- **Voice feedback** — offline text-to-speech (pyttsx3 + eSpeak NG) with
  speak / stop / replay / shared rate control and a visual text equivalent.
- **Indoor Navigation Demo** — hand-written BFS and A* shortest-path
  algorithms on a clearly fictional sample building map, with a drawn
  route map, step-by-step distances, and a BFS-vs-A* comparison (cost,
  hops, nodes explored, timing, side-by-side maps).
- **Privacy by default** — all processing is local; no cloud APIs, no
  telemetry (Streamlit usage stats disabled), no data retention.

## Status

**Release (Phase 8) complete + post-release enhancement pass.** All 9
phases (0–8) built, tested, and documented: image upload, local YOLO
detection, relative positions, offline speech, local OCR read-aloud,
controlled live camera, and the BFS/A* navigation demo. The enhanced
build adds a mode-switched Scene Assistant, a single shared voice-rate
control, matplotlib route maps, uploader/clear-flow fixes, and theme
coherence. **81/81 tests pass, verified from a clean virtual environment
install** (exact steps in `docs/SETUP_GUIDE.txt`); ruff lint clean. Full
history in `docs/PHASE_LOG.txt`, final results in `docs/TEST_PLAN.txt`.

## Screenshots

_(To be added from your browser: dashboard, Scene Assistant results with
bounding boxes, Read Text results, Navigation comparison view.)_ See `docs/PHASE_LOG.txt` for progress.

## Tech Stack

| Area            | Choice                                    |
|-----------------|-------------------------------------------|
| Language        | Python                                    |
| UI              | Streamlit                                 |
| Detection       | Ultralytics YOLO (nano, CPU)              |
| Image handling  | OpenCV                                    |
| OCR             | EasyOCR (lazy-loaded)                     |
| Speech          | pyttsx3 + eSpeak NG (offline)             |
| Navigation      | Hand-written BFS / A* (+ NetworkX for viz)|
| Tests           | pytest                                    |

## Installation

See `docs/SETUP_GUIDE.txt` for the full Fedora/Linux guide. Short version:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
sudo dnf install espeak-ng   # offline speech backend (Linux)
```

## Run

```bash
streamlit run app.py
```

## Tests

```bash
pytest -q
```

## Screenshots

_(Added in later phases as features land.)_

## Documentation

- `docs/PROJECT_OVERVIEW.txt` — problem, users, scope, success criteria
- `docs/ARCHITECTURE.txt` — modules, data flow, diagram
- `docs/UI_UX_GUIDELINES.txt` — design tokens, HCI rules, a11y checklist
- `docs/SAFETY_AND_LIMITATIONS.txt` — honest limitations and safe wording
- `docs/TEST_PLAN.txt` — test matrix and results
- `docs/PHASE_LOG.txt` — phase-by-phase build log
- `docs/SETUP_GUIDE.txt` — installation and troubleshooting

## Safety Disclaimer

This project is an educational prototype. It is not a certified assistive
technology product and is not a replacement for a white cane, guide dog,
mobility training, or human assistance. Object detection can miss or
misidentify objects; a 2D image cannot provide reliable distance or
collision risk; the navigation demo uses a fixed fictional map and provides
no live guidance. See `docs/SAFETY_AND_LIMITATIONS.txt`.

## Licence Notes

Application code is part of a university educational project. Third-party
models/packages (Ultralytics YOLO, EasyOCR, eSpeak NG) remain under their
own licences; model provenance, version, and licence are documented in
`docs/` when the detection phase lands.
