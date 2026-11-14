# AccessAble AI

An offline-first accessibility assistant **prototype**: understand a scene
through object detection, read visible text with OCR, hear concise spoken
summaries, and explore indoor route planning on a sample map.

> **Educational prototype only. AI may miss or misidentify objects.
> Do not rely on this app for safe mobility or obstacle avoidance.**

## Features

- **Scene Assistant** — upload an image, detect common objects with a small
  pretrained YOLO model (CPU), see confidence values, and get cautious
  relative-position descriptions ("appears on the left side of the image").
- **Read Text** — extract printed text from images with local OCR
  (English), with a speak-aloud option.
- **Voice feedback** — offline text-to-speech (pyttsx3 + eSpeak NG) with
  speak / stop / replay / rate controls and a visual text equivalent.
- **Indoor Navigation Demo** — BFS and A* shortest-path algorithms on a
  clearly fictional sample building map, with simulation-only wording.
- **Privacy by default** — all processing is local; no cloud APIs, no
  telemetry, no data retention.

## Status

Phase 4 complete: local OCR text reading (EasyOCR, English, lazy-loaded)
with a "no text found" state distinct from errors, copy/download support,
and read-aloud. The app now covers Scene Assistant, Read Text, and offline
speech; camera (Phase 5) and navigation (Phase 6) are next. See
`docs/PHASE_LOG.txt` for progress and `docs/TEST_PLAN.txt` for tests.

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
