# AccessAble AI

An accessibility assistant that describes images, reads printed text aloud,
and demonstrates indoor route planning. It runs locally on CPU for full
on-device speech and camera support, and can also be hosted on a free
Streamlit Community Cloud instance. No cloud AI APIs, accounts, or telemetry.

> Educational prototype. It is not a certified assistive-technology product and
> must not be used for safe mobility or obstacle avoidance. AI can miss or
> misidentify objects. See `docs/SAFETY_AND_LIMITATIONS.txt`.

## Features

- **Scene description** — upload an image, use a live camera locally, or take a
  browser camera snapshot on a hosted instance; detect common objects with
  YOLO11n, with confidence values, an annotated preview, and cautious
  image-relative position descriptions.
- **Open-vocabulary search** — look for custom items such as `door`, `stairs`,
  `ramp`, `elevator`, or `handrail` with YOLO-World. Results are less reliable
  than fixed-class detection and may contain false positives.
- **Relative proximity (optional)** — depth bands (very close → far) computed
  within a single image only. These are never metres and never a safety signal.
- **Rubik's cube face check (optional)** — a classical OpenCV 3×3 colour-grid
  heuristic for scrambled cube faces. Not a trained model and not 100%
  accurate; solved single-colour faces, glare, and very small cubes are missed.
- **Read Text** — local EasyOCR extraction (English) with faithful copy/download
  and read-aloud.
- **Voice feedback** — offline text-to-speech locally, or your browser's voices
  on a hosted instance, with speak, stop, replay, and a shared rate control.
  Every spoken message has a visual text equivalent.
- **Camera auto-briefing (optional, local live camera)** — announces only
  objects that appear or clear, with change confirmation to ignore flicker, and
  a persistent text log.
- **Indoor navigation demo** — an interactive two-floor schematic of a clearly
  fictional sample building. Click the map to set a start/destination or close a
  segment, choose Fastest, Step-free, or Quiet profiles, run simulated closure
  scenarios, and step through spoken turn-by-turn instructions. A BFS-vs-A*
  comparison shows how hop-count and distance searches differ.
- **Privacy by design** — no analytics, no data retention, and no external
  requests at runtime beyond one-time model downloads. Images are processed in
  memory and never saved; on a public instance they are processed on the
  hosting server.

## Tech stack

| Area | Choice |
|---|---|
| Language | Python |
| UI | Streamlit |
| Detection | Ultralytics YOLO11n + YOLO-World (CPU) |
| Depth | Depth Anything V2 Small (optional, relative bands) |
| Image handling | OpenCV, Pillow |
| OCR | EasyOCR (loaded on demand) |
| Speech | pyttsx3 + eSpeak NG locally; browser Web Speech API when hosted |
| Navigation | Hand-written BFS / A*, Plotly interactive route maps |
| Tests | pytest |

## Getting started

Full instructions: `docs/SETUP_GUIDE.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
sudo dnf install espeak-ng   # Linux offline speech backend

streamlit run app.py
```

Open http://localhost:8501.

## Free public hosting

Streamlit Community Cloud hosts this app from the GitHub repository on its free
tier. No secrets, API keys, or paid services are required:

1. Sign in at <https://share.streamlit.io> with GitHub.
2. Choose **Create app**, then select repository
   `hashiransari3000/Accesable_Ai`, branch `main`, and entrypoint `app.py`.
3. Pick an app subdomain. The default Python 3.12 works with the pinned
   dependencies.
4. Deploy. `requirements.txt` installs Python packages and `packages.txt`
   installs the Linux libraries OpenCV needs.

The app detects a hosted browser automatically and switches to browser speech
and a browser camera snapshot. Free instances sleep after inactivity and wake on
the next visit; the first model load after a restart downloads weights again.
Memory is capped on the free tier, so avoid loading open-vocabulary detection,
depth, and OCR all at once on a single instance. Live camera mode and offline
(on-server) speech require running the app locally.

## Tests

```bash
pytest -q
```

## Project layout

```
app.py                 Streamlit entry point and page routing
modules/               Detection, OCR, speech, camera, navigation, checks
ui/                    Design tokens, shared components, route maps
assets/                Sample images and the fictional sample map
docs/                  Setup, architecture, and safety documentation
models/                Runtime model cache (weights are never committed)
tests/                 Unit tests and headless UI checks
```

## Model provenance

Model weights download once on first use, then everything works offline; see
`models/README.txt` for sizes and licences.

- YOLO11n and YOLO-World v2 via Ultralytics (AGPL-3.0)
- Depth Anything V2 Small via Hugging Face transformers
- EasyOCR English models (Apache-2.0)
- eSpeak NG (GPL-3.0, system package)

## Safety

Detection confidence is a model score, not a probability of safety. Position
and proximity describe the analysed image only, never the user's physical
surroundings or distance. The navigation demo uses a fixed fictional map and is
not live guidance. Read `docs/SAFETY_AND_LIMITATIONS.txt` before use.

## Licence

Application code is provided for educational use. Third-party models and
packages remain under their own licences as listed above.
