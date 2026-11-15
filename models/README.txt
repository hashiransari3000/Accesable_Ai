Runtime-downloaded model weights live here (never committed to Git).

yolo11n.pt
  Model:     YOLO11n (nano) pretrained on COCO (80 object classes)
  Size:      ~5.4 MB
  Download:  automatic, once, from Ultralytics official assets
             (github.com/ultralytics/assets, release v8.4.0) on the first
             detection run
  Package:   ultralytics 8.4.212
  Licence:   AGPL-3.0 (Ultralytics). Used under its terms for this
             educational project; the trained weights are distributed
             by Ultralytics under the same licence terms.
  After the first download, detection works fully offline.

EasyOCR models
  Model:     English detection (craft_mlt_25k, ~80 MB) + recognition
             (english_g2, ~15 MB)
  Download:  automatic, once, on the first "Extract text" use
  Location:  ~/.EasyOCR/ (user home cache managed by EasyOCR itself,
             NOT inside this repository - therefore never committed)
  After the first download, OCR works fully offline.

yolov8s-worldv2.pt
  Model:     YOLO-World v2 small - open-vocabulary detection (text prompts)
  Size:      ~24.7 MB
  Download:  automatic, once, from Ultralytics official assets on the
             first open-vocabulary run
  Extra:     CLIP ViT-B/32 text encoder (~354 MB) also downloads once
             into weights/clip/ (gitignored runtime cache) on the same
             first run
  Package:   ultralytics 8.4.212
  Licence:   AGPL-3.0 (Ultralytics); CLIP weights via ultralytics assets
  After the first download, open-vocabulary detection works offline.

Depth Anything V2 Small (optional)
  Model:     depth-anything/Depth-Anything-V2-small-hf
  Download:  automatic, once, via Hugging Face transformers on the first
             "Estimate relative proximity" run (~100 MB)
  Location:  ~/.cache/huggingface/ (user cache, NOT in this repository)
  Output:    relative depth only - no metric scale, no safety meaning.
  After the first download, proximity bands work offline.
