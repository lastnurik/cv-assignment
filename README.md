# Gesture Lab | Assignment 2

Repository: https://github.com/lastnurik/cv-assignment (private).

Three-class hand gesture recognition: peace, stop, thumbs_up. A local web app provides upload/camera input, probabilities and actual Grad-CAM. ConvNeXtTiny is the modern component; the controlled ablation compares pretrained versus random frozen ConvNeXt features.

## Run the app

Python 3.12.15 was verified. From this project directory:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-runtime.lock.txt
.venv/bin/python demo.py --bundle deliverables/model_bundle.json
```

Reuse an existing working `.venv` instead of recreating it. Open `http://127.0.0.1:8765`. Upload is verified in Chrome 155 on Linux. Camera permission denial falls back to upload; physical-camera capture is unverified. Inputs remain in memory and are not saved. The model has no validated no-hand detector.

The bundle refers to `runs/S1_limited_v1/best.keras` and checks its SHA-256. Use the private project assets alongside the source repository; model/data files are deliberately excluded from public-source packaging. See `deliverables/RELEASE_GUIDE.md`.

## Verified experiment results

| Run | Method | Test accuracy | Macro-F1 |
|---|---|---:|---:|
| B0 | Original simple CNN approach, clean retraining | 47.06% | 0.3670 |
| B1 | MobileNetV2 transfer/fine-tuning | 86.27% | 0.8050 |
| S1 | ImageNet-pretrained frozen ConvNeXtTiny | 78.43% | 0.7300 |
| A0 | Random frozen ConvNeXtTiny, matched head training | 41.18% | 0.4108 |

All runs use the same 51 test images. S1 was selected on validation before test evaluation; it remains the deployed model even though B1 scored better on test. S1's CPU classifier mean latency is 395.36 ms (10 warmups, 100 calls), excluding explanation. Raw measurements, class support, probabilities, confusion matrices and curves are in `artifacts/evaluation/` and `runs/*/predictions.json`.

## Dataset and limits

464 A1 images were cropped uniformly to remove burned-in class text; 11 no-hand stop samples were excluded. The frozen `a1_limited_v2` manifest contains 453 retained samples: 344 train, 58 validation, 51 test. Known dHash similarity components and exact/source duplicates stay together. Unknown original sessions are not invented. This remains a correlated, one-participant/one-indoor-setting evaluation. Test stop support is only four. No independent-person/session/environment performance is claimed.

The user confirmed project-use consent for the participant and background classmates. Public redistribution is not authorized. Synthetic training share is zero. Earlier privacy-crop and EfficientNet candidates were not used in final experiments.

## Reproduce checks and experiments

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m scripts.validate_data data/manifests/a1_limited_v2.json
.venv/bin/python -m scripts.evaluate_predictions runs/B0_limited_v1/predictions.json runs/B1_limited_v1/predictions.json runs/S1_limited_v1/predictions.json runs/A0_limited_v1/predictions.json --output /tmp/gesture_comparison.json
.venv/bin/python -m scripts.plot_release
```

Training refuses to overwrite existing run IDs. For a new reproduction, copy a frozen run config, change only `run_id` to a fresh name, then run `train.py --config <new-config>`. Keep data/contract hashes and controlled settings intact. The original A1 inputs remain preserved. ConvNeXt uses built-in normalization on RGB [0,255]; MobileNet uses embedded [-1,1] scaling. Crop exactly once.

The ConvNeXt official weights are fetched by Keras; use `KERAS_HOME=artifacts/method/pilot_keras_cache` to reuse the local cache. Initialization checksums/source references are recorded in `artifacts/method/feasible_alternative.md`. S1/A0 train identical seeded Dense heads for 20 epochs; frozen train/validation features are cached, never test features.

## Deliverables and state

- Defense slides: `deliverables/A2_Defense_Presentation.pptx` (9 editable slides, speaker notes).
- Report: `output/pdf/A2_Gesture_Recognition_Report.pdf` (7 pages, visually checked).
- Selected model bundle: `deliverables/model_bundle.json`.
- Silent demo: `deliverables/A2_Demo_Silent.mp4` (04:10).
- Record your narration using `deliverables/A2_Demo_Transcript.md`. Add the aligned audio with `python scripts/add_demo_audio.py path/to/narration.wav`.
- Presentation screens: `deliverables/presentation/method.html`.
- Source and private asset archives: `deliverables/`. See `deliverables/FINAL_HANDOFF.md` for submission files.
- Authoritative execution checkpoint: `A2_EXECUTION_PLAN.md`.

The source repository is published privately on GitHub. Give the course reviewer access before sharing its link. Video narration/hosting, Moodle submission and the practice defense remain pending.
