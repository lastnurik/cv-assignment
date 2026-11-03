# Assignment 2 handoff

Use the source repository together with the private project assets. Do not publish identifiable images or private asset archives without separate distribution permission.

## Start

1. Extract the source repository and private assets into the same project root.
2. Create Python 3.12 environment and install `requirements-runtime.lock.txt`.
3. Run `python demo.py --bundle deliverables/model_bundle.json`.
4. Open http://127.0.0.1:8765 and upload an image. The app predicts and computes Grad-CAM on the actual selected ConvNeXt checkpoint.

Physical camera capture is unverified; upload is the verified delivery path. The three-class model can confidently label unrelated inputs.

## Files

- Defense presentation: deliverables/A2_Defense_Presentation.pptx (9 slides with speaker notes and editable charts/tables).
- Seven-page report: output/pdf/A2_Gesture_Recognition_Report.pdf.
- Demo: deliverables/A2_Demo_Silent.mp4 (04:10; actual browser interactions with held result frames, no audio).
- Voiceover: deliverables/A2_Demo_Transcript.md (timestamps match the video).
- Presentation screens: deliverables/presentation/method.html.
- Audio helper: scripts/add_demo_audio.py; writes A2_Demo_With_Your_Audio.mp4.
- Source repository export: deliverables/repository/.
- Source archive: deliverables/A2_Source.zip.
- Private assets: deliverables/A2_Private_Assets.zip (data/model files for permitted project review; not a public upload).
- Measured results: artifacts/evaluation/ and runs/*_limited_v1/predictions.json.

The selected S1 model scores 78.43% accuracy / .7300 macro-F1; B1 scores 86.27% / .8050. Selection used validation only. Do not describe S1 as outperforming B1 on the held-out test.

## Reproduction

All final model/data/config hashes are preserved. `scripts/evaluate_predictions.py` recomputes metrics without TensorFlow inference. `scripts/plot_release.py` rebuilds curves/matrices from saved evidence. Training configs are frozen; use a fresh run ID when reproducing. Source-only checkout requires the private data/checkpoints or regenerated runs. Official pretrained weights are obtained through the Keras loader, not rehosted in the source archive.

Report authoring additionally uses reportlab 5.1.0 and pypdf 6.21.4. Browser/video checks use Playwright 1.64.1, Chrome 155, system ffmpeg and eSpeak NG. They are delivery tools, not inference dependencies.

## Final external handoff

Repository: https://github.com/lastnurik/cv-assignment (private, pushed via SSH). Instructor access is not configured. Video hosting, Moodle submission receipt and practice defense remain pending. Publish only the reviewed source/package under the chosen account and audience; obtain any additional image-distribution permission first. Record your own narration using the timestamped transcript, add it to the silent master and check playback before submission. The v1 generated-speech recording is archived outside the current deliverables.
