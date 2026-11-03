# Assignment 2 handoff

The app, experiments, seven-page report and silent demo are ready locally. Student narration, external links and submission remain pending.

| Item | File |
|---|---|
| Report | output/pdf/A2_Gesture_Recognition_Report.pdf |
| Defense slides (editable PPTX) | deliverables/A2_Defense_Presentation.pptx |
| Silent demo, 04:10 | deliverables/A2_Demo_Silent.mp4 |
| Timestamped voiceover | deliverables/A2_Demo_Transcript.md |
| Presentation screens | deliverables/presentation/method.html |
| Source repository | deliverables/repository/ |
| Source ZIP | deliverables/A2_Source.zip |
| Restricted data and model weights | deliverables/A2_Private_Assets.zip |

Start the app using README.md. Open http://127.0.0.1:8765.

## Finish the video

Record the eight transcript sections at their marked times. Place the voice track at 00:00 in the silent video and keep the total length at 04:10. If you export one aligned audio file, the helper can add it:

~~~sh
.venv/bin/python scripts/add_demo_audio.py path/to/your-narration.wav
~~~

It writes deliverables/A2_Demo_With_Your_Audio.mp4. Listen and watch the full result before submitting. The silent master contains no generated speech or music.

## Submit

Choose the repository and video destination and audience. Publication still requires your approval. Keep identifiable images and private assets restricted to permitted course review.

Provide the seven-page PDF, accessible source repository and narrated video links, and the app run guide or app link as requested by the course portal. Submission and the practice defense have not occurred.

## Defense notes

- ConvNeXt is a modern convolutional architecture introduced at CVPR 2022. We do not claim it is the best model in 2026.
- The A1 topic and data remain the same. A fixed top-20% crop removes collection label text; originals are preserved.
- Known dHash similarity components remain within one split. Unknown sessions remain unknown. This is a one-participant, one-setting comparison.
- A0/S1 match architecture, head initialization, seed, optimizer and training budget. Backbone initialization is the ablation factor.
- S1 was selected on validation before test. Its accuracy is 78.43%; CNN scores 47.06%, random ConvNeXt 41.18%, and MobileNetV2 86.27%. The SOTA component did not beat MobileNet on this test.
- Grad-CAM can include the face and background. The report and video include a real error.
- Mean CPU prediction latency is 395.36 ms over 100 calls after 10 warmups, excluding explanation.
- Project-use consent is confirmed. No synthetic training images were used.

Next action: record and add your narration, review playback, then authorize publication or upload the deliverables yourself. No new pictures or retraining are required.
