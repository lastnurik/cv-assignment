# Gesture Lab demo transcript

Video: A2_Demo_Silent.mp4. Total length: 04:10. The video has no audio track. Record the text below in your own voice and place each section at its start timestamp. A short pause at the end of a section is intentional. Do not read the screen directions or headings aloud.

The recorded clicks and model outputs are real. Each example plays at its recorded speed, then holds on the result to leave time for narration. Presentation screens summarize the saved experiment results.

Pronunciation: ConvNeXt = “conv next”; Grad-CAM = “grad cam”; macro-F1 = “macro F one”.

| Time | Screen |
|---|---|
| 00:00–00:24 | The project |
| 00:24–01:00 | Upload and explain |
| 01:00–01:26 | A stop gesture |
| 01:26–01:52 | A thumbs-up gesture |
| 01:52–02:26 | An actual error |
| 02:26–03:00 | Method and controlled ablation |
| 03:00–03:40 | Measured results |
| 03:40–04:10 | Limits and deliverables |

## Voiceover

### 00:00–00:24 | The project

This is Gesture Lab, our hand gesture recognition project. We kept the Assignment One topic and dataset: peace, stop, and thumbs up. For Assignment Two, we added ConvNeXt, compared it with our original model approaches, and built this local app with visual explanations.

### 00:24–01:00 | Upload and explain

I’ll start with a peace image. The original photo has a collection label at the top. Before prediction, the pipeline removes the same top twenty percent from every image, so that label does not reach the model. We then get probabilities for all three classes and a Grad-CAM heatmap. The warmer areas indicate regions that contribute to this class score. Here, peace has the highest probability, but the model is not very certain.

### 01:00–01:26 | A stop gesture

Here is a stop gesture. It goes through the same crop and preprocessing, and the app predicts stop. The timing panel separates the prediction from the explanation, because calculating the heatmap takes longer. These are measured request times. Uploaded photos stay in memory on the local server and are not saved.

### 01:26–01:52 | A thumbs-up gesture

Next is thumbs up. The predicted label is correct, but the heatmap also includes the face and background. That matters: a correct answer does not prove the model has learned only the hand gesture. The app supports camera capture too. We verified upload and the fallback when camera access is unavailable.

### 01:52–02:26 | An actual error

This example shows a real failure. The image is labeled peace, but the model predicts stop. The two probabilities are almost tied, so the result is uncertain even though the interface must choose one label. We kept this failure in the demonstration. Our data contains one main participant in one room. Similarity groups stay together across the split, but we cannot claim the model will generalize to new people or environments.

### 02:26–03:00 | Method and controlled ablation

Our modern component is ConvNeXtTiny, from the ConvNeXt architecture introduced at CVPR in twenty twenty-two. We freeze the backbone and train a three-class head. To test the value of pretraining, we compare ImageNet weights with a randomly initialized frozen ConvNeXt backbone. The architecture, head initialization, optimizer, seed, and twenty-epoch head-training budget are the same. Only backbone initialization changes. The original CNN and MobileNet use the same frozen data split.

### 03:00–03:40 | Measured results

All four models were evaluated on the same fifty-one test images. The simple CNN scored forty-seven point zero six percent accuracy. MobileNet reached eighty-six point two seven percent. Pretrained ConvNeXt reached seventy-eight point four three percent, with macro F one of zero point seven three. Random frozen ConvNeXt reached forty-one point one eight percent. Pretraining clearly helped in this experiment, but ConvNeXt did not outperform MobileNet. We kept ConvNeXt in the app because it was selected using validation before we evaluated the test set.

### 03:40–04:10 | Limits and deliverables

The measured mean prediction time was about three hundred ninety-five milliseconds on our CPU, excluding the heatmap. The main limitations are the single participant, shared setting, and only four stop images in the test set. Everyone pictured consented to course project use. We used no synthetic training images. Our deliverables are the working app, seven-page report, repository package, and this demonstration. The report includes training curves, class results, the ablation, and these limitations.

## Add your recording

Record in a quiet room at a conversational pace. Keep each paragraph within its time window; the script averages about 125 words per minute. Export one audio file starting at 00:00 and ending at 04:10, then align it with the silent video in your editor. Check the full result once before submission.

If you use the supplied helper, run it from the project root after recording:

~~~sh
.venv/bin/python scripts/add_demo_audio.py path/to/your-narration.wav
~~~

It writes deliverables/A2_Demo_With_Your_Audio.mp4 and leaves the silent master unchanged. The audio must be between 4:00 and 4:10; a shorter final pause is padded with silence. Recording per section and aligning sections in an editor gives the closest match.
