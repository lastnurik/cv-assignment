# Recommended candidate — ConvNeXtTiny (CVPR 2022)

Follow-up review 2026-11-04. This candidate supersedes the older EfficientNet recommendation below; the initial pilot is preserved as evidence. **ConvNeXtTiny pretrained inference/head-training/save-reload feasibility passes.** Final G03 still needs method selection, ablation and experiment contract; no final training occurred.

## Method and citation mapping

The [primary CVPR 2022 paper, A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545v2), modernizes a conventional CNN with Transformer-inspired design choices. Its architecture is appropriate for a transferable image classifier. Paper-level claims about the model family must not become claims of measured gesture improvement or current 2026 best performance.

The [Neurohive CNN guide dated 25 October 2026](https://neurohive.io/en/guides/convolutional-neural-network/) explicitly names ConvNeXt in its architecture table (2022), CNN-versus-Transformer section, and FAQ. It describes the modernized convolutional design and its competitiveness with similarly sized Transformers. This is a directly relevant citation for a one-page ConvNeXt explanation, although it is a general CNN guide rather than a dedicated ConvNeXt article. The local plan requires a relevant Neurohive citation, not a dedicated article; under that wording this is a defensible mapping. If the instructor requires a dedicated method article or a newer-than-2022 model, that narrower requirement remains unverified.

## Verified implementation and weights

[Keras 3.15.1 ConvNeXt source](https://github.com/keras-team/keras/blob/v3.15.1/keras/src/applications/convnext.py) documents ImageNet-1K pretrained weights and embedded Normalization expecting RGB pixels in [0,255]. A Context7 query was attempted first but returned unrelated EfficientNet snippets; exact primary source was inspected to resolve ConvNeXt-specific syntax. No external MobileNet normalization should be applied.

Official Keras no-top weights: https://storage.googleapis.com/tensorflow/keras-applications/convnext/convnext_tiny_notop.h5 . Exact bytes 111,650,432; SHA256 `d547c096cabd03329d7be5562c5e14798aa39ed24b474157cef5e85ab9e49ef1`, matching the Keras loader's expected checksum. Implementation SHA256 `56354f749a93afa6fcfab6cc2b939e414a3b4a8fa7d52a853eaf3730df3df908`. Cache `artifacts/method/pilot_keras_cache/models/convnext_tiny_notop.h5`.

[Original ConvNeXt project](https://github.com/facebookresearch/ConvNeXt) distributes pretrained checkpoints and states MIT licensing for the project. Keras conversion implementation is Apache-2.0. Official Keras source links its conversion repository, whose current raw README URL returned 404 during this review. Do not assert a separately verified license attached to the converted H5; distribute a download recipe/checksum rather than rehosting those weights until publication review. This does not block local evaluation through the documented official loader.

## Actual bounded pilot

Acceptance written before execution in `pilot_convnext_acceptance.md`. Command `.venv/bin/python -m scripts.sota_pilot_convnext > artifacts/method/pilot_convnext.log 2>&1` exited 0. Evidence `pilot_convnext.json`: CPU TensorFlow 2.21.0/Keras 3.15.1, seed 123, three already-cropped samples, distinct finite `(3,768)` feature tensor, 27,820,128 backbone parameters. One disposable dense-head batch trained with finite loss; full model reload probability agreement within atol=1e-5, rtol=1e-4. Download/load 34.69 seconds; initial three-image batch 0.76 seconds; three warm single-image calls 0.375–0.395 seconds. These are bounded feasibility measurements, not final latency or accuracy estimates.

Spatial Grad-CAM compatibility has a separate pre-execution record and script: `pilot_convnext_spatial_acceptance.md`, `scripts/sota_pilot_convnext_spatial.py`, and corresponding JSON/log. Its status is PASS: finite 7x7 maps for all targets, nonzero maps for all nine sample/target pairs, target/input dependence, finite head training and full save/reload parity. No verified final bundle was created.

## Integration proposal

Use spatial `ConvNeXtTiny(include_top=False, pooling=None, include_preprocessing=True)` as a nested frozen backbone; top-level global-average pooling and Dense3 allow the current engine to target the backbone's spatial output. A0/S1 could compare random versus ImageNet initialization of the same frozen architecture and identical classifier head/optimization; label it a representation/pretraining ablation, not fine-tuning. Frozen-head training with cached embeddings is practical on this CPU. Alternatively explicitly control frozen versus fine-tuned stages with identical initialization and budget. Choose the contract before final runs.

The clean B0/B1 comparisons remain original CNN/MobileNetV2. Synthetic share is zero for this branch. Replace generator-specific P07/P10 requirements and S1 preflight checks before execution, preserving genuine SOTA integration, one-factor ablation, explanation and environment limitations. No source or training files outside isolated pilot scripts changed.

---

## Earlier EfficientNet feasibility investigation (superseded recommendation)

# Feasible non-generative alternative — EfficientNetB0

Research and genuine pilot completed 2026-11-04 by method_feasibility agent. Recommendation: technically feasible candidate, with direct required-style citation. **Not a claim that G03 or assignment eligibility is fully passed.** No current-best or instructor-approval claim is made.

## Evidence and scope

[The primary ICML 2019 paper](https://arxiv.org/abs/1905.11946v5) introduces compound scaling of depth, width and resolution and an EfficientNet model family with strong ImageNet and transfer-learning results at publication. B0 is the smallest base model; the paper's B7 headline result must not be attributed to B0. A pretrained B0 feature extractor is directly relevant to the project's small gesture classification dataset. That relevance is an implementation judgment, not evidence of improved gesture accuracy.

[Neurohive's English article dated 2 June 2019](https://neurohive.io/en/news/efficientnet-achieving-state-of-the-art-results-by-systematically-scaling-deep-neural-networks/) discusses this exact method and links the primary paper/Google explanation. Located through the site's public WordPress search API, after ordinary search missed it. This supplies a relevant Neurohive citation; however EfficientNet is 2019 historical SOTA, not demonstrated 2026 SOTA. If “modern” in the instructor's actual requirements means a recent year, eligibility needs clarification or a newer candidate. The available local plan permits other modern methods but does not supply an explicit year cutoff.

[Keras's primary transfer-learning example](https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/) documents `EfficientNetB0(include_top=False, weights='imagenet')`, pretrained feature extraction and embedded normalization. Feed RGB float32 pixels in [0,255]; do not add MobileNet's [-1,1] scaling. Context7 `/keras-team/keras-io` retrieved this guidance before coding.

## Actual pilot

Acceptance record was written before execution: `artifacts/method/pilot_efficientnet_acceptance.md`.
Command: `.venv/bin/python -m scripts.sota_pilot_efficientnet > artifacts/method/pilot_efficientnet.log 2>&1`. Exit 0.

- Python project environment: TensorFlow CPU 2.21.0, Keras 3.15.1. Seed 123. CPU four intra-op/two inter-op threads.
- Official Keras ImageNet no-top weights: https://storage.googleapis.com/keras-applications/efficientnetb0_notop.h5 . Download 16,705,208 bytes.
- Weights MD5: `50bc09e76180e00e4465e1a485ddc09d`; SHA256: `0683edcf6151811fb3ddd6f098ac48df644d74592c4919a5acdee1c37554add2`. Loader hash and independently recomputed hashes match.
- Exact implementation: Keras 3.15.1 `keras/src/applications/efficientnet.py`, SHA256 `785242004ccdd20df8ac65cd1f9810eb0e79ed597586b4f4e33c931cb9829c0b`.
- Three cleaned inputs, one per class; input hashes in `pilot_efficientnet.json`. No evaluation split was consulted or accuracy measured.
- Real pretrained feature output `(3,1280)`, finite and differing across images. Backbone 4,049,571 parameters. One disposable three-class dense-head training batch finite. Full model save/reload agrees at atol=1e-5, rtol=1e-4.
- Load/download about 2.34 s; first three-image batch about 0.31 s; three warmed single-image measurements 0.149–0.184 s. These are feasibility observations, not report-grade latency (requires the full defined protocol).
- Weights cached at `artifacts/method/pilot_keras_cache/models/efficientnetb0_notop.h5`. Disposable trained model removed with its temporary directory. No final checkpoint or training dataset modifications.

## License and delivery

[Keras 3.15.1 source](https://github.com/keras-team/keras/blob/v3.15.1/LICENSE) and [original TensorFlow TPU repository source](https://github.com/tensorflow/tpu/blob/master/LICENSE) are Apache-2.0. The official Keras loader exposes the ImageNet weights publicly for its transfer-learning workflow. A standalone license statement specifically attached to the downloaded H5 was not established here; do not silently infer redistribution rights from the code license. The simplest reproducibility package references the official download and checksum and excludes the cached weights pending publication review. No publication occurred.

## Proposed integration, not frozen experiment design

Use B0 original simple CNN, B1 original MobileNetV2, S1 ImageNet-pretrained EfficientNetB0, with identical cleaned split/resize/augmentation and documented budgets. A controlled ablation can compare S1 to the same EfficientNetB0 architecture initialized randomly, keeping all training settings and new classifier-head initialization controlled; this requires an additional A0 run, since B1 is not a valid one-factor pretraining ablation. Alternatively pretrained frozen versus fine-tuned EfficientNet is feasible if the fine-tuning factor is explicit. Select on validation only.

Synthetic share would be zero; P07/P10 generator wording must be replaced before final execution. Existing training preflight hardcodes B1/S1 MobileNetV2 and requires synthetic S1; it must be updated with tests for the real branch. The pilot's nested dense-head structure is not directly compatible with the current Grad-CAM engine's last-Dense expectation; final production model should use a top-level Dense head, or adapt/verify explanations. No code outside the isolated pilot was changed.

Remaining G03 decisions: eligibility review of 2019 alternative, confirmed weights publication terms if redistributing, selected method and frozen controlled budget/ablation contract. Technical download/inference feasibility is no longer a blocker for this candidate.
