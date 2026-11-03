# Gesture dataset

The A1 collection contains 464 JPEGs: 151 peace, 163 stop and 150 thumbs_up. Images are 640 × 480. Original files and models remain preserved with SHA-256 checks in artifacts/a1_original/ and artifacts/audit/a1_inventory.json.

## Cleaning

Collection labels were burned into the top of each original image. A fixed top-20% crop removes 96 rows from every image, regardless of class. Derived RGB PNGs are 640 × 384 and are resized to 224 × 224 for the model. Apply this crop exactly once to raw input.

Visual review excluded 11 stop images, stop_0152 through stop_0162, because no stop gesture was visible. They were neither deleted nor relabeled. The retained dataset has 453 images. Some blur, occlusion and cropped fingertips remain. There are no synthetic training images.

## Frozen evaluation split

The final manifest is data/manifests/a1_limited_v2.json. All four experiments use the same image IDs.

| Class | Train | Validation | Test |
|---|---:|---:|---:|
| Peace | 105 | 23 | 23 |
| Stop | 136 | 12 | 4 |
| Thumbs up | 103 | 23 | 24 |
| Total | 344 | 58 | 51 |

All 38 connected components of the original dHash64 similarity graph (distance at most 4), including components spanning labels, remain within one split. Known similarity edges and exact/source duplicate hashes have no cross-split overlap. Assignment was deterministic and did not use model outcomes.

Original capture sessions are unknown. The data depicts one main participant in one indoor setting. Similarity groups do not establish independent people, sessions or environments. Test support for stop is only four, so its estimate is especially uncertain. These results support a limited within-setting comparison; they do not establish generalization to new people.

## Consent and access

The user confirmed that the teammate and all visible background classmates mutually consented to course project use. This is recorded in artifacts/audit/consent_scope.json. Public redistribution has not been authorized. Keep identifiable images and private model/data archives restricted to permitted course review.

The user explicitly deferred expansion. No additional pictures are needed for this agreed evaluation. New participants and independently staged sessions would be needed for stronger generalization claims.

Raw collection now saves frames before adding display labels and uses unique session directories. The operator must supply truthful consent metadata. The deployed app processes uploads in memory without saving them.

## Final experiments

B0 (CNN), B1 (MobileNetV2), S1 (pretrained ConvNeXtTiny) and A0 (random frozen ConvNeXtTiny) are complete. Results are in artifacts/evaluation/comparison.json. Historical A1 figures from labeled full-frame images are not directly comparable with these cleaned-data runs.
