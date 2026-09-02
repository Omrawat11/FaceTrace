# FaceTrace Demo Dataset Specification

This directory contains controlled, ethically generated demo images for local search and face-matching validation in Phase 2B.

## Image Manifest

| Filename | Description | Expected Pipeline Outcome |
| :--- | :--- | :--- |
| `user_primary.jpg` | Baseline portrait of the consenting test subject (neutral expression, studio lighting). Used as the **Query Face**. | 1 face detected, generates 512D ArcFace query vector. |
| `user_match_01.jpg` | Same consenting subject (identical studio portrait). | **MATCH** (Cosine similarity $\approx 1.00$). |
| `user_match_02.jpg` | Same subject in casual outdoor cafe lighting, slight head tilt and smile. | **MATCH** (Cosine similarity $> 0.45$). |
| `user_match_03.jpg` | Same subject in conference setting wearing a tailored blazer. | **MATCH** (Cosine similarity $> 0.45$). |
| `non_match_01.jpg` | Unrelated person (elderly Caucasian professor with silver hair and beard). | **NO_MATCH** (Cosine similarity $< 0.45$). |
| `non_match_02.jpg` | Unrelated person (young East Asian woman in autumn park). | **NO_MATCH** (Cosine similarity $< 0.45$). |
| `multi_face_01.jpg` | Group photo containing two colleagues standing side-by-side. | **MULTI_FACE**: Evaluates all faces, records highest similarity. |
| `no_face_01.jpg` | Alpine mountain and lake landscape containing zero human faces. | **NO_FACE_DETECTED**: Similarity assigned $0.0$, skipped gracefully. |

## Custom Consented Images

To use your own photos:
1. Place your reference portrait as `data/demo_images/user_primary.jpg`.
2. Place candidate photos of yourself or other individuals as `user_match_XX.jpg` or `non_match_XX.jpg`.
3. The local pipeline automatically reads from this directory via `MockSearchProvider`.
