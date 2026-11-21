# Attribution

## Sample video (media/demo_sameview.mp4, media/demo_crossview.mp4)

Derived from:

> Zheng, QingWei (2025). *Taekwondo Poomsae (8 Forms) — Front & Side Videos with 12-Keypoint Pose
> Annotations (MediaPipe)*. Mendeley Data, V1. <https://doi.org/10.17632/bjy7vr4xkt.1>
> National Kaohsiung University of Science and Technology. Licensed CC BY 4.0.

**What was changed.** The clips are not redistributed raw. For each of the two demo videos:

- the frontal (or lateral) source clip was decoded at its native 30 fps;
- 12 body keypoints per frame were re-estimated with MediaPipe Pose Landmarker (`full` and
  `heavy`, float16) rather than using the dataset's CSV annotations directly — the *disagreement
  between those two estimators* is itself a measured quantity in `out/uncertainty.json`;
- 9 joint angles were computed from those keypoints (hip-centred, torso-scaled), aligned to the
  reference perform­ance by DTW, and compared per movement;
- an overlay was drawn on top of the source frames: skeleton, per-joint deviation in degrees,
  the verdict band, and a burned-in summary; audio was dropped and frames were re-encoded to
  H.264 at 30 fps (768 frames, 25.6 s).

No claim of endorsement by the dataset's author or institution is made. The CC BY 4.0 licence
permits this derivative; attribution is provided here and in the README.

## Method and prior art

Discussed and cited in [docs/PRIOR-ART.md](docs/PRIOR-ART.md). No third-party code was vendored: the
pipeline is written against MediaPipe's public Python API, numpy, pandas and OpenCV.

## Not included

The 94-athlete labelled dataset used for the 8-class result is **not** redistributed — see
[docs/DATA.md](docs/DATA.md) §3. Only the numbers derived from it are published.
