# Source provenance

Architecture and production-shape facts are pinned to OpenDriveLab/UniAD commit
`609ee083ea51c3521c323f1279dfc4cee0e60467` and its released nuScenes stage-2
configuration. The upstream source is Apache-2.0:

- repository: https://github.com/OpenDriveLab/UniAD
- pinned tree: https://github.com/OpenDriveLab/UniAD/tree/609ee083ea51c3521c323f1279dfc4cee0e60467
- commit identity (SHA-256):
  `04dc11a82f7de530bd86fdac9bda0b4b65298e38d9732b74248ba84cf82b15a5`
- CVPR 2023 paper:
  https://openaccess.thecvf.com/content/CVPR2023/html/Hu_Planning-Oriented_Autonomous_Driving_CVPR_2023_paper.html
- supplementary material:
  https://openaccess.thecvf.com/content/CVPR2023/supplemental/Hu_Planning-Oriented_Autonomous_Driving_CVPR_2023_supplemental.pdf

The SHA-256 above is the digest of the ASCII commit identifier, used as the
stable profile provenance key. `tools/verify_provenance.py` checks it without
network access. No upstream code, checkpoint, config file, dataset sample, or
trademark asset is copied into this repository.

The implementation-level findings used by the technical article are summarized
in `evidence/uniad-architecture-study.md`. That note separates upstream
architecture facts from the executable synthetic C profile.

The interaction system was informed by three public explanatory-design
precedents:

- Bartosz Ciechanowski, [Mechanical Watch](https://ciechanow.ski/mechanical-watch/)
  for direct manipulation and stable one-variable experiments;
- Distill, [Why Momentum Really Works](https://distill.pub/2017/momentum/)
  for placing interaction at the point of inquiry and keeping interpretation
  adjacent to the changing result;
- The Pudding, [How to implement scrollytelling](https://pudding.cool/process/how-to-implement-scrollytelling/)
  for observing scroll without hijacking it.

The resulting audit, design grammar, storyboard, and acceptance criteria are
recorded in `evidence/interaction-design-study.md`.
