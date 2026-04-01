<div align="center">

<img src="docs/assets/unt-logo.svg" alt="University of North Texas" width="320">

<h1>Mind the Hitch: Dynamic Calibration and Articulated Perception for Autonomous Trucks</h1>

<p>
  <a href="https://arxiv.org/pdf/2603.23711" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/Paper-CVPR_2026-red" alt="Paper">
  </a>
  <a href="https://arxiv.org/abs/2603.23711">
    <img src="https://img.shields.io/badge/arXiv-2603.23711-b31b1b" alt="arXiv">
  </a>
  <a href="docs/DATA_DOWNLOAD.md">
    <img src="https://img.shields.io/badge/Dataset-STT4AT-green" alt="Dataset">
  </a>
  <a href="docs/DATA_DOWNLOAD.md">
    <img src="https://img.shields.io/badge/Checkpoint-dCAP-blue" alt="Checkpoint">
  </a>
</p>

**Morui Zhu, Yongqi Zhu, Song Fu, Qing Yang**

Official code release for the CVPR 2026 paper **"Mind the Hitch: Dynamic Calibration and Articulated Perception for Autonomous Trucks"**.

</div>

---

## Overview

`dCAP` addresses a core challenge in autonomous trucking: the tractor-trailer system is **articulated**, so cross-rig camera extrinsics are **time-varying** rather than fixed.

This repository provides:
- a dynamic trailer camera pose pipeline (`dcap.camera_pose`)
- an articulated downstream perception pipeline built on BEVFormer (`dcap.perception.bevformer`)
- the `STT4AT` benchmark and devkit assets for reproduction
- runnable training/evaluation scripts and released checkpoints

## News

- **2026-04-01**: 🎉 Code, checkpoints, and STT4AT dataset assets are released.
- **2026-04-01**: Paper is available on arXiv: [2603.23711](https://arxiv.org/abs/2603.23711).

## Resources

- [Installation](docs/INSTALL.md)
- [Data & Weights](docs/DATA_DOWNLOAD.md)
- [Train & Eval](docs/TRAIN_EVAL.md)

## Citation

```bibtex
@misc{zhu2026mindhitchdynamiccalibration,
  title={Mind the Hitch: Dynamic Calibration and Articulated Perception for Autonomous Trucks},
  author={Morui Zhu and Yongqi Zhu and Song Fu and Qing Yang},
  year={2026},
  eprint={2603.23711},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2603.23711},
}
```

## Acknowledgements

- [BEVFormer](https://github.com/fundamentalvision/BEVFormer)
- [VGGT](https://github.com/facebookresearch/vggt)
- [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive)

## License

MIT License. See [LICENSE](LICENSE).
