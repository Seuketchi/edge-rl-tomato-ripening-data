# Edge-RL: Tomato Ripening — Data & Reproducibility

Public data, code, and manuscript companion for the thesis

> **Edge-RL: Autonomous Post-Harvest Tomato Ripening Control on a \$50 Edge Device**
> Tristan O. Jadman — Department of Computer Engineering, MSU-IIT

A Deep Q-Network policy is trained in a physics-based digital twin, distilled to
a 5,443-parameter MLP, and deployed on an ESP32-S3 microcontroller that drives
active heating and cooling. This repository holds the **citable record** of the
two physical hardware demonstration runs and everything needed to reproduce the
hardware-run figures and the ripening timelapse from the raw telemetry.

The **firmware and simulation source code** live in the companion repository:
<https://github.com/Seuketchi/edge-rl-tomato-ripening>.

## Contents

| Path | What it is |
|------|------------|
| `Manuscript.pdf` | The compiled thesis manuscript. |
| `data/demo-2026-06-10.csv` | **Run 1** telemetry — five-day run (RGB565 path), 7,418 rows. |
| `data/demo-2026-06-22.csv` | **Run 2** telemetry — continuous re-run (RGB888 path), 7,045 rows. |
| `data/README.md` | Column schema, cleaning rules, and run details. |
| `scripts/` | Cleaning and figure/timelapse generation scripts. |
| `assets/` | Calibration frames and bench photographs used by the timelapse. |
| `media/ripening_timelapse.mp4` | The Run 2 ripening timelapse (chamber camera vs. true-colour bench). |

Each row of telemetry is a once-per-minute sample; column meanings are documented
in `data/README.md`.

## Reproduce the figures

```bash
python -m pip install -r scripts/requirements.txt

python scripts/plot_demo_run.py    # -> figures/demo_run.png            (Run 1)
python scripts/plot_run2.py        # -> figures/run2_perceived_panels.png (Run 2)
python scripts/make_timelapse.py   # -> media/ripening_timelapse.mp4     (timelapse; needs ffmpeg)
```

The figures are regenerated deterministically from `data/`; they are not checked
in (see `.gitignore`).

## Citing

Please cite the thesis and reference this repository for the data. For an exact,
reproducible reference, pin a commit hash or release tag, e.g.:

> T. O. Jadman, *Edge-RL: Autonomous Post-Harvest Tomato Ripening Control on a
> \$50 Edge Device*, MSU-IIT, 2026. Data and code:
> `https://github.com/Seuketchi/edge-rl-tomato-ripening-data` (commit `<hash>`).

## License

- **Code** (`scripts/`): [MIT](LICENSE).
- **Data, manuscript, and media** (`data/`, `Manuscript.pdf`, `media/`, `assets/`):
  [CC-BY-4.0](data/LICENSE) — free to share and adapt with attribution.

© 2026 Tristan O. Jadman.
