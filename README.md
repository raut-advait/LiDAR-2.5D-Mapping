# Project Balerion — Adaptive LiDAR Perception System

Project Balerion is a distance-aware and detail-aware adaptive LiDAR perception system built for real-time risk assessment, local resolution refinement, and dynamic safety decision-making.


## Repository Structure

```
balerion/
├── data/
│   └── predictions/           # Cached .npz prediction files
├── preprocessing/
│   ├── __init__.py
│   └── loader.py              # Frame loading, coordinate transformation, 60m clipping
├── model/
│   ├── __init__.py
│   └── infer.py                # PointNet++ inference & export pipeline
├── perception/
│   ├── __init__.py
│   ├── adaptive_2_5d.py        # Adaptive2_5DConverter (distance/detail-aware grid)
│   └── cells.py                # Cell rendering and aggregation helpers
├── demo/
│   ├── __init__.py
│   ├── render_bev.py           # Bird's-Eye-View (BEV) rendering pipeline
│   ├── hazard_sim.py           # Dynamic hazard simulation state & trajectory stepping
│   ├── risk_engine.py          # Risk scoring, TTC calculation, and safety actions
│   └── dashboard.py            # Real-time Matplotlib monitoring dashboard shell
├── docs/
│   └── BALERION_EXECUTION_PLAN.md
└── README.md
```

## Setup & Execution

Python 3.8+ with NumPy and Matplotlib.

See `docs/BALERION_EXECUTION_PLAN.md` for interface contracts, formulas, and integration checkpoints.
