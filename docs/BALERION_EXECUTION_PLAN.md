# Project Balerion — Hackathon Execution Plan

Rebuild from scratch in a 6-hour window (no pre-built repo reused). This doc is the
single source of truth for the team — read the Interface Contracts section first,
since a mismatch there is what usually kills a 4-person hackathon build at
integration time.

## Team & Tracks

| Track | Owner | Responsibility |
|---|---|---|
| **A — Model** | Member with nuScenes dataset | Run/verify PointNet++ inference, export cached `.npz` predictions |
| **B — Preprocessing** | 3rd member | Load raw frames, ego-centered coordinate handling, 60m clipping, hand raw points to the perception layer |
| **C — Dashboard/UI** | 4th member | Matplotlib dashboard shell, panel layout, live metrics wiring, event log |
| **D — Risk/Integration (you)** | Advait | Adaptive2.5D cell rendering, hazard simulation, risk engine, TTC/safety logic, final integration lead |

**Integration lead = you (Track D).** You own merging the other three tracks together at each checkpoint below — this is the role most likely to catch schema mismatches before they become 5pm fires.

## Repo Structure

```
balerion/
├── data/
│   └── predictions/           # cached .npz files (Track A output)
├── preprocessing/
│   └── loader.py              # Track B: frame load, clip, coordinate handling
├── model/
│   └── infer.py                # Track A: PointNet++ inference → .npz export
├── perception/
│   ├── adaptive_2_5d.py        # Existing Adaptive2_5DConverter (DO NOT rewrite)
│   └── cells.py                # cell rendering helpers
├── demo/
│   ├── render_bev.py           # BEV rendering (points/cells/rings/ego/trajectory)
│   ├── hazard_sim.py           # velocity-based hazard state + tick stepping
│   ├── risk_engine.py          # risk score, TTC, safety action
│   └── dashboard.py            # Track C: full dashboard assembly
├── docs/
│   └── BALERION_EXECUTION_PLAN.md   # this file
└── README.md
```

## Interface Contracts (agree on these in the first 15 minutes, before anyone writes code)

These are the exact handoff points between tracks. Mismatches here are the #1 cause of last-hour integration failure — nail them down verbally as a group before splitting up.

**A → B/D (prediction file format):**
```
points       : (N, 5) float32   — columns: x, y, z, intensity, ring (or whatever col 4/5 are — confirm)
model_points : (N, 5) float32
labels       : (N,) int64       — 0=Drivable, 1=Static, 2=Dynamic
source_file  : str
```

**B → D (coordinate convention — do not deviate from this):**
```
+X = forward/backward (ego heading)
+Y = left/right
+Z = height
Ego already at origin (0,0) — no additional transform needed
Ego self-returns already stripped before handoff
```

**D → C (per-tick dashboard data contract):**
```
{
  distance: float,
  risk_score: float,          # 0-1
  risk_level: str,             # LOW/MEDIUM/HIGH/CRITICAL
  ttc: float | "SAFE",
  action: str,                 # PROCEED/SLOW DOWN/BRAKE
  refinement_active: bool,
  cells_before: int,           # local region, pre-refinement
  cells_after: int,            # local region, post-refinement
  total_cells: int,
  baseline_cells: int,         # fixed-fine-resolution equivalent for comparison
}
```
Track C builds the dashboard against a **mock version of this dict** immediately, without waiting for D's real risk engine — this is what lets tracks run in parallel instead of serially blocking each other.

## Phase-by-Phase Timeline

### Phase 0 — Setup (0:00–0:15)
- Create repo, push skeleton folder structure above
- Whole team agrees on the three interface contracts above out loud — do not skip this, it's the cheapest insurance in the whole plan
- Create feature branches: `track-a-model`, `track-b-preprocess`, `track-c-dashboard`, `track-d-risk`
- **You (D):** create a small synthetic mock `.npz` (few hundred random points, all 3 labels, matching the schema) so Track C and your own Track D work can start immediately without waiting on Track A/B's real output

### Phase 1 — Parallel Build I (0:15–2:00)
- **Track A:** Verify PointNet++ inference runs on at least 1 real frame; export to the agreed `.npz` schema. Target: 3-5 verified frames minimum, more if time allows.
- **Track B:** Frame loading + coordinate verification (confirm +X forward, ego at origin, self-returns stripped) + 60m radius clipping filter with a printed before/after count.
- **Track C:** Build dashboard shell against the mock dict (BEV panel placeholder + text metrics panel + baseline/Balerion cell-count row + event log placeholder). Layout only — real data wiring comes later.
- **Track D (you):** Against the synthetic mock `.npz`: semantic BEV render → adaptive rings overlay → actual `Adaptive2_5DConverter` cell rendering (not raw points) → ego marker + predicted-path trajectory line.

**Known gotchas to skip past (already debugged once, don't re-discover):**
- Cell *centers* can land just past the 60m boundary even after point-level clipping — filter cells by center radius explicitly, not just the raw points.
- Matplotlib's default `scatter(x,y)` will render forward sideways if X=forward — plot as `scatter(-y, x)` so forward renders "up."
- Don't hardcode duplicate tier resolution constants (0.25/0.5/1.0m) — read them from `Adaptive2_5DConverter`'s own config so a later change to it doesn't silently desync your rendering.

### Checkpoint 1 (2:00) — Integration
- Swap Track D's synthetic mock for Track A's first real verified frame + Track B's clipped/verified points
- Confirm the whole render pipeline (BEV → rings → cells → ego) works against real data with no schema surprises
- If something breaks here, **fix the contract mismatch immediately** — don't let two tracks diverge on their own interpretation of the schema for another hour

### Phase 2 — Parallel Build II (2:00–3:30)
- **Track A:** Continue generating the full cached prediction set across more frames (buffer for later, not required for the core 90s demo)
- **Track B:** Support Track D on any coordinate/alignment issues surfaced at Checkpoint 1; otherwise free to help Track C wire real data into the dashboard shell
- **Track C:** Wire the dashboard's mock dict inputs over to real (still risk-engine-less) values — real point counts, real cell counts — so only the risk/TTC fields remain mocked
- **Track D (you):** Hazard simulation (`{x,y,vx,vy}`, tick-based stepping) + risk engine + local resolution refinement

**Use these pre-derived formula/constants directly — do not re-derive live, we already found and fixed the bugs in these:**
```
Outside corridor (|y| >= 2.5m):
  base_risk = clamp(1 - dist/50, 0, 1)
  closing_lateral_speed = -vy * sign(y)
  closing_factor = 0.2 if closing_lateral_speed <= 0 else min(1.2, closing_lateral_speed / 1.8)
  lat_factor = clamp(1 - max(0, |y|-2.5)/8, 0.2, 1.0)
  final_risk = min(base_risk * closing_factor * lat_factor, 0.58)   # cap, not 0.60 — safety margin

Inside corridor (|y| < 2.5m, 0 < x <= 40):
  base_risk = clamp(1 - dist/50, 0, 1)
  vx_toward_ego = -vx if vx < 0 else 0
  closing_factor = 1 + vx_toward_ego / 4.0
  final_risk = clamp(base_risk * 1.15 * closing_factor, 0, 1)   # NO floor — a hardcoded floor here breaks the LOW→MEDIUM→HIGH→CRITICAL narrative, we found this the hard way

Thresholds: LOW [0,0.35) / MEDIUM [0.35,0.6) / HIGH [0.6,0.85) / CRITICAL [0.85,1.0]

Refinement: trigger at risk >= 0.60, radius 5.0m around hazard, re-bin from RAW SOURCE POINTS
(not existing converter cells) at the fine tier's cell size — this is what makes it genuinely
finer rather than just subdividing already-coarse data.
```

**Critical scenario-design constraint (solve this on paper before wiring the UI, budget 15 min):**
With `closing_factor≈1.25`, HIGH only occupies roughly a 20–29m distance band; below ~20m it's already CRITICAL. A demo hazard needs to (a) enter the corridor with distance still in the low-to-high 20s, and (b) stay inside the corridor long enough (i.e. slow enough lateral velocity relative to forward velocity) for x to drop under ~20m before it laterally exits — otherwise it either skips straight to CRITICAL or never reaches it. We hit this exact bug twice already; pick your hazard's start position and velocity with this math in mind, don't just eyeball it.

### Checkpoint 2 (3:30) — Integration
- Track D's risk engine output wired into Track C's dashboard, replacing the last mocked fields
- Full dry run: load real frame → render → introduce hazard → watch risk/TTC/action update live in the dashboard
- Verify the chosen hazard scenario actually visits all four risk levels (LOW→MEDIUM→HIGH→CRITICAL) with enough dwell time in each for a judge to read it — if HIGH flashes for one frame, adjust velocity/dt, don't ship it as-is

### Phase 3 — Safety Layer + Polish (3:30–4:30)
- **Track D:** TTC (path-relative — pre-corridor: time to lateral entry; in-corridor: `x / closing x-speed`, displayed as "SAFE" not raw infinity/negative), map risk→action (LOW/MEDIUM→PROCEED, HIGH→SLOW DOWN, CRITICAL→BRAKE)
- **Track C:** Event log (plain text, append on each state transition), baseline-vs-Balerion panel using **measured** cell counts (baseline = fixed fine-resolution equivalent cell count for the same frame extent, Balerion = actual live converter+refinement cell count — both computed, never invented)
- **Track A/B:** Free capacity — help stress-test with 2-3 different real frames, or start on the optional multi-frame replay (Section 24 of the spec) only if everything else is solid; treat this as a stretch goal, not a requirement

### Phase 4 — Freeze & Rehearse (4:30–6:00)
- **4:30–5:00:** Bug-fixing buffer only — no new features past this point
- **5:00–5:30:** Full run-through of the 60-90s judge script (below) at least twice, out loud, against the live demo
- **5:30–5:45:** Record a screen-capture backup video of a clean full run — insurance against a live-demo failure in front of judges
- **5:45–6:00:** Final freeze — no code changes after this point, just script rehearsal

## Cut List (in order, if you're behind schedule)

1. Multi-frame real sequence replay (Section 24 / Stage 8.1) — a single well-chosen frame is enough for a 90s demo
2. Technical backup screen (Section 19) — nice for Q&A depth, not required for the core demo
3. ADD VEHICLE / ADD PEDESTRIAN as separate buttons — collapse to a single INTRODUCE HAZARD trigger
4. Real interactive GUI buttons — a keypress-advance or scripted animation reads as "live" to a judge and costs far less time than click-handler wiring

**Do not cut:** the risk progression through all 4 levels, the local refinement visual (COARSE→FINE around the hazard), TTC/action display, and the baseline-vs-Balerion measured comparison — these five things are what Section 21 of the spec says must be visibly proven.

## 60–90 Second Judge Script (unchanged from spec — rehearse this, don't rewrite it live)

1. **0–10s Hook:** "This is Project Balerion, our adaptive LiDAR perception system. Instead of treating every point equally, we allocate detail according to where it matters."
2. **10–25s Normal Scene:** point at near/medium/far rings — "The near field gets fine resolution, while farther regions are intentionally coarser."
3. **25–45s Hazard:** press INTRODUCE HAZARD — "Now I am introducing a moving vehicle into the ego vehicle's path." Show Risk: LOW → HIGH.
4. **45–65s Adaptive Response:** "As the object becomes relevant, the risk increases and Balerion increases resolution only around the critical region." Show Resolution: COARSE → FINE.
5. **65–75s Safety:** Show TTC + action. "The refined local representation preserves the important obstacle and supports the safety response."
6. **75–90s Close:** "We are not trying to process more LiDAR. We are trying to process the right LiDAR, at the right resolution, at the right time."
