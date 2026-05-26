# video_sync — User Guide

Synchronize screen recordings of the racing gate experiment with the live
replay data, and render annotated clips with real-time metric overlays.

---

## 1. Install

```bash
cd services/video_sync
uv sync
```

This installs OpenCV + pandas and links to `workload_inference` (used to
load trial data).

---

## 2. What the tool does

For one subject (e.g. `CXOE`), `video_sync`:

1. Finds the screen recording (`Recording_CXOE.mp4`) and loads all racing trials.
2. Aligns video frames with data timestamps using one or two sync points.
3. Renders either a **single trial clip** or a **crash montage** across all trials.

All overlays are drawn as a horizontal bar at the top of the video. The bar
is opaque, so it doubles as a cover for the Unity title / toolbar visible
in the original recording.

Overlays available:

| Name | Shows |
|---|---|
| `workload` | Inferred workload state (Low/Med/High) + probabilities |
| `drones` | Alive count + cumulative crashes |
| `adaptation` | Current adaptation step (renamed from CWL step) |
| `gates` | Gates passed / total + progress bar (deduped) |
| `timeline` | Race timer T+MM:SS.cs, anchored to first plane crossing |

---

## 3. Step 1 — Find the sync offsets

The tool aligns video time with data time via **anchor events**:

- **START anchor** (`--sync-offset`): when trial 1 begins in the video
- **END anchor** (`--sync-offset-end`, optional): when the last trial's last
  gate is passed — gives you a second point so we can compensate for fps
  drift between recording and data clocks.

Start by listing the timing of every event:

```bash
uv run video_sync --subject CXOE --print-sync-info
```

Example output:

```
Video: Recording_CXOE.mp4  1920x1080  30.00 fps  72050 frames
Anchor mode: trigger-plane
START anchor (trial 1, trigger-plane):    1716210576120 ms (Unix)
END   anchor (last-gate of last trial):   1716212954820 ms (Unix)
  -> span between anchors: 2378.70 s

Per-trial offsets relative to trial-1 anchor (assuming --sync-offset 0):
  trial                   t0    1st gate   last gate
  trial  1 trial_1     -3.28s    +0.00s    +148.42s
  trial  2 trial_2   +309.17s  +312.45s    +461.10s
  ...

Detected 14 crash events across all trials
```

Then in the video:

1. Scrub to the frame where trial 1's **race timer ticks to 00:00** — note
   that time as `--sync-offset`.
2. (Optional) scrub to the frame where the **last drone passes the last
   gate of the last trial** — note that time as `--sync-offset-end`.

Both points together give the most accurate alignment.

---

## 4. Anchor modes

The default `--anchor trigger-plane` estimates when the swarm centroid
crosses an invisible plane just in front of gate 1 — typically the moment
the on-screen race timer triggers. Tune the plane distance via
`--trigger-distance` (meters before gate 1, default 2.0).

Other choices:

- `--anchor first-gate` — anchor on the first gate's `first_pass_timestamp`
  (drones already touching the gate plane)
- `--anchor t0` — anchor on data-logger start (drones spawn)

The `timeline` overlay always uses the same anchor as the sync, so the
on-screen `T+` matches your reference event.

---

## 5. Step 2 — Render a single trial

```bash
uv run video_sync \
  --subject CXOE \
  --trial 3 \
  --sync-offset 12.5 \
  --sync-offset-end 2391.2 \
  --trim
```

What this does:
- Aligns video using two sync points (more accurate over long recordings)
- Renders trial 3 only, trimming to the trial's data window
- Default output: `<subject_dir>/CXOE_trial_3.mp4`

Pick specific overlays:

```bash
uv run video_sync --subject CXOE --trial 3 --sync-offset 12.5 \
  --overlays workload drones timeline
```

Custom output path:

```bash
uv run video_sync --subject CXOE --trial 3 --sync-offset 12.5 \
  --output ~/Desktop/cxoe_t3.mp4
```

---

## 6. Step 3 — Build a crash montage

```bash
uv run video_sync \
  --subject CXOE \
  --montage \
  --sync-offset 12.5 \
  --sync-offset-end 2391.2 \
  --padding 3.0
```

What this does:
- Detects every drone `alive 1→0` transition across all trials
- Extracts ±`padding` seconds around each crash
- Adds a title card before each clip ("Clip 3/14 — trial_2 — Crash at T+45.2s")
- Concatenates clips into one video
- Default output: `<subject_dir>/CXOE_crash_montage.mp4`

Merge clips when several drones crash together:

```bash
uv run video_sync --subject CXOE --montage --sync-offset 12.5 \
  --merge-window 1.0
```

Crashes within 1 second of each other become a single clip.

---

## 7. All flags

| Flag | Default | Purpose |
|---|---|---|
| `--subject CODE` | required | Subject directory name (e.g. `CXOE`) |
| `--data-dir PATH` | `workload_inference/data/experiments/experiment_racing_gates` | Root experiment folder |
| `--sync-offset SECONDS` | `0.0` | Video time of START anchor (trial 1) |
| `--sync-offset-end SECONDS` | — | Video time of END anchor; enables 2-point drift correction |
| `--anchor {trigger-plane,first-gate,t0}` | `trigger-plane` | START anchor event |
| `--trigger-distance METERS` | `2.0` | Plane distance before gate 1 for `trigger-plane` |
| `--trial N` | — | Render trial N only (1-based) |
| `--montage` | off | Build crash montage instead |
| `--padding SECONDS` | `3.0` | Clip half-window for montage |
| `--merge-window SECONDS` | `0.5` | Merge crashes within this window |
| `--trim` | off | Trim single-trial output to data window |
| `--overlays ...` | all 5 | Subset of overlays to render |
| `--output PATH` | auto | Output file path |
| `--print-sync-info` | off | Print timing info and exit |
| `--verbose` / `-v` | off | More logging |

---

## 8. Tips

- **Audio is dropped.** OpenCV writes video only.
- **Output codec**: H.264 (`mp4v`). Plays in VLC, PowerPoint, QuickTime, etc.
- **Two-point sync** absorbs constant fps drift across the recording. If a
  trial in the middle still looks off, the recording probably has a
  discontinuity (pause/resume) — split it into two clips and sync separately.
- **Trigger-plane heuristic** can fail if the centroid never approaches
  gate 1 (e.g. corrupted trial). The tool falls back to `first-gate` and
  prints a warning.
- **Gate duplicates** in the raw CSVs are deduplicated by `id` in the
  overlay; the displayed count reflects unique gates.

---

## 9. Batch mode

Process many subjects at once via a YAML config (see
[`batch_adaptive.yaml`](batch_adaptive.yaml)).

```bash
uv run video_sync --batch batch_adaptive.yaml
```

The config has four sections:

```yaml
data_dir: ...                # input experiment folder
output_dir: ...              # where per-subject mp4s land

defaults:                    # shared settings (overlays, padding, etc.)
  anchor: trigger-plane
  padding_s: 3.0
  merge_window_s: 0.5
  do_trials: true
  do_montage: true
  trim: true

subjects:                    # per-subject sync points
  - code: CXOE
    sync_offset: 12.5        # leave as null to skip
    sync_offset_end: 2391.2  # leave as null for 1-point sync
```

For each subject the batch script writes:

```
output_dir/<CODE>/<CODE>_trial_1.mp4 ... trial_N.mp4
output_dir/<CODE>/<CODE>_crash_montage.mp4
```

If `sync_offset` is left as `null` the subject is skipped (with a log line).
Failures on individual subjects are logged and don't stop the rest of the batch.

---

## 10. Quick reference

```bash
# 1. Check timing
uv run video_sync --subject CXOE --print-sync-info

# 2. Render one trial (two-point sync for accuracy)
uv run video_sync --subject CXOE --trial 2 \
  --sync-offset 12.5 --sync-offset-end 2391.2 --trim

# 3. Build crash montage
uv run video_sync --subject CXOE --montage \
  --sync-offset 12.5 --sync-offset-end 2391.2 --padding 3
```
