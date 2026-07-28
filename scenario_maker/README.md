# scenario_maker

A live matplotlib GUI for building NMPC scenarios by hand: click to place a
start point, an end point, and any number of intermediate waypoints between
them; drag to "paint" circular obstacles at a coordinate with a chosen
radius. Saves everything to `scenario.json`.

Deliberately zero-dependency on the rest of the repo, same as
`mpc_visualization/`: no CasADi, no Acados, no `nmpc/` imports. Nothing in
`nmpc/` reads this file yet — it's purely an authoring tool for now.

## Running it

```bash
python scenario_maker/scenario_editor.py
```

Optional flags:

- `--out PATH` — where to save/load (default: `scenario_maker/scenario.json`)
- `--load` — load an existing file at `--out` on startup instead of starting blank
- `--sim-time` / `--u-init` — initial values for the two scalar fields (editable in the GUI too)

## Controls

The **Mode** radio button on the right picks what a left-click on the map does:

- **Start** — click to place/replace the start point (green circle)
- **Waypoint** — click to append an intermediate waypoint (orange diamond, numbered in click order)
- **Goal** — click to place/replace the end point (red star)
- **Obstacle** — click-and-drag: press sets the obstacle center, drag out sizes its radius live, release finalizes it. A plain click (no drag) drops one at the "Default Obstacle R" size instead.
- **Remove** — click near any existing point or obstacle to delete the nearest one (within a click tolerance that scales with the current zoom level)

Other widgets:

- **Sim Time (s)** / **Initial Surge u (m/s)** — text boxes for the two scalar fields that go into `mmg_init`/`sim_time`
- **Default Obstacle R (m)** — radius used for a no-drag obstacle click
- **Save scenario.json** — writes the current map to `--out` (requires both Start and Goal to be set)
- **Load** — reloads `--out` into the editor, replacing the current map
- **Undo** — reverts the last placement/removal (single-level stack per action)
- **Clear All** — wipes start/goal/waypoints/obstacles

Standard matplotlib pan/zoom toolbar buttons work as usual for navigating a
larger map before placing points.

## Axis convention

Matches `mpc_visualization/visualizer.py`: **X is North/Longitudinal**
(vertical on screen), **Y is East/Lateral** (horizontal on screen). All
points are stored and saved as `(x, y)` tuples in that order.

## `scenario.json` format

Same shape as one entry of `SCENARIOS` in `nmpc/run_live.py`
(`waypoints` / `mmg_init` / `sim_time`), plus an `obstacles` key:

```json
{
  "waypoints": [[0.0, 0.0], [30.0, 0.0], [30.0, 30.0], [0.0, 30.0], [-10.0, -10.0]],
  "mmg_init": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
  "sim_time": 600.0,
  "obstacles": [[15.0, 10.0, 2.5], [5.0, 20.0, 1.0]]
}
```

- `waypoints` — `[start, ...intermediate waypoints in click order..., goal]`, each `[x, y]` in meters
- `mmg_init` — `[u, v, r, x, y, psi]`; `x`/`y` are auto-synced to the start point, `u` comes from the "Initial Surge" box, everything else defaults to 0
- `sim_time` — seconds, from the "Sim Time" box
- `obstacles` — `[[x, y, radius], ...]` in meters, matching the `(ox, oy, orad)` triples `nmpc/path_following.py`'s `pad_obstacles()` already expects — not read by `nmpc/run_live.py` yet

## Dependencies

`numpy`, `matplotlib` only.
