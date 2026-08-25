"""controller_effort_logger: Reusable controller-effort / cost-breakdown logger.

Computes, per accepted NMPC solve, the individual Q-weighted state-error terms
and R-weighted input terms from the NMPC stage cost, plus standard
control-effort diagnostics (effort magnitude, rate/smoothness, total
variation, saturation activity, peak/RMS), and streams them to
costs_errors.csv inside an already-open ExperimentLogger run_dir.

Architecture: this class is designed to be fed exclusively via
log_sample_async() -- an O(1) non-blocking queue.Queue.put_nowait -- from
logger_node's ROS subscription callback for /nmpc/controller_effort_raw.
Every line of arithmetic below and every write to costs_errors.csv happens on
this class's own background thread, so the work is decoupled both from the
real-time NMPC control loop (nmpc_node/mmg_node, a separate process entirely)
and from logger_node's own per-tick telemetry.csv callback (a separate thread
within the same process). finalize() returns a summary fragment that
logger_node folds into the SAME run's summary.json (ExperimentLogger.finalize
writes it once, so there is no cross-process/cross-thread merge race).
"""
import collections
import csv
import math
import os
import queue
import threading
from typing import Dict, List, Optional, Sequence


def _wrap_to_pi(angle: float) -> float:
    """NumPy-free numeric twin of nmpc/path_following.py's wrap180_casadi,
    applied here to the psi row of dx only (see _process_sample) -- the same
    row nmpc_acados.py's own NONLINEAR_LS cost wraps before squaring."""
    return math.atan2(math.sin(angle), math.cos(angle))


class _RunningColumnStats:
    """min/max/sum/count for one CSV column, backing summary.json's per-term
    min/max/sum/avg. Note: for the *_cum columns (running totals that are
    monotonically non-decreasing across a run -- J_u_cum, J_du_cum, tv_cum_*,
    peak_cum_*, rms_cum_*), `max` doubles as that column's final/total value,
    since it can only grow; summing a cumulative column across rows on top of
    that would not be a meaningful additional number, so no separate
    final-value bookkeeping is kept beyond the plain max already computed here."""
    __slots__ = ("min", "max", "sum", "count")

    def __init__(self):
        self.min = math.inf
        self.max = -math.inf
        self.sum = 0.0
        self.count = 0

    def update(self, value):
        v = float(value)
        if v < self.min:
            self.min = v
        if v > self.max:
            self.max = v
        self.sum += v
        self.count += 1

    def as_dict(self) -> Dict[str, float]:
        avg = self.sum / self.count if self.count > 0 else 0.0
        return {
            "min": self.min if self.count > 0 else 0.0,
            "max": self.max if self.count > 0 else 0.0,
            "sum": self.sum,
            "avg": avg,
        }


class ControllerEffortLogger:
    """Streams costs_errors.csv into an already-open ExperimentLogger run_dir
    and accumulates the per-column min/max/sum/avg (plus saturation event
    counts and drop counts) that finalize() returns for logger_node to fold
    into that same run's summary.json."""

    def __init__(
        self,
        run_dir: str,
        state_names: Sequence[str],
        control_names: Sequence[str],
        q_diag: Sequence[float],
        r_diag: Sequence[float],
        r_delta_diag: Sequence[float],
        u_min: Sequence[float],
        u_max: Sequence[float],
        psi_index: int,
        saturation_tol: float = 1e-3,
        rolling_window: int = 50,
        queue_maxsize: int = 500,
        flush_every: int = 10,
    ):
        self.run_dir = run_dir
        self.state_names = list(state_names)
        self.control_names = list(control_names)
        self.q_diag = [float(v) for v in q_diag]
        self.r_diag = [float(v) for v in r_diag]
        self.r_delta_diag = [float(v) for v in r_delta_diag]
        self.u_min = [float(v) for v in u_min]
        self.u_max = [float(v) for v in u_max]
        self.psi_index = psi_index
        self.saturation_tol = saturation_tol
        self.rolling_window = max(1, rolling_window)
        self.flush_every = max(1, flush_every)
        self.n_state = len(self.state_names)
        self.n_control = len(self.control_names)

        self.csv_path = os.path.join(self.run_dir, "costs_errors.csv")
        self._csv_file = open(self.csv_path, mode="w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)
        self._header = self._build_header()
        self._csv_writer.writerow(self._header)
        self._csv_file.flush()

        self._col_stats: Dict[str, _RunningColumnStats] = {name: _RunningColumnStats() for name in self._header}

        # per-channel running/rolling state
        self._prev_u: Optional[List[float]] = None
        self._roll_u = [collections.deque(maxlen=self.rolling_window) for _ in range(self.n_control)]
        self._cum_u_sumsq = [0.0] * self.n_control
        self._cum_u_count = 0
        self._cum_u_absmax = [0.0] * self.n_control
        self._J_u_cum = 0.0
        self._J_du_cum = 0.0
        self._tv_cum = [0.0] * self.n_control
        self._sat_count_min = [0] * self.n_control
        self._sat_count_max = [0] * self.n_control
        self._sat_prev_at_bound = [False] * self.n_control
        self._sat_entries = [0] * self.n_control
        self._sat_exits = [0] * self.n_control

        self._local_drop_count = 0
        self._step_count = 0

        self._queue: "queue.Queue" = queue.Queue(maxsize=queue_maxsize)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._finalized = False

        print(f"[ControllerEffortLogger] costs_errors.csv -> {self.csv_path}")

    def start(self):
        self._thread.start()

    # ------------------------------------------------------------ header
    def _build_header(self) -> List[str]:
        header = ["step", "time_s", "dt", "solve_time_ms", "success"]
        header += [f"q_term_{name}" for name in self.state_names]
        for name in self.control_names:
            header += [
                f"u_{name}", f"absu_{name}", f"du_{name}", f"absdu_{name}",
                f"r_term_{name}", f"rdelta_term_{name}",
                f"sat_near_min_{name}", f"sat_near_max_{name}",
                f"tv_cum_{name}", f"peak_roll_{name}", f"rms_roll_{name}",
                f"peak_cum_{name}", f"rms_cum_{name}",
            ]
        header += [
            "q_sum", "r_sum", "J_u_step", "J_u_cum", "J_du_step", "J_du_cum",
            "upstream_drops", "local_drops",
        ]
        return header

    # ------------------------------------------------------------ ingest
    def log_sample_async(self, step, dt, u, x, x_ref, solve_time, success, upstream_drops) -> bool:
        """Non-blocking; call this from logger_node's ROS subscription callback.
        Returns False (and counts a local drop) if the worker has fallen behind
        and the internal queue is full -- never raises, never blocks the caller."""
        try:
            self._queue.put_nowait((step, dt, list(u), list(x), list(x_ref), solve_time, success, upstream_drops))
            return True
        except queue.Full:
            self._local_drop_count += 1
            return False

    # ------------------------------------------------------------ worker
    def _worker(self):
        while not self._stop_event.is_set():
            try:
                sample = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self._process_sample(*sample)
        # best-effort drain of whatever was still queued when stop() was requested
        while True:
            try:
                sample = self._queue.get_nowait()
            except queue.Empty:
                break
            self._process_sample(*sample)

    def _process_sample(self, step, dt, u, x, x_ref, solve_time, success, upstream_drops):
        n_c = self.n_control

        dx = [x[i] - x_ref[i] for i in range(self.n_state)]
        dx[self.psi_index] = _wrap_to_pi(dx[self.psi_index])
        q_term = [self.q_diag[i] * dx[i] ** 2 for i in range(self.n_state)]
        r_term = [self.r_diag[i] * u[i] ** 2 for i in range(n_c)]
        q_sum = sum(q_term)
        r_sum = sum(r_term)

        if self._prev_u is None:
            du = [0.0] * n_c
        else:
            du = [u[i] - self._prev_u[i] for i in range(n_c)]
        self._prev_u = list(u)

        rdelta_term = [self.r_delta_diag[i] * du[i] ** 2 for i in range(n_c)]
        J_u_step = r_sum
        J_du_step = sum(rdelta_term)
        self._J_u_cum += J_u_step
        self._J_du_cum += J_du_step

        abs_u = [abs(v) for v in u]
        abs_du = [abs(v) for v in du]

        near_min = [False] * n_c
        near_max = [False] * n_c
        peak_roll = [0.0] * n_c
        rms_roll = [0.0] * n_c
        peak_cum = [0.0] * n_c
        rms_cum = [0.0] * n_c

        for i in range(n_c):
            self._tv_cum[i] += abs_du[i]

            self._cum_u_sumsq[i] += u[i] ** 2
            self._cum_u_absmax[i] = max(self._cum_u_absmax[i], abs_u[i])

            self._roll_u[i].append(u[i])
            window = self._roll_u[i]
            peak_roll[i] = max(abs(v) for v in window)
            rms_roll[i] = math.sqrt(sum(v * v for v in window) / len(window))

            near_min[i] = u[i] <= (self.u_min[i] + self.saturation_tol)
            near_max[i] = u[i] >= (self.u_max[i] - self.saturation_tol)
            at_bound = near_min[i] or near_max[i]
            if near_min[i]:
                self._sat_count_min[i] += 1
            if near_max[i]:
                self._sat_count_max[i] += 1
            if at_bound and not self._sat_prev_at_bound[i]:
                self._sat_entries[i] += 1
            elif (not at_bound) and self._sat_prev_at_bound[i]:
                self._sat_exits[i] += 1
            self._sat_prev_at_bound[i] = at_bound

        self._cum_u_count += 1
        for i in range(n_c):
            peak_cum[i] = self._cum_u_absmax[i]
            rms_cum[i] = math.sqrt(self._cum_u_sumsq[i] / self._cum_u_count)

        self._step_count += 1

        row: Dict[str, float] = {
            "step": int(step),
            "time_s": round(step * dt, 4),
            "dt": dt,
            "solve_time_ms": solve_time * 1000.0,
            "success": int(bool(success)),
        }
        for i, name in enumerate(self.state_names):
            row[f"q_term_{name}"] = q_term[i]
        for i, name in enumerate(self.control_names):
            row[f"u_{name}"] = u[i]
            row[f"absu_{name}"] = abs_u[i]
            row[f"du_{name}"] = du[i]
            row[f"absdu_{name}"] = abs_du[i]
            row[f"r_term_{name}"] = r_term[i]
            row[f"rdelta_term_{name}"] = rdelta_term[i]
            row[f"sat_near_min_{name}"] = int(near_min[i])
            row[f"sat_near_max_{name}"] = int(near_max[i])
            row[f"tv_cum_{name}"] = self._tv_cum[i]
            row[f"peak_roll_{name}"] = peak_roll[i]
            row[f"rms_roll_{name}"] = rms_roll[i]
            row[f"peak_cum_{name}"] = peak_cum[i]
            row[f"rms_cum_{name}"] = rms_cum[i]
        row["q_sum"] = q_sum
        row["r_sum"] = r_sum
        row["J_u_step"] = J_u_step
        row["J_u_cum"] = self._J_u_cum
        row["J_du_step"] = J_du_step
        row["J_du_cum"] = self._J_du_cum
        row["upstream_drops"] = int(upstream_drops)
        row["local_drops"] = self._local_drop_count

        self._csv_writer.writerow([row[col] for col in self._header])
        if self._step_count % self.flush_every == 0:
            self._csv_file.flush()

        for col in self._header:
            self._col_stats[col].update(row[col])

    # ------------------------------------------------------------ finalize
    def finalize(self) -> Dict:
        """Stops the worker thread, closes costs_errors.csv, and returns the
        "costs_errors" summary fragment for logger_node to fold into the same
        run's summary.json (single write, owned by ExperimentLogger.finalize())."""
        if self._finalized:
            return {}
        self._finalized = True
        self._stop_event.set()
        self._thread.join(timeout=5.0)

        try:
            self._csv_file.flush()
            self._csv_file.close()
        except Exception:
            pass

        terms = {col: stats.as_dict() for col, stats in self._col_stats.items()}

        saturation_events = {
            name: {"entries": self._sat_entries[i], "exits": self._sat_exits[i]}
            for i, name in enumerate(self.control_names)
        }
        saturation_fraction = {
            name: {
                "near_min": (self._sat_count_min[i] / self._step_count) if self._step_count else 0.0,
                "near_max": (self._sat_count_max[i] / self._step_count) if self._step_count else 0.0,
            }
            for i, name in enumerate(self.control_names)
        }
        dropped_samples = {
            "upstream_nmpc_node": int(terms["upstream_drops"]["max"]) if self._step_count else 0,
            "local_logger_node": int(terms["local_drops"]["max"]) if self._step_count else 0,
        }

        print(f"[ControllerEffortLogger] costs_errors.csv finalized: {self._step_count} steps, "
              f"J_u_total={self._J_u_cum:.4f}, J_du_total={self._J_du_cum:.4f}, "
              f"dropped(upstream/local)={dropped_samples['upstream_nmpc_node']}/{dropped_samples['local_logger_node']}")

        return {
            "costs_errors": {
                "csv_path": self.csv_path,
                "total_steps": self._step_count,
                "terms": terms,
                "saturation_events": saturation_events,
                "saturation_fraction": saturation_fraction,
                "dropped_samples": dropped_samples,
            }
        }
