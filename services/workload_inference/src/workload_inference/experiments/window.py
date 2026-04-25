"""Concrete ExperimentManagerWindow subclasses for each experiment type."""

import logging
import time
from dataclasses import dataclass

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

from workload_inference.experiments.base import DRONE_COUNT
from workload_inference.experiments.base_window import ExperimentManagerWindow
from workload_inference.experiments.data_structures import (
    ExperimentState,
    ExperimentStatus,
    NBackData,
)
from workload_inference.experiments.manager import (
    GateRacingExperimentManager,
    NBackExperimentManager,
)

logger = logging.getLogger("ExperimentManagerWindow")


# ── Gate racing card helpers ──────────────────────────────────────────────────


def _card_style(bg: str, fg: str, border: str = "1px solid #555555") -> str:
    """Build a Qt stylesheet for a card widget and its QLabel children."""
    return (
        f"QWidget{{background-color:{bg};border:{border};border-radius:4px;}}"
        f"QLabel{{background-color:transparent;border:none;color:{fg};}}"
    )


_GATE_STYLES = {
    0: _card_style("#4a4a4a", "#888888"),  # Idle
    1: _card_style("#e3f2fd", "#0d47a1", "2px solid #1e88e5"),  # Next (current)
    2: _card_style("#fff9c4", "#5d4037", "1px solid #f9a825"),  # PartialComplete
    3: _card_style("#c8e6c9", "#1b5e20", "1px solid #66bb6a"),  # Completed
}
_TRIAL_STYLE_FUTURE = _card_style("#4a4a4a", "#888888")
_TRIAL_STYLE_CURRENT = _card_style("#e3f2fd", "#0d47a1", "2px solid #1e88e5")
_TRIAL_STYLE_PAST = _card_style("#e8e8e8", "#333333", "1px solid #bbbbbb")


@dataclass
class _GateCard:
    widget: QWidget
    title_lbl: QLabel
    split_lbl: QLabel


@dataclass
class _TrialCard:
    widget: QWidget
    title_lbl: QLabel
    time_lbl: QLabel
    crash_lbl: QLabel
    collision_lbl: QLabel


# ══════════════════════════════════════════════════════════════════════════════
# N-back window
# ══════════════════════════════════════════════════════════════════════════════


class NBackExperimentManagerWindow(ExperimentManagerWindow):
    """Window for the N-back experiment — shows sequence, level, and score panel."""

    def __init__(self, experiment_manager: NBackExperimentManager):
        super().__init__(experiment_manager)
        self.setWindowTitle("Experiment Manager - N-back")

    # ── Drone visualizer ──────────────────────────────────────────────────────

    def _create_drone_visualizer(self):
        from workload_inference.visualize import DroneDataCanvas

        return DroneDataCanvas(
            parent=self,
            num_drones=DRONE_COUNT,
            plotting_window=200,
        )

    # ── Info panel ────────────────────────────────────────────────────────────

    def _initialize_experiment_info_panel(self) -> None:
        """Build the N-back info row: order, current level, sequence, score."""
        nback_info_widget = QWidget()
        nback_info_layout = QHBoxLayout()
        nback_info_widget.setLayout(nback_info_layout)
        self._experiment_specific_panel_layout.addWidget(nback_info_widget)

        nback_order_label = QLabel("N-back order:")
        nback_order_label.setStyleSheet("font-weight: bold;")
        nback_info_layout.addWidget(nback_order_label)
        self._nback_levels_value_label = QLabel("N/A")
        nback_info_layout.addWidget(self._nback_levels_value_label)

        nback_info_layout.addWidget(self._make_separator())

        current_nback_label = QLabel("Current N-back:")
        current_nback_label.setStyleSheet("font-weight: bold;")
        nback_info_layout.addWidget(current_nback_label)
        self._current_nback_level_value_label = QLabel("N/A")
        nback_info_layout.addWidget(self._current_nback_level_value_label)

        nback_info_layout.addWidget(self._make_separator())

        nback_sequence_label = QLabel("Sequence:")
        nback_sequence_label.setStyleSheet("font-weight: bold;")
        nback_info_layout.addWidget(nback_sequence_label)
        self._nback_sequence_value_label = QLabel("[N/A]")
        self._nback_sequence_value_label.setTextFormat(Qt.TextFormat.MarkdownText)
        nback_info_layout.addWidget(self._nback_sequence_value_label, 1)

        nback_info_layout.addWidget(self._make_separator())

        self.nback_score_label = QLabel("Score: N/A")
        self.nback_score_label.setStyleSheet("font-weight: bold;")
        nback_info_layout.addWidget(self.nback_score_label)

    def _update_experiment_info_panel(self, status: ExperimentStatus) -> None:
        """Refresh N-back level, sequence, and score labels."""
        self._nback_levels_value_label.setText(
            " -> ".join(map(str, status.nback_levels_order))
            if status.nback_levels_order
            else "N/A"
        )
        self._current_nback_level_value_label.setText(
            f"{status.current_nback_level}"
            if status.current_nback_level >= 0
            else "N/A"
        )
        nback_data = self.experiment_manager.nback_latest_datas  # type: ignore[attr-defined]
        if nback_data is not None:
            stimuli = list(self._generate_nback_stimulus_click_expected(nback_data))
            self._nback_sequence_value_label.setText(" -> ".join(stimuli))
            score = sum(1 for data in nback_data if data.is_correct)
            num_stimuli = sum(1 for data in nback_data if data.timestamp > 0)
            self.nback_score_label.setText(f"Score: {score}/{num_stimuli}")

    def _generate_nback_stimulus_click_expected(self, sequence: list[NBackData]):
        """Yield stimuli formatted with HTML red/bold for expected clicks."""
        nback_level = sequence[0].nback_level
        for idx, data in enumerate(sequence):
            if idx < nback_level:
                yield str(data.stimulus)
                continue
            expected_click = data.stimulus == sequence[idx - nback_level].stimulus
            if expected_click:
                yield f"<span style='color: red;'><b>{data.stimulus}</b></span>"
            else:
                yield str(data.stimulus)


# ══════════════════════════════════════════════════════════════════════════════
# Gate racing window
# ══════════════════════════════════════════════════════════════════════════════

_PLACEHOLDER_GATES = 4
_PLACEHOLDER_TRIALS = 2


class GateRacingExperimentManagerWindow(ExperimentManagerWindow):
    """Window for the gate racing experiment.

    Info panel layout (left header panel):
      Upper half — one card per gate: title (G #N), split time from trial start.
                   Coloured by gate state: dark=idle, blue=next, yellow=partial,
                   green=all-passed.
      Lower half — one card per trial: title, lap time, crashed drones, collisions.
                   Highlighted for the current trial; grayed out for future ones.
    """

    def __init__(self, experiment_manager: GateRacingExperimentManager):
        super().__init__(experiment_manager)
        self.setWindowTitle("Experiment Manager - Racing Gates")
        experiment_manager.register_api_ready_listener(self._on_api_ready)

    def _on_api_ready(self) -> None:
        """Called when the API recovers — refresh gate layout on the canvas."""
        self._drone_visualizer.update_gates(
            self.experiment_manager._gate_layout  # type: ignore[attr-defined]
        )
        self.experiment_manager._gate_layout_receiver.register_on_data_changed_listener(
            self._drone_visualizer.update_gates
        )

    # ── Drone visualizer ──────────────────────────────────────────────────────

    def _create_drone_visualizer(self):
        from workload_inference.visualize import DroneDataCanvasGateRacing

        return DroneDataCanvasGateRacing(
            parent=self,
            num_drones=DRONE_COUNT,
            gates=self.experiment_manager._gate_layout,  # type: ignore[attr-defined]
        )

    # ── Info panel ────────────────────────────────────────────────────────────

    def _initialize_experiment_info_panel(self) -> None:
        """Build upper (gate cards) / lower (trial cards) info panel."""
        main = QWidget()
        main_layout = QVBoxLayout(main)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)
        self._experiment_specific_panel_layout.addWidget(main)

        # ── Upper: gate cards ─────────────────────────────────────────────
        self._gates_container = QWidget()
        self._gates_layout = QHBoxLayout(self._gates_container)
        self._gates_layout.setContentsMargins(2, 2, 2, 2)
        self._gates_layout.setSpacing(4)

        gate_scroll = QScrollArea()
        gate_scroll.setWidget(self._gates_container)
        gate_scroll.setWidgetResizable(True)
        gate_scroll.setFixedHeight(60)
        gate_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        gate_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        main_layout.addWidget(gate_scroll, 1)

        # ── Lower: trial cards ────────────────────────────────────────────
        self._trials_container = QWidget()
        self._trials_layout = QHBoxLayout(self._trials_container)
        self._trials_layout.setContentsMargins(2, 2, 2, 2)
        self._trials_layout.setSpacing(4)

        trial_scroll = QScrollArea()
        trial_scroll.setWidget(self._trials_container)
        trial_scroll.setWidgetResizable(True)
        trial_scroll.setFixedHeight(88)
        trial_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        trial_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        main_layout.addWidget(trial_scroll, 1)

        # ── Internal state ────────────────────────────────────────────────
        self._gate_cards: list[_GateCard] = []
        self._trial_cards: list[_TrialCard] = []
        self._displayed_gate_count = 0
        self._displayed_trial_count = 0

        self._rebuild_gate_cards(_PLACEHOLDER_GATES)
        self._rebuild_trial_cards(_PLACEHOLDER_TRIALS)

    # ── Card factories ────────────────────────────────────────────────────────

    def _make_gate_card(self, label: str) -> _GateCard:
        card = QWidget()
        card.setFixedWidth(70)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(1)

        title = QLabel(label)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(title)

        split = QLabel("—")
        split.setAlignment(Qt.AlignmentFlag.AlignCenter)
        split.setStyleSheet("font-size: 11px;")
        layout.addWidget(split)
        layout.addStretch(1)

        card.setStyleSheet(_GATE_STYLES[0])
        return _GateCard(widget=card, title_lbl=title, split_lbl=split)

    def _make_trial_card(self, trial_num: int) -> _TrialCard:
        card = QWidget()
        card.setFixedWidth(80)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(0)

        def _lbl(text: str, bold: bool = False, size: int = 11) -> QLabel:
            lbl = QLabel(text)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            style = f"font-size:{size}px;"
            if bold:
                style += "font-weight:bold;"
            lbl.setStyleSheet(style)
            return lbl

        title = _lbl(f"Trial #{trial_num}", bold=True, size=12)
        time_lbl = _lbl("—")
        crash_lbl = _lbl("—")
        collision_lbl = _lbl("—")

        for w in (title, time_lbl, crash_lbl, collision_lbl):
            layout.addWidget(w)
        layout.addStretch(1)

        card.setStyleSheet(_TRIAL_STYLE_FUTURE)
        return _TrialCard(
            widget=card,
            title_lbl=title,
            time_lbl=time_lbl,
            crash_lbl=crash_lbl,
            collision_lbl=collision_lbl,
        )

    # ── Card rebuild ──────────────────────────────────────────────────────────

    def _rebuild_gate_cards(self, n_gates: int) -> None:
        while self._gates_layout.count():
            item = self._gates_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._gate_cards.clear()
        for i in range(n_gates):
            card = self._make_gate_card(f"G #{i + 1}")
            self._gates_layout.addWidget(card.widget)
            self._gate_cards.append(card)
        self._gates_layout.addStretch(1)
        self._displayed_gate_count = n_gates

    def _rebuild_trial_cards(self, n_trials: int) -> None:
        while self._trials_layout.count():
            item = self._trials_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._trial_cards.clear()
        for i in range(n_trials):
            card = self._make_trial_card(i + 1)
            self._trials_layout.addWidget(card.widget)
            self._trial_cards.append(card)
        self._trials_layout.addStretch(1)
        self._displayed_trial_count = n_trials

    # ── Update ────────────────────────────────────────────────────────────────

    def _update_experiment_info_panel(self, status: ExperimentStatus) -> None:
        manager: GateRacingExperimentManager = self.experiment_manager  # type: ignore[assignment]

        # Rebuild cards if layout size changed
        n_gates = len(manager._gate_layout) or _PLACEHOLDER_GATES
        n_trials = max(status.total_trials, _PLACEHOLDER_TRIALS)
        if n_gates != self._displayed_gate_count:
            self._rebuild_gate_cards(n_gates)
        if n_trials != self._displayed_trial_count:
            self._rebuild_trial_cards(n_trials)

        # ── Gate cards ────────────────────────────────────────────────────
        alive_drone_count = DRONE_COUNT - manager.trial_crashed_drones.get(
            status.current_trial, DRONE_COUNT
        )
        prev_pass_ts: int | None = manager.trial_start_timestamp
        for i, card in enumerate(self._gate_cards):
            if i < len(manager._gate_layout):
                gate_id = int(manager._gate_layout[i].id)
                card.title_lbl.setText(f"G #{gate_id}")
            else:
                gate_id = i  # placeholder index

            gate_status = manager._latest_gate_statuses.get(gate_id)
            pass_count = int(gate_status.pass_count) if gate_status else 0

            # Colour driven by pass_count, not gate_state enum:
            #   none passed + is next gate → blue; none passed → idle dark
            #   1+ passed but not all      → yellow
            #   all passed                 → green
            if pass_count == 0:
                state = int(gate_status.gate_state) if gate_status else 0
                style = _GATE_STYLES.get(state, _GATE_STYLES[0])
            elif pass_count < alive_drone_count:
                style = _GATE_STYLES[2]  # yellow (PartialComplete)
            else:
                style = _GATE_STYLES[3]  # green (Completed)
            card.widget.setStyleSheet(style)

            # Split: relative to previous gate's first pass (or trial start for gate 0)
            if (
                gate_status
                and gate_status.first_pass_timestamp > 0
                and prev_pass_ts is not None
            ):
                split_s = (int(gate_status.first_pass_timestamp) - prev_pass_ts) / 1000
                card.split_lbl.setText(f"+{split_s:.2f}s")
                prev_pass_ts = int(gate_status.first_pass_timestamp)
            else:
                card.split_lbl.setText("—")
                # Don't advance prev_pass_ts — gate not yet passed

        # Update canvas with gate statuses for coloring
        gate_statuses_dict = {
            int(gate_id): {
                "gate_state": int(status.gate_state),
                "pass_count": int(status.pass_count),
            }
            for gate_id, status in manager._latest_gate_statuses.items()
        }
        self._drone_visualizer.update_gate_statuses(gate_statuses_dict)

        # ── Trial cards ───────────────────────────────────────────────────
        current_trial = status.current_trial
        is_finished = status.current_state == ExperimentState.Finished
        for i, card in enumerate(self._trial_cards):
            trial_num = i + 1
            finish = manager.trial_finish_times.get(trial_num)

            if is_finished:
                # In Finished state, current_trial from the API may be reset/unreliable.
                # Show every trial that has a recorded finish time as a past result.
                if finish is not None:
                    card.widget.setStyleSheet(_TRIAL_STYLE_PAST)
                    m, s = divmod(int(finish), 60)
                    card.time_lbl.setText(f"{m:02d}:{s:02d}")
                    crashed = manager.trial_crashed_drones.get(trial_num)
                    card.crash_lbl.setText(
                        f"{crashed} crashed" if crashed is not None else "—"
                    )
                    card.collision_lbl.setText("—")
                else:
                    card.widget.setStyleSheet(_TRIAL_STYLE_FUTURE)
                    card.time_lbl.setText("—")
                    card.crash_lbl.setText("—")
                    card.collision_lbl.setText("—")
            elif trial_num < current_trial:
                card.widget.setStyleSheet(_TRIAL_STYLE_PAST)
                if finish is not None:
                    m, s = divmod(int(finish), 60)
                    card.time_lbl.setText(f"{m:02d}:{s:02d}")
                else:
                    card.time_lbl.setText("done")
                crashed = manager.trial_crashed_drones.get(trial_num)
                if crashed is not None:
                    card.crash_lbl.setText(f"{crashed} crashed")
                else:
                    card.crash_lbl.setText("—")
                card.collision_lbl.setText("—")
            elif trial_num == current_trial:
                card.widget.setStyleSheet(_TRIAL_STYLE_CURRENT)
                # Check if trial has a recorded finish time (for Countdown/ReadyScreen)
                if finish is not None:
                    # Trial is completed, show the final recorded time
                    m, s = divmod(int(finish), 60)
                    card.time_lbl.setText(f"{m:02d}:{s:02d}")
                elif status.current_state == ExperimentState.Trial:
                    # Trial is active
                    if manager.trial_start_timestamp is not None:
                        elapsed_s = (
                            time.time() * 1000 - manager.trial_start_timestamp
                        ) / 1000
                        m, s = divmod(int(elapsed_s), 60)
                        card.time_lbl.setText(f"{m:02d}:{s:02d}")
                    else:
                        card.time_lbl.setText("00:00")
                else:
                    # On ReadyScreen/Countdown without finish time, show 00:00
                    card.time_lbl.setText("00:00")
                crashed = manager.trial_crashed_drones.get(trial_num, 0)
                card.crash_lbl.setText(f"{crashed} crashed")
                card.collision_lbl.setText("N/A")
            else:
                card.widget.setStyleSheet(_TRIAL_STYLE_FUTURE)
                card.time_lbl.setText("—")
                card.crash_lbl.setText("—")
                card.collision_lbl.setText("—")
