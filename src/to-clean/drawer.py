
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from models import Player, Ball, Frame

PITCH_LENGTH = 105.3
PITCH_WIDTH = 68.0
from pathlib import Path


def draw_frame(frame, show_labels: bool = False,
    save: bool = False) -> None:
    """Draw a football frame with its players and ball."""

    fig, ax = plt.subplots(figsize=(12, 8))

    # Pitch background and boundary
    ax.set_facecolor("#287a43")
    ax.add_patch(
        Rectangle(
            (0, 0),
            PITCH_LENGTH,
            PITCH_WIDTH,
            facecolor="#287a43",
            edgecolor="white",
            linewidth=2,
        )
    )

    # Pitch markings
    ax.plot(
        [PITCH_LENGTH / 2, PITCH_LENGTH / 2],
        [0, PITCH_WIDTH],
        color="white",
        linewidth=1,
    )
    ax.add_patch(
        Circle(
            (PITCH_LENGTH / 2, PITCH_WIDTH / 2),
            9.15,
            fill=False,
            edgecolor="white",
        )
    )

    # Penalty areas
    penalty_depth = 16.5
    penalty_width = 40.32
    goal_area_depth = 5.5
    goal_area_width = 18.32

    for x in (0, PITCH_LENGTH - penalty_depth):
        ax.add_patch(
            Rectangle(
                (x, (PITCH_WIDTH - penalty_width) / 2),
                penalty_depth,
                penalty_width,
                fill=False,
                edgecolor="white",
            )
        )

    # Goal areas
    for x in (0, PITCH_LENGTH - goal_area_depth):
        ax.add_patch(
            Rectangle(
                (x, (PITCH_WIDTH - goal_area_width) / 2),
                goal_area_depth,
                goal_area_width,
                fill=False,
                edgecolor="white",
            )
        )

    # Players, grouped by team
    team_colors = {}
    colors = ["dodgerblue", "tomato", "gold", "violet"]

    for player in frame.players:
        if player.team not in team_colors:
            team_colors[player.team] = colors[
                len(team_colors) % len(colors)
            ]

        ax.scatter(
            player.x,
            player.y,
            s=110,
            color=team_colors[player.team],
            edgecolors="white",
            linewidths=1,
            zorder=3,
        )

        if show_labels:
            ax.annotate(
                str(player.player_id),
                (player.x, player.y),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color="white",
            )

    # Ball
    ax.scatter(
        frame.ball.x,
        frame.ball.y,
        s=70,
        color="black",
        edgecolors="white",
        linewidths=1,
        marker="o",
        zorder=5,
        label="Ball",
    )

    # Axes and metadata
    ax.set_xlim(0, PITCH_LENGTH)
    ax.set_ylim(PITCH_WIDTH, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Length (m)")
    ax.set_ylabel("Width (m)")
    ax.set_title(f"Frame {frame.id} | Timestamp: {frame.timestamp}")
    ax.grid(False)

    plt.tight_layout()
    if save:
        output_dir = Path("./plots")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{frame.id}-{frame.timestamp}.png"
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
    else:
        plt.show()

MAX_COORDINATE_CHANGE_M = 5.0

from dataclasses import dataclass
from typing import Sequence

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class CoordinateAnomaly:
    player_id: int
    axis: str
    previous_frame_id: int
    current_frame_id: int
    previous_value: float
    current_value: float
    change_m: float


def detect_player_coordinate_anomalies(
    frames: Sequence["Frame"],
    player_id: int,
    max_change_m: float = MAX_COORDINATE_CHANGE_M,
) -> list[CoordinateAnomaly]:
    """Plot player coordinates and report excessive changes between frames."""
    if max_change_m <= 0:
        raise ValueError("max_change_m must be greater than zero")

    # Extract and sort observations by frame ID.
    observations: list[tuple[int, float, float]] = []

    for frame in sorted(frames, key=lambda f: f.id):
        player = next(
            (p for p in frame.players if p.player_id == player_id),
            None,
        )

        if player is not None:
            observations.append((frame.id, player.x, player.y))

    if not observations:
        print(f"No observations found for player {player_id}.")
        return []

    frame_ids = [frame_id for frame_id, _, _ in observations]
    x_coords = [x for _, x, _ in observations]
    y_coords = [y for _, _, y in observations]

    # Plot x and y coordinates against frame ID.
    fig, (ax_x, ax_y) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(12, 8),
        sharex=True,
    )

    for ax, coords, axis_name in (
        (ax_x, x_coords, "X"),
        (ax_y, y_coords, "Y"),
    ):
        ax.plot(frame_ids, coords, marker="o", markersize=3)
        ax.set_title(f"Player {player_id}: {axis_name} coordinate")
        ax.set_ylabel(f"{axis_name} (m)")
        ax.grid(True, alpha=0.3)

    ax_y.set_xlabel("Frame ID")
    fig.tight_layout()
    plt.show()

    # Detect jumps between consecutive observations.
    anomalies: list[CoordinateAnomaly] = []

    for previous, current in zip(observations, observations[1:]):
        previous_id, previous_x, previous_y = previous
        current_id, current_x, current_y = current

        for axis, previous_value, current_value in (
            ("x", previous_x, current_x),
            ("y", previous_y, current_y),
        ):
            change = abs(current_value - previous_value)

            if change > max_change_m:
                anomalies.append(
                    CoordinateAnomaly(
                        player_id=player_id,
                        axis=axis,
                        previous_frame_id=previous_id,
                        current_frame_id=current_id,
                        previous_value=previous_value,
                        current_value=current_value,
                        change_m=change,
                    )
                )

    # Print the report.
    print(f"\nAnomaly report for player {player_id}")
    print(f"Maximum allowed coordinate change: {max_change_m:.2f} m")
    print(f"Frames analyzed: {len(observations)}")
    print(f"Anomalies detected: {len(anomalies)}")

    if not anomalies:
        print("No coordinate anomalies detected.")
    else:
        for anomaly in anomalies:
            print(
                f"[{anomaly.axis.upper()}] "
                f"Frame {anomaly.previous_frame_id} -> "
                f"Frame {anomaly.current_frame_id}: "
                f"{anomaly.previous_value:.2f} m -> "
                f"{anomaly.current_value:.2f} m "
                f"(change: {anomaly.change_m:.2f} m)"
            )

    return anomalies


from collections.abc import Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Assumes PITCH_LENGTH and PITCH_WIDTH are defined in your project.
# from your_module import PITCH_LENGTH, PITCH_WIDTH, Frame, Player


def detect_player_coordinate_anomalies(
    frames: Sequence["Frame"],
    player_id: int,
    max_change_m: float = MAX_COORDINATE_CHANGE_M,
) -> list[CoordinateAnomaly]:
    """Plot player coordinates and report excessive changes between frames."""
    if max_change_m <= 0:
        raise ValueError("max_change_m must be greater than zero")

    # Extract and sort observations by frame ID.
    observations: list[tuple[int, float, float]] = []

    for frame in sorted(frames, key=lambda f: f.id):
        player = next(
            (p for p in frame.players if p.player_id == player_id),
            None,
        )

        if player is not None:
            observations.append((frame.id, player.x, player.y))

    if not observations:
        print(f"No observations found for player {player_id}.")
        return []

    frame_ids = [frame_id for frame_id, _, _ in observations]
    x_coords = [x for _, x, _ in observations]
    y_coords = [y for _, _, y in observations]

    # Plot x and y coordinates against frame ID.
    fig, (ax_x, ax_y) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(12, 8),
        sharex=True,
    )

    for ax, coords, axis_name in (
        (ax_x, x_coords, "X"),
        (ax_y, y_coords, "Y"),
    ):
        ax.plot(frame_ids, coords, marker="o", markersize=3)
        ax.set_title(f"Player {player_id}: {axis_name} coordinate")
        ax.set_ylabel(f"{axis_name} (m)")
        ax.grid(True, alpha=0.3)

    ax_y.set_xlabel("Frame ID")
    fig.tight_layout()
    plt.show()

    # Detect jumps between consecutive observations.
    anomalies: list[CoordinateAnomaly] = []

    for previous, current in zip(observations, observations[1:]):
        previous_id, previous_x, previous_y = previous
        current_id, current_x, current_y = current

        for axis, previous_value, current_value in (
            ("x", previous_x, current_x),
            ("y", previous_y, current_y),
        ):
            change = abs(current_value - previous_value)

            if change > max_change_m:
                anomalies.append(
                    CoordinateAnomaly(
                        player_id=player_id,
                        axis=axis,
                        previous_frame_id=previous_id,
                        current_frame_id=current_id,
                        previous_value=previous_value,
                        current_value=current_value,
                        change_m=change,
                    )
                )

    # Print the report.
    print(f"\nAnomaly report for player {player_id}")
    print(f"Maximum allowed coordinate change: {max_change_m:.2f} m")
    print(f"Frames analyzed: {len(observations)}")
    print(f"Anomalies detected: {len(anomalies)}")

    if not anomalies:
        print("No coordinate anomalies detected.")
    else:
        for anomaly in anomalies:
            print(
                f"[{anomaly.axis.upper()}] "
                f"Frame {anomaly.previous_frame_id} -> "
                f"Frame {anomaly.current_frame_id}: "
                f"{anomaly.previous_value:.2f} m -> "
                f"{anomaly.current_value:.2f} m "
                f"(change: {anomaly.change_m:.2f} m)"
            )

    return anomalies


def plot_player_track(
    frames: Sequence["Frame"],
    player_id: int,
    max_step_distance: float = 10.0,
) -> None:
    """Plot a player's track on a football pitch and highlight anomalies."""
    track: list[tuple[int, float, float]] = []

    for frame_index, frame in enumerate(frames):
        player = next(
            (p for p in frame.players if p.player_id == player_id),
            None,
        )

        if player is not None:
            track.append((frame_index, player.x, player.y))

    if not track:
        raise ValueError(f"Player {player_id} was not found in any frame.")

    frame_indices = [point[0] for point in track]
    xs = [point[1] for point in track]
    ys = [point[2] for point in track]

    anomaly_indices = set(
        detect_player_coordinate_anomalies(
            frames, player_id, max_step_distance
        )
    )
    anomaly_points = [
        point for point in track if point[0] in anomaly_indices
    ]

    fig, ax = plt.subplots(figsize=(14, 9))

    # Draw the pitch.
    ax.add_patch(
        Rectangle(
            (0, 0),
            PITCH_LENGTH,
            PITCH_WIDTH,
            linewidth=2,
            edgecolor="black",
            facecolor="none",
        )
    )

    # Halfway line and centre circle.
    ax.plot(
        [PITCH_LENGTH / 2, PITCH_LENGTH / 2],
        [0, PITCH_WIDTH],
        color="black",
        linewidth=1,
    )
    ax.add_patch(
        plt.Circle(
            (PITCH_LENGTH / 2, PITCH_WIDTH / 2),
            9.15,
            fill=False,
            color="black",
        )
    )

    # Player's trajectory.
    ax.plot(xs, ys, color="royalblue", linewidth=1.5, alpha=0.8)
    ax.scatter(xs, ys, color="royalblue", s=15, label="Player track")
    ax.scatter(
        xs[0], ys[0], color="green", s=100, marker="o", label="Start"
    )
    ax.scatter(
        xs[-1], ys[-1], color="black", s=100, marker="x", label="End"
    )

    # Highlight suspicious positions.
    if anomaly_points:
        ax.scatter(
            [point[1] for point in anomaly_points],
            [point[2] for point in anomaly_points],
            color="red",
            marker="x",
            s=100,
            linewidths=2,
            label="Potential anomalies",
            zorder=5,
        )

    ax.set_xlim(0, PITCH_LENGTH)
    ax.set_ylim(0, PITCH_WIDTH)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X position (meters)")
    ax.set_ylabel("Y position (meters)")
    ax.set_title(f"Player {player_id} — Coordinate Anomaly Detection")
    ax.legend(loc="upper right")
    ax.grid(False)

    plt.tight_layout()
    plt.show()