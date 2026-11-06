from typing import Optional

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle


PITCH_LENGTH = 105.3
PITCH_WIDTH = 68.0
from pathlib import Path
from models.frame import Frame
from typing import Protocol
from pathlib import Path
from typing import Protocol

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle


class DrawablePlayer(Protocol):
    x: float
    y: float
    player_team: str


def plot_coordinates(
    x_coordinates: list[Optional[float]],
    y_coordinates: list[Optional[float]],
    frames: list[int],
) -> None:
    if not (len(x_coordinates) == len(y_coordinates) == len(frames)):
        raise ValueError(
            "x_coordinates, y_coordinates, and frames must have the same length."
        )

    fig, (ax_x, ax_y) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax_x.plot(frames, x_coordinates, marker=".", label="X coordinate")
    ax_x.set_title("X Coordinates vs. Frames")
    ax_x.set_ylabel("X")
    ax_x.grid(True, alpha=0.3)
    ax_x.legend()

    ax_y.plot(frames, y_coordinates, marker=".", label="Y coordinate")
    ax_y.set_title("Y Coordinates vs. Frames")
    ax_y.set_xlabel("Frame")
    ax_y.set_ylabel("Y")
    ax_y.grid(True, alpha=0.3)
    ax_y.legend()

    fig.tight_layout()
    plt.show()


def draw_frame(
    players: list[DrawablePlayer],
    frame_id: int | str,
    timestamp: str | float | None = None,
    show_labels: bool = False,
    save: bool = False,
    save_name_prefix: str = ""
) -> None:
    """Draw a football pitch with players and their positions."""
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

    # Penalty and goal areas
    for depth, width in ((16.5, 40.32), (5.5, 18.32)):
        for x in (0, PITCH_LENGTH - depth):
            ax.add_patch(
                Rectangle(
                    (x, (PITCH_WIDTH - width) / 2),
                    depth,
                    width,
                    fill=False,
                    edgecolor="white",
                )
            )

    # Players
    colors = ["dodgerblue", "tomato", "gold", "violet"]
    team_colors: dict[str, str] = {}

    for player in players:
        team = player.team

        if team not in team_colors:
            team_colors[team] = colors[len(team_colors) % len(colors)]

        ax.scatter(
            player.x,
            player.y,
            s=110,
            color=team_colors[team],
            edgecolors="white",
            linewidths=1,
            zorder=3,
        )

        if show_labels and hasattr(player, "player_id"):
            ax.annotate(
                str(player.player_id),
                (player.x, player.y),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color="white",
            )

    # Axes and metadata
    ax.set_xlim(0, PITCH_LENGTH)
    ax.set_ylim(PITCH_WIDTH, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Length (m)")
    ax.set_ylabel("Width (m)")
    title = f"Frame {frame_id}"

    if timestamp is not None:
        title += f" | Timestamp: {timestamp}"

    ax.set_title(title)
    ax.grid(False)

    plt.tight_layout()

    if save:
        output_dir = Path("./plots")
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{save_name_prefix}_{frame_id}"
        if timestamp is not None:
            filename += f"-{timestamp}"

        fig.savefig(
            output_dir / f"{filename}.png",
            bbox_inches="tight",
            dpi=150,
        )
        plt.close(fig)
    else:
        plt.show()