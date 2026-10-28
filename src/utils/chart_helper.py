from typing import Optional

import matplotlib.pyplot as plt


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