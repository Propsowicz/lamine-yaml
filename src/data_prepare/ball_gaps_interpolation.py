import math
import copy


def is_ball_visible(ball) -> bool:
    if ball is None:
        return False
    
    if ball.x is None or ball.y is None:
        return False
    
    if math.isnan(ball.x) or math.isnan(ball.y):
        return False
    
    return True


def timestamp_to_frame(minutes, seconds, fps=25) -> int:
    return int((minutes * 60 + seconds) * fps)


def find_gaps(frames, exclude_halftime=False) -> list[dict]:
    gaps = []
    start = None
    
    halftime_start = timestamp_to_frame(59, 8)
    halftime_end = timestamp_to_frame(75, 49)

    for i, frame in enumerate(frames):
        if exclude_halftime and halftime_start <= frame.id <= halftime_end:
            if start is not None:
                gaps.append(_make_gap(frames, start, i - 1))
                start = None
            continue
        
        visible = is_ball_visible(frame.ball)

        if not visible and start is None:
            start = i
        elif visible and start is not None:
            gaps.append(_make_gap(frames, start, i - 1))
            start = None

    if start is not None:
        gaps.append(_make_gap(frames, start, len(frames) - 1))

    return gaps


def _make_gap(frames, start_idx, end_idx) -> dict:
    length = end_idx - start_idx + 1

    if length <= 5:
        category = "short"
    elif length <= 25:
        category = "medium"
    else:
        category = "long"

    return {
        "start_idx": start_idx,
        "end_idx": end_idx,
        "start_frame": frames[start_idx].id,
        "end_frame": frames[end_idx].id,
        "start_ts": frames[start_idx].timestamp,
        "end_ts": frames[end_idx].timestamp,
        "length": length,
        "duration": length * 0.04,
        "category": category,
    }


def _valid(v) -> bool:
    return v is not None and not math.isnan(v)


def interpolate_short_gaps(frames, exclude_halftime=False) -> int:
    gaps = find_gaps(frames, exclude_halftime=exclude_halftime)
    
    filled = 0

    for g in gaps:
        if g["category"] != "short":
            continue

        s, e = g["start_idx"], g["end_idx"]
        if s == 0 or e == len(frames) - 1:
            continue

        a = frames[s - 1].ball
        b = frames[e + 1].ball
        n = e - s + 2

        for k, idx in enumerate(range(s, e + 1), start=1):
            t = k / n
            ball = frames[idx].ball
            if ball is None:
                ball = copy.copy(b)
                frames[idx].ball = ball

            ball.x = a.x + (b.x - a.x) * t
            ball.y = a.y + (b.y - a.y) * t
            if _valid(a.z) and _valid(b.z):
                ball.z = a.z + (b.z - a.z) * t

        filled += g["length"]

    return filled
