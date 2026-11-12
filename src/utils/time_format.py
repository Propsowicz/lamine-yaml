from datetime import timedelta


def format_elapsed_time(timestamp: timedelta) -> str:
    """Format elapsed time as HH:MM:SS, truncating fractional seconds."""
    total_seconds = int(timestamp.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
