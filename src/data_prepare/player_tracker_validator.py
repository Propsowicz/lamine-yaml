from copy import copy
from math import hypot, isfinite
from models.player import Player

def is_player_visible(player: Player) -> bool:
    return (
        isfinite(player.x)
        and isfinite(player.y)
    )


def calculate_distance(player1: Player, player2: Player) -> float:
    return hypot(
        player1.x - player2.x,
        player1.y - player2.y,
    )


def validate_player_tracking(
    current_players: list[Player],
    previous_players: list[Player],
    merge_distance: float = 2.0,
    reappearance_distance: float = 2.0,
) -> list[Player]:

    previous_by_id = {
        player.player_id: player
        for player in previous_players
    }

    current_by_id = {
        player.player_id: player
        for player in current_players
    }

    result = list(current_players)

    current_indices = {
        player.player_id: index
        for index, player in enumerate(current_players)
    }

    # --------------------------------------------------
    # STEP 1: Detect missing and newly appearing players
    # --------------------------------------------------

    previous_visible_ids = {
        p.player_id
        for p in previous_players
        if is_player_visible(p)
    }

    current_visible_ids = {
        p.player_id
        for p in current_players
        if is_player_visible(p)
    }

    # Previously visible, now a placeholder.
    missing_ids = previous_visible_ids - current_visible_ids

    # Previously placeholders, now visible.
    appeared_ids = current_visible_ids - previous_visible_ids

    # Visible in both frames.
    surviving_ids = previous_visible_ids & current_visible_ids

    # --------------------------------------------------
    # STEP 2: Find potential merged players
    # --------------------------------------------------

    # Each candidate:
    # (missing_previous_player, closest_surviving_current_player)

    merge_candidates: list[tuple[Player, Player]] = []

    for missing_id in missing_ids:

        missing_player = previous_by_id[missing_id]

        closest_neighbor = min(
            (
                previous_by_id[player_id]
                for player_id in surviving_ids
            ),
            key=lambda player: calculate_distance(
                missing_player,
                player,
            ),
            default=None,
        )

        if closest_neighbor is None:
            continue

        distance = calculate_distance(
            missing_player,
            closest_neighbor,
        )

        if distance > merge_distance:
            continue

        # Get the neighbor's position in the CURRENT frame.
        current_neighbor = current_by_id[
            closest_neighbor.player_id
        ]

        merge_candidates.append(
            (missing_player, current_neighbor)
        )

    # --------------------------------------------------
    # STEP 3: Check if newly appearing players are
    # actually previously missing players
    # --------------------------------------------------

    possible_matches = []

    for candidate_index, (missing_player, neighbor) in enumerate(
        merge_candidates
    ):
        for appeared_id in appeared_ids:

            appeared_player = current_by_id[appeared_id]

            distance = calculate_distance(
                neighbor,
                appeared_player,
            )

            if distance <= reappearance_distance:
                possible_matches.append(
                    (distance, candidate_index, appeared_id)
                )

    # Closest matches first.
    possible_matches.sort(key=lambda match: match[0])

    matched_candidates = set()
    matched_appeared_ids = set()

    for _, candidate_index, appeared_id in possible_matches:

        if candidate_index in matched_candidates:
            continue

        if appeared_id in matched_appeared_ids:
            continue

        missing_player, _ = merge_candidates[candidate_index]
        appeared_player = current_by_id[appeared_id]

        # Save the newly detected player's original ID.
        original_appeared_id = appeared_player.player_id

        # Create a copy of its previous placeholder.
        placeholder = copy(previous_by_id[original_appeared_id])
        placeholder.x = float("nan")
        placeholder.y = float("nan")

        # IMPORTANT: Create the recovered player using the
        # newly detected player's actual position and attributes.
        recovered_player = copy(appeared_player)

        # Explicitly restore the original player's identity.
        recovered_player.player_id = missing_player.player_id
        recovered_player.team = missing_player.team
        recovered_player.is_from_player_tracker_validator = True

        # Put the recovered player into the missing player's slot.
        missing_index = current_indices[missing_player.player_id]
        appeared_index = current_indices[original_appeared_id]

        result[missing_index] = recovered_player

        # The newly detected ID becomes a placeholder again.
        result[appeared_index] = placeholder

        matched_candidates.add(candidate_index)
        matched_appeared_ids.add(appeared_id)

    # --------------------------------------------------
    # STEP 4: Reconstruct players that have not reappeared
    # --------------------------------------------------

    for candidate_index, (missing_player, neighbor) in enumerate(
        merge_candidates
    ):

        if candidate_index in matched_candidates:
            continue

        # Use the closest player's CURRENT position.
        reconstructed_player = copy(neighbor)

        reconstructed_player.player_id = missing_player.player_id
        reconstructed_player.team = missing_player.team
        reconstructed_player.is_from_player_tracker_validator = True

        result[
            current_indices[missing_player.player_id]
        ] = reconstructed_player

    return result