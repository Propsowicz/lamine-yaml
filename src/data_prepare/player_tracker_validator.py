
from copy import copy
from math import hypot, isfinite
from models.player import Player

def is_player_visible(player: Player) -> bool:
    return isfinite(player.x) and isfinite(player.y)


def calculate_player_distance(
    player1: Player,
    player2: Player,
) -> float:
    return hypot(
        player1.x - player2.x,
        player1.y - player2.y,
    )


def make_placeholder(player: Player) -> Player:
    placeholder = copy(player)
    placeholder.x = float("nan")
    placeholder.y = float("nan")
    return placeholder


def validate_player_tracking(
    current_players: list[Player],
    previous_players: list[Player],
    max_tracking_distance: float = 3.0,
    merge_distance: float = 2.0,
    reappearance_distance: float = 2.0,
    same_id_preference: float = 0.05,
) -> list[Player]:
    """
    Validates player identities across consecutive frames.

    Handles:
    1. Incorrect ID assignments and ID swapping.
    2. Players merged by the tracker.
    3. Missing players reconstructed as virtual players.
    4. Reappearing players with previously assigned IDs.
    5. Preservation of virtual-player anchor relationships.

    All distances are measured in pitch coordinates.
    """

    if min(
        max_tracking_distance,
        merge_distance,
        reappearance_distance,
        same_id_preference,
    ) < 0:
        raise ValueError("Distance thresholds cannot be negative.")

    previous_by_id = {
        player.player_id: player
        for player in previous_players
    }

    current_by_id = {
        player.player_id: player
        for player in current_players
    }

    if len(previous_by_id) != len(previous_players):
        raise ValueError("Duplicate IDs in previous frame.")

    if len(current_by_id) != len(current_players):
        raise ValueError("Duplicate IDs in current frame.")

    current_indices = {
        player.player_id: index
        for index, player in enumerate(current_players)
    }

    # --------------------------------------------------
    # STEP 1: Initialize result with placeholders.
    # --------------------------------------------------

    result_by_id = {
        player.player_id: make_placeholder(player)
        for player in previous_players
    }

    previous_anchor_ids = {
        player.player_id: getattr(
            player,
            "validator_anchor_id",
            None,
        )
        for player in previous_players
    }

    # Previous real players. Virtual players are excluded
    # because they share coordinates with their anchors.
    previous_real_players = [
        player
        for player in previous_players
        if (
            is_player_visible(player)
            and previous_anchor_ids[player.player_id] is None
        )
    ]

    current_visible_players = [
        player
        for player in current_players
        if is_player_visible(player)
    ]

    # --------------------------------------------------
    # STEP 2: GLOBAL IDENTITY MATCHING.
    #
    # Compare ALL previously visible real players with
    # ALL current detections, regardless of their IDs.
    #
    # This is what fixes the ID 11 / ID 31 / ID 33 case.
    # --------------------------------------------------

    possible_matches = []

    for previous_player in previous_real_players:
        for detected_player in current_visible_players:

            distance = calculate_player_distance(
                previous_player,
                detected_player,
            )

            if distance > max_tracking_distance:
                continue

            # Small preference for unchanged IDs.
            # Distance remains the primary matching criterion.
            score = distance

            if (
                previous_player.player_id
                != detected_player.player_id
            ):
                score += same_id_preference

            possible_matches.append(
                (
                    score,
                    distance,
                    previous_player.player_id,
                    detected_player.player_id,
                )
            )

    # Closest global matches first.
    possible_matches.sort(key=lambda match: (match[0], match[1]))

    matched_previous_ids: set[int] = set()
    matched_detection_ids: set[int] = set()

    for _, distance, previous_id, detected_id in possible_matches:

        if previous_id in matched_previous_ids:
            continue

        if detected_id in matched_detection_ids:
            continue

        previous_player = previous_by_id[previous_id]
        detected_player = current_by_id[detected_id]

        # Use the actual detection's coordinates and attributes,
        # but restore the previous player's identity.
        recovered_player = copy(detected_player)

        recovered_player.player_id = previous_id
        recovered_player.team = previous_player.team

        recovered_player.is_from_player_tracker_validator = (
            getattr(
                previous_player,
                "is_from_player_tracker_validator",
                False,
            )
            or previous_id != detected_id
        )

        # A globally matched player is a real detection,
        # not a virtual player.
        recovered_player.validator_anchor_id = None

        result_by_id[previous_id] = recovered_player

        matched_previous_ids.add(previous_id)
        matched_detection_ids.add(detected_id)

    # --------------------------------------------------
    # STEP 3: Collect existing virtual players.
    #
    # They must continue following their original anchors.
    # --------------------------------------------------

    targets: dict[int, tuple[Player, int]] = {}

    for previous_player in previous_players:

        anchor_id = previous_anchor_ids[
            previous_player.player_id
        ]

        if anchor_id is None:
            continue

        targets[previous_player.player_id] = (
            previous_player,
            anchor_id,
        )

    # --------------------------------------------------
    # STEP 4: Find newly missing real players.
    #
    # A previously visible player that could not be matched
    # may have been merged into another nearby player.
    # --------------------------------------------------

    surviving_players = [
        player
        for player in previous_real_players
        if player.player_id in matched_previous_ids
    ]

    for missing_player in previous_real_players:

        missing_id = missing_player.player_id

        if missing_id in matched_previous_ids:
            continue

        possible_anchors = [
            player
            for player in surviving_players
            if (
                player.player_id != missing_id
                and is_player_visible(
                    result_by_id[player.player_id]
                )
            )
        ]

        if not possible_anchors:
            continue

        closest_anchor = min(
            possible_anchors,
            key=lambda player: calculate_player_distance(
                missing_player,
                player,
            ),
        )

        distance = calculate_player_distance(
            missing_player,
            closest_anchor,
        )

        if distance > merge_distance:
            continue

        targets[missing_id] = (
            missing_player,
            closest_anchor.player_id,
        )


    # --------------------------------------------------
    # STEP 5: Find detections not used by global matching.
    #
    # These may belong to existing virtual players or
    # newly appearing players.
    #
    # Do not reuse detections whose original ID has already
    # been matched to another detection.
    # --------------------------------------------------

    available_detections = [
        player
        for player in current_visible_players
        if (
            player.player_id not in matched_detection_ids
            and player.player_id not in matched_previous_ids
        )
    ]

    # --------------------------------------------------
    # STEP 6: Match available detections with virtual players.
    #
    # Use the CURRENT anchor position.
    # --------------------------------------------------

    possible_virtual_matches = []

    for target_id, (target_player, anchor_id) in targets.items():

        anchor = result_by_id.get(anchor_id)

        if anchor is None or not is_player_visible(anchor):
            anchor = previous_by_id.get(anchor_id)

        if anchor is None or not is_player_visible(anchor):
            continue

        for detected_player in available_detections:

            distance = calculate_player_distance(
                anchor,
                detected_player,
            )

            if distance > reappearance_distance:
                continue

            score = distance

            if detected_player.player_id != target_id:
                score += same_id_preference

            possible_virtual_matches.append(
                (
                    score,
                    distance,
                    target_id,
                    detected_player.player_id,
                )
            )

    possible_virtual_matches.sort(
        key=lambda match: (match[0], match[1])
    )

    matched_targets: set[int] = set()
    matched_virtual_detections: set[int] = set()

    for _, distance, target_id, detected_id in possible_virtual_matches:

        if target_id in matched_targets:
            continue

        if detected_id in matched_virtual_detections:
            continue

        target_player, _ = targets[target_id]
        detected_player = current_by_id[detected_id]

        recovered_player = copy(detected_player)

        recovered_player.player_id = target_id
        recovered_player.team = target_player.team
        recovered_player.is_from_player_tracker_validator = True
        recovered_player.validator_anchor_id = None

        result_by_id[target_id] = recovered_player

        matched_targets.add(target_id)
        matched_virtual_detections.add(detected_id)


    # --------------------------------------------------
    # STEP 7: Preserve unmatched virtual players.
    #
    # They continue following their original anchors.
    # --------------------------------------------------

    for target_id, (target_player, anchor_id) in targets.items():

        if target_id in matched_targets:
            continue

        anchor = result_by_id.get(anchor_id)

        if anchor is None or not is_player_visible(anchor):
            anchor = previous_by_id.get(anchor_id)

        if anchor is not None and is_player_visible(anchor):
            virtual_player = copy(anchor)
        else:
            virtual_player = copy(target_player)

        virtual_player.player_id = target_id
        virtual_player.team = target_player.team

        virtual_player.is_from_player_tracker_validator = True
        virtual_player.validator_anchor_id = anchor_id

        result_by_id[target_id] = virtual_player

    # --------------------------------------------------
    # STEP 8: Accept genuinely new players.
    #
    # Remaining detections can retain their own ID only
    # when that ID was a placeholder in the previous frame.
    # --------------------------------------------------

    used_virtual_detections = matched_virtual_detections

    for detected_player in available_detections:

        detected_id = detected_player.player_id

        if detected_id in used_virtual_detections:
            continue

        # Never overwrite an identity already matched or
        # a virtual identity that must remain connected.
        if detected_id in matched_previous_ids:
            continue

        if detected_id in targets:
            continue

        previous_player = previous_by_id[detected_id]

        if is_player_visible(previous_player):
            continue

        new_player = copy(detected_player)

        new_player.player_id = detected_id
        new_player.team = previous_player.team

        new_player.is_from_player_tracker_validator = (
            getattr(
                previous_player,
                "is_from_player_tracker_validator",
                False,
            )
        )
        new_player.validator_anchor_id = None

        result_by_id[detected_id] = new_player

    # --------------------------------------------------
    # STEP 9: Return exactly the original player slots.
    # --------------------------------------------------

    return [
        result_by_id[player.player_id]
        for player in current_players
    ]
