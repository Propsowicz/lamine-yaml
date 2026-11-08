import logging
from collections.abc import Sequence
from copy import copy
from math import hypot, isfinite
from typing import Any
from models.player import Player, DetectedPlayer
from models.ball import Ball
from utils.constants import PITCH_LENGTH, PITCH_WIDTH
from statistics import median
from collections import Counter, defaultdict
from uuid import uuid4
class Combiner:
    UNKNOWN_TEAMS = {"", "tbd", "unknown", "none"}

    def __init__(
        self,
        max_match_distance: float = 3.0,
        team_confidence_threshold: float = 0.90,
        normalization_search_distance: float = 6.0,
        normalization_inlier_tolerance: float = 1.5,
    ) -> None:
        self.max_match_distance = max_match_distance
        self.team_confidence_threshold = team_confidence_threshold
        self.normalization_search_distance = normalization_search_distance
        self.normalization_inlier_tolerance = normalization_inlier_tolerance
        self.logger = logging.getLogger(__name__)

    def combine(
        self,
        players: Sequence[Player],
        detected_players: Sequence[DetectedPlayer],
        ball: Ball,
        players_by_team: dict,
    ) -> list[Player]:            
        # STEP 1: Normalize detected positions.
        normalized_detections = self._normalize_detected_positions(
            players,
            detected_players,
        )

        # STEP 2: Match players.
        matches = self._match_players(
            players,
            normalized_detections,
        )

         # STEP 3: Build team mapping from matched players.
        team_mapping = self._build_detected_team_mapping(
            players,
            normalized_detections,
            matches,
        )

        matched_detections = set(matches.values())
        used_player_ids = {player.player_id for player in players}

        result: list[Player] = []

        # Process existing players.
        for player_index, player in enumerate(players):
            combined_player = copy(player)

            detection_index = matches.get(player_index)

            if detection_index is not None:
                detection = normalized_detections[detection_index]

                # Only players originally created from detections
                # can have their team updated.
                if combined_player.is_from_atg_detection is False:
                    self._reconcile_team(combined_player, detection)

                combined_player.box = detection.box

            result.append(combined_player)

        # Process unmatched detections.
        for detection_index, detection in enumerate(normalized_detections):
            if detection_index in matched_detections:
                continue

            distances = [
                (
                    player.player_id,
                    self._calculate_distance(
                        player.x,
                        player.y,
                        detection.x,
                        detection.y,
                    ),
                )
                for player in players
                if isfinite(player.x) and isfinite(player.y)
            ]

            _, nearest_distance = min(
                distances,
                key=lambda item: item[1],
                default=(None, float("inf")),
            )
      
            if nearest_distance <= self.max_match_distance:
                continue

            player_id = self._generate_player_id(
                used_player_ids,
                detection_index,
            )

            used_player_ids.add(player_id)

            inferred_team = self._infer_unmatched_team(
                detection,
                team_mapping,
            )

            new_player = self._create_player(
                detection=detection,
                player_id=player_id,
                ball=ball,
                players_by_team=players_by_team,
                team=inferred_team,
            )
            result.append(new_player)        
        
        return result

    def _build_detected_team_mapping(
        self,
        players: Sequence[Player],
        detected_players: Sequence[DetectedPlayer],
        matches: dict[int, int],
    ) -> dict[str, str]:
        """
        Maps DetectedPlayer team names to the corresponding Player team names.

        Uses already matched players as the source of truth.
        A team mapping is created only when there is a clear majority.
        """

        votes: dict[str, Counter[str]] = defaultdict(Counter)
        team_names: dict[str, str] = {}

        for player_index, detection_index in matches.items():
            player = players[player_index]
            detection = detected_players[detection_index]

            detected_team_key = self._team_key(detection.team)
            player_team_key = self._team_key(player.team)

            if detected_team_key is None or player_team_key is None:
                continue

            votes[detected_team_key][player_team_key] += 1
            team_names.setdefault(player_team_key, player.team)

        team_mapping: dict[str, str] = {}

        for detected_team, team_votes in votes.items():
            ranked_votes = team_votes.most_common()

            # Avoid making decisions when there is no clear majority.
            if (
                len(ranked_votes) > 1
                and ranked_votes[0][1] == ranked_votes[1][1]
            ):
                self.logger.warning(
                    "Ambiguous team mapping for detected team '%s': %s",
                    detected_team,
                    dict(team_votes),
                )
                continue

            player_team_key = ranked_votes[0][0]
            team_mapping[detected_team] = team_names[player_team_key]

        self.logger.info("Detected team mapping: %s", team_mapping)

        return team_mapping


    def _infer_unmatched_team(
        self,
        detection: DetectedPlayer,
        team_mapping: dict[str, str],
    ) -> str:
        """
        Infers the correct team for an unmatched detection using
        the team mapping built from already matched players.
        """

        detected_team = self._get_detection_team(detection)

        if detected_team is None:
            return "tbd"

        team_key = self._team_key(detected_team)
        mapped_team = team_mapping.get(team_key, detected_team)
        return mapped_team


    def _team_key(self, team: str | None) -> str | None:
        if not team:
            return None

        normalized_team = team.strip().casefold()

        if normalized_team in self.UNKNOWN_TEAMS:
            return None

        return normalized_team

    def _normalize_detected_positions(
            self,
            players: Sequence[Player],
            detected_players: Sequence[DetectedPlayer],
        ) -> list[DetectedPlayer]:
            """
            Estimates and corrects the global positional offset of detected players.
    
            Requires at least 3 reliable matches to apply normalization.
            Does not mutate the original DetectedPlayer objects.
            """
    
            normalized = [copy(detection) for detection in detected_players]
    
            if len(players) < 3 or len(normalized) < 3:
                return normalized
    
            # Find approximate matches using a wider distance threshold.
            rough_matches = self._match_players(
                players,
                normalized,
                max_distance=self.normalization_search_distance,
            )
    
            if len(rough_matches) < 3:
                return normalized
    
            # Calculate the positional offsets for matched players.
            offsets = [
                (
                    players[player_index].x - normalized[detection_index].x,
                    players[player_index].y - normalized[detection_index].y,
                )
                for player_index, detection_index in rough_matches.items()
            ]
    
            # Median offset is resistant to incorrectly matched players.
            median_dx = median(dx for dx, _ in offsets)
            median_dy = median(dy for _, dy in offsets)
    
            # Keep only offsets close to the estimated global translation.
            inliers = [
                (dx, dy)
                for dx, dy in offsets
                if hypot(dx - median_dx, dy - median_dy)
                <= self.normalization_inlier_tolerance
            ]
    
            if len(inliers) < 3:
                self.logger.warning(
                    "Could not normalize detected player positions: "
                    "not enough consistent matches."
                )
                return normalized
    
            dx = median(x for x, _ in inliers)
            dy = median(y for _, y in inliers)
    
            # Apply correction to all detected players.
            for detection in normalized:
                detection.x += dx
                detection.y += dy
    
            self.logger.info(
                "Normalized detected player positions: dx=%.2f, dy=%.2f "
                "(%d matched players)",
                dx,
                dy,
                len(inliers),
            )
    
            return normalized

    def _match_players(
        self,
        players: Sequence[Player],
        detected_players: Sequence[DetectedPlayer],
        max_distance: float | None = None,
    ) -> dict[int, int]:
        distance_limit = (
            max_distance
            if max_distance is not None
            else self.max_match_distance
        )

        # Build candidate detections for every metadata player.
        candidates: dict[int, list[tuple[float, int]]] = {}

        for player_index, player in enumerate(players):
            player_candidates = []

            for detection_index, detection in enumerate(detected_players):
                distance = self._calculate_distance(
                    player.x,
                    player.y,
                    detection.x,
                    detection.y,
                )

                if distance <= distance_limit:
                    player_candidates.append((distance, detection_index))

            candidates[player_index] = sorted(player_candidates)

        # Detection -> metadata player.
        detection_matches: dict[int, int] = {}

        def assign(
            player_index: int,
            visited: set[int],
        ) -> bool:
            for _, detection_index in candidates[player_index]:
                if detection_index in visited:
                    continue

                visited.add(detection_index)

                previous_player = detection_matches.get(detection_index)

                if (
                    previous_player is None
                    or assign(previous_player, visited)
                ):
                    detection_matches[detection_index] = player_index
                    return True

            return False

        # Players with fewer candidates are harder to match.
        player_order = sorted(
            candidates,
            key=lambda index: len(candidates[index]),
        )

        for player_index in player_order:
            assign(player_index, set())

        # Convert to metadata player -> detection.
        return {
            player_index: detection_index
            for detection_index, player_index in detection_matches.items()
        }

    @staticmethod
    def _calculate_distance(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> float:
        return hypot(x1 - x2, y1 - y2)

    def _reconcile_team(
        self,
        player: Player,
        detection: DetectedPlayer,
    ) -> None:
        """Changes an existing player's team only with high confidence."""
        detected_team = self._get_detection_team(detection)

        if detected_team is None or player.team == detected_team:
            return

        confidence = self._get_team_confidence(detection)

        if confidence >= self.team_confidence_threshold:
            self.logger.info(
                "Correcting team for player %s: %s -> %s (confidence: %.2f)",
                player.player_id,
                player.team,
                detected_team,
                confidence,
            )

            player.team = detected_team
        else:
            self.logger.warning(
                "Possible team anomaly for player %s: current=%s, "
                "detected=%s, confidence=%.2f. Keeping current team.",
                player.player_id,
                player.team,
                detected_team,
                confidence,
            )

    def _create_player(
        self,
        detection: DetectedPlayer,
        player_id: Any,
        ball: Ball,
        players_by_team: dict,
        team: str,
    ) -> Player:

        player = Player(
            player_id=player_id,
            x=detection.x / PITCH_LENGTH,
            y=detection.y / PITCH_WIDTH,
            speed=0.0,
            players_by_team=players_by_team,
            ball=ball,
            player_team=team,
            from_detection=True,
        )

        player.box = detection.box

        return player

    def _get_detection_team(
        self,
        detection: DetectedPlayer,
    ) -> str | None:
        """Returns the recognized team, regardless of confidence."""

        team = detection.team

        if not team or team.strip().lower() in self.UNKNOWN_TEAMS:
            return None

        return team

    @staticmethod
    def _get_team_confidence(detection: DetectedPlayer) -> float:
        try:
            confidence = float(detection.team_confidence)
        except (TypeError, ValueError):
            return 0.0

        if not isfinite(confidence):
            return 0.0

        return max(0.0, min(1.0, confidence))

    @staticmethod
    def _generate_player_id(
        used_player_ids: set[Any],
        detection_index: int,
    ) -> int | str:
        """Generates a unique ID for a newly detected player."""

        if all(
            isinstance(player_id, int) and not isinstance(player_id, bool)
            for player_id in used_player_ids
        ):
            player_id = max(used_player_ids, default=-1) + 1

            while player_id in used_player_ids:
                player_id += 1

            return player_id

        player_id = f"detected_{detection_index}"

        while player_id in used_player_ids:
            player_id += "_new"

        return player_id