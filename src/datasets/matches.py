from dataclasses import dataclass


@dataclass(frozen=True)
class MatchPairStats:
    pair_name: str
    matches_count: int


@dataclass(frozen=True)
class MatchSummary:
    pair_count: int
    min_matches: int
    mean_matches: float
    max_matches: int
    top_pairs: list[MatchPairStats]
