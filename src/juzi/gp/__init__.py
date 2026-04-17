# Copyright (c) 2025, Tom Ouellette
# Licensed under the BSD 3-Clause License

# Copyright (c) 2025, Tom Ouellette
# Licensed under the BSD 3-Clause License

from ._nmf import nmf_fit
from ._prune import nmf_prune

from ._similarity import (
    similarity,
    similarity_compute,
    similarity_filter,
)

from ._cluster import (
    programs_cluster,
    programs_centroid,
    programs_progressive,
    programs_merge,
    programs_remove,
    programs_threshold,
)

from ._score import score_cells, score_classify
from ._stability import programs_stability
from ._aggregate import score_aggregate
from ._associate import score_associate
from ._annotate import programs_annotate

__all__ = [
    "nmf_fit",
    "nmf_prune",
    "similarity",
    "similarity_compute",
    "similarity_filter",
    "programs_cluster",
    "programs_centroid",
    "programs_progressive",
    "programs_merge",
    "programs_remove",
    "score_cells",
    "score_classify",
    "programs_stability",
    "score_aggregate",
    "score_associate",
    "programs_annotate",
]
