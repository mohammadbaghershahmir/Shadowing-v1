"""Experiment profiles and resolved architecture flags."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExperimentProfile:
    """Resolved architecture and loss switches for a training run."""

    name: str
    use_structural_stem: bool = True
    use_local_dino: bool = False
    use_global_fusion: bool = False
    use_transition_head: bool = False
    use_main_logit_boundary_loss: bool = False
    use_affinity_loss: bool = False
    use_aux_head: bool = True
    use_repetition_loss: bool = False


PROFILES: dict[str, ExperimentProfile] = {
    "BEST_V1": ExperimentProfile(
        name="BEST_V1",
        use_structural_stem=True,
        use_local_dino=True,
        use_global_fusion=True,
        use_transition_head=True,
        use_main_logit_boundary_loss=True,
        use_affinity_loss=True,
        use_aux_head=True,
        use_repetition_loss=False,
    ),
    "ABLATE_NO_DINO": ExperimentProfile(
        name="ABLATE_NO_DINO",
        use_structural_stem=True,
        use_local_dino=False,
        use_global_fusion=False,
        use_transition_head=True,
        use_main_logit_boundary_loss=True,
        use_affinity_loss=True,
        use_aux_head=True,
    ),
    "ABLATE_NO_LOCAL_DINO": ExperimentProfile(
        name="ABLATE_NO_LOCAL_DINO",
        use_structural_stem=True,
        use_local_dino=False,
        use_global_fusion=True,
        use_transition_head=True,
        use_main_logit_boundary_loss=True,
        use_affinity_loss=True,
        use_aux_head=True,
    ),
    "ABLATE_NO_GLOBAL_DINO": ExperimentProfile(
        name="ABLATE_NO_GLOBAL_DINO",
        use_structural_stem=True,
        use_local_dino=True,
        use_global_fusion=False,
        use_transition_head=True,
        use_main_logit_boundary_loss=True,
        use_affinity_loss=True,
        use_aux_head=True,
    ),
    "ABLATE_NO_TRANSITION": ExperimentProfile(
        name="ABLATE_NO_TRANSITION",
        use_structural_stem=True,
        use_local_dino=True,
        use_global_fusion=True,
        use_transition_head=False,
        use_main_logit_boundary_loss=False,
        use_affinity_loss=True,
        use_aux_head=True,
    ),
    "ABLATE_NO_AFFINITY": ExperimentProfile(
        name="ABLATE_NO_AFFINITY",
        use_structural_stem=True,
        use_local_dino=True,
        use_global_fusion=True,
        use_transition_head=True,
        use_main_logit_boundary_loss=True,
        use_affinity_loss=False,
        use_aux_head=True,
    ),
    "ABLATE_NO_STRUCTURAL_STEM": ExperimentProfile(
        name="ABLATE_NO_STRUCTURAL_STEM",
        use_structural_stem=False,
        use_local_dino=True,
        use_global_fusion=True,
        use_transition_head=True,
        use_main_logit_boundary_loss=True,
        use_affinity_loss=True,
        use_aux_head=True,
    ),
    "ABLATE_NO_AUX": ExperimentProfile(
        name="ABLATE_NO_AUX",
        use_structural_stem=True,
        use_local_dino=True,
        use_global_fusion=True,
        use_transition_head=True,
        use_main_logit_boundary_loss=True,
        use_affinity_loss=True,
        use_aux_head=False,
    ),
}


def get_profile(name: str) -> ExperimentProfile:
    if name not in PROFILES:
        raise KeyError(f"Unknown profile {name!r}. Available: {sorted(PROFILES)}")
    return PROFILES[name]
