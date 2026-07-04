"""Phase 2 experiment registry.

Phase 2 starts from the completed Phase 1 benchmark framework and focuses on
turning SpatialSelf into a manuscript-ready method with clear advantages.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Phase2Experiment:
    exp_id: str
    title: str
    objective: str
    config_path: str
    output_dir: str
    status: str = "planned"


PHASE2_EXPERIMENTS = (
    Phase2Experiment(
        exp_id="p2_goal1_weak_signal_recovery",
        title="Weak-signal recovery model improvement",
        objective=(
            "Improve recovery of low-expression and subtle CCC by testing "
            "weak-positive mining, LR-wise balanced sampling, focal-style loss, "
            "and mechanism-specific recall diagnostics."
        ),
        config_path="configs/experiments/phase2_goal1_weak_signal_recovery.yaml",
        output_dir="outputs/phase2/goal1_weak_signal_recovery",
        status="implementation_started",
    ),
    Phase2Experiment(
        exp_id="p2_goal2_fdr_calibration",
        title="Permutation/FDR calibration improvement",
        objective=(
            "Reduce over-conservative significance calls while retaining "
            "negative-control calibration using alternative nulls and group-level FDR."
        ),
        config_path="configs/experiments/phase2_goal2_fdr_calibration.yaml",
        output_dir="outputs/phase2/goal2_fdr_calibration",
    ),
    Phase2Experiment(
        exp_id="p2_goal3_model_selection",
        title="Default model selection",
        objective=(
            "Select a Phase 2 default SpatialSelf model across controlled, "
            "semi-synthetic, and lymph-node validation evidence."
        ),
        config_path="configs/experiments/phase2_goal3_model_selection.yaml",
        output_dir="outputs/phase2/goal3_model_selection",
    ),
    Phase2Experiment(
        exp_id="p2_goal4_high_resolution",
        title="High-resolution real-data extension",
        objective=(
            "Run SpatialSelf on at least one high-resolution ST dataset or large "
            "subset with runtime and biological validation."
        ),
        config_path="configs/experiments/phase2_goal4_high_resolution.yaml",
        output_dir="outputs/phase2/goal4_high_resolution",
    ),
    Phase2Experiment(
        exp_id="p2_goal5_interpretability",
        title="Biological interpretability",
        objective=(
            "Validate predicted communication through downstream target response, "
            "LR embedding/module enrichment, and pathway-level summaries."
        ),
        config_path="configs/experiments/phase2_goal5_interpretability.yaml",
        output_dir="outputs/phase2/goal5_interpretability",
    ),
    Phase2Experiment(
        exp_id="p2_goal6_paper_figures",
        title="Paper-ready result tables and figures",
        objective=(
            "Assemble reproducible source tables and figure scripts for benchmark, "
            "ablation, FDR, validation, scalability, and interpretability claims."
        ),
        config_path="configs/experiments/phase2_goal6_paper_figures.yaml",
        output_dir="outputs/phase2/goal6_paper_figures",
    ),
)


def list_phase2_experiments():
    return list(PHASE2_EXPERIMENTS)


def get_phase2_experiment(exp_id):
    for experiment in PHASE2_EXPERIMENTS:
        if experiment.exp_id == exp_id:
            return experiment
    known = ", ".join(exp.exp_id for exp in PHASE2_EXPERIMENTS)
    raise KeyError(f"Unknown Phase 2 experiment: {exp_id}. Known ids: {known}")
