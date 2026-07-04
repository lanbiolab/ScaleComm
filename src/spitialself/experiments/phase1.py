"""Phase 1 experiment registry.

This registry makes the first manuscript-rescue phase explicit in code. Each
entry points to the runnable config/output location and records the current
evidence status used by the Phase 1 audit.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Phase1Experiment:
    exp_id: str
    title: str
    objective: str
    config_path: str
    output_dir: str
    status: str = "evidence_complete"


PHASE1_EXPERIMENTS = (
    Phase1Experiment(
        exp_id="exp01_controlled_benchmark",
        title="Controlled ground-truth CCC simulation benchmark",
        objective=(
            "Evaluate recovery of juxtacrine, paracrine, and chemokine-like "
            "CCC under known ground truth, including weak signals and false "
            "positive high-expression LR pairs."
        ),
        config_path="configs/experiments/exp01_controlled_benchmark.yaml",
        output_dir="outputs/phase1/exp01_controlled_benchmark",
    ),
    Phase1Experiment(
        exp_id="exp02_semisynthetic_real_background",
        title="Real-coordinate and real-expression semi-synthetic benchmark",
        objective=(
            "Inject known LR signals into real ST coordinate and expression "
            "backgrounds to test robustness under realistic tissue structure."
        ),
        config_path="configs/experiments/exp02_semisynthetic_real_background.yaml",
        output_dir="outputs/phase1/exp02_semisynthetic_real_background",
    ),
    Phase1Experiment(
        exp_id="exp03_baseline_comparison",
        title="Traditional and deep-learning CCC baseline comparison",
        objective=(
            "Compare SpatialSelf against COMMOT, CellChat, NICHES, CellNEST, "
            "DeepTalk, SpaTalk, and downstream NicheNet validation under shared "
            "candidate sets and LR database."
        ),
        config_path="configs/experiments/exp03_baseline_comparison.yaml",
        output_dir="outputs/phase1/exp03_baseline_comparison",
    ),
    Phase1Experiment(
        exp_id="exp04_model_ablation",
        title="Model component ablation",
        objective=(
            "Quantify the contribution of LR identity, GAT/spatial graph, "
            "distance, expression, and pseudo-label strategy."
        ),
        config_path="configs/experiments/exp04_model_ablation.yaml",
        output_dir="outputs/phase1/exp04_model_ablation",
    ),
    Phase1Experiment(
        exp_id="exp05_weak_signal_false_positive",
        title="Weak-signal recovery and high-expression false-positive suppression",
        objective=(
            "Stress-test weak LR recovery and suppression of non-permissive "
            "high-expression LR pairs."
        ),
        config_path="configs/experiments/exp05_weak_signal_false_positive.yaml",
        output_dir="outputs/phase1/exp05_weak_signal_false_positive",
    ),
    Phase1Experiment(
        exp_id="exp06_permutation_fdr",
        title="Permutation p-value and FDR control",
        objective=(
            "Estimate empirical p-values and BH-adjusted FDR using spatial, "
            "receiver, and LR-identity permutation null models."
        ),
        config_path="configs/experiments/exp06_permutation_fdr.yaml",
        output_dir="outputs/phase1/exp06_permutation_fdr",
    ),
    Phase1Experiment(
        exp_id="exp07_lymph_node_global_validation",
        title="Human lymph node global validation",
        objective=(
            "Evaluate all LR pairs with region enrichment, downstream target "
            "correlation, FDR, and CCR7-CCL19 case visualization."
        ),
        config_path="configs/experiments/exp07_lymph_node_global_validation.yaml",
        output_dir="outputs/phase1/exp07_lymph_node_global_validation",
    ),
    Phase1Experiment(
        exp_id="exp08_runtime_memory",
        title="Runtime and memory benchmark",
        objective=(
            "Measure scalability across cell/spot counts, candidate edges, "
            "LR pairs, wall-clock time, CPU memory, and GPU memory."
        ),
        config_path="configs/experiments/exp08_runtime_memory.yaml",
        output_dir="outputs/phase1/exp08_runtime_memory",
    ),
)


def list_phase1_experiments():
    return list(PHASE1_EXPERIMENTS)


def get_phase1_experiment(exp_id):
    for experiment in PHASE1_EXPERIMENTS:
        if experiment.exp_id == exp_id:
            return experiment
    known = ", ".join(exp.exp_id for exp in PHASE1_EXPERIMENTS)
    raise KeyError(f"Unknown Phase 1 experiment: {exp_id}. Known ids: {known}")
