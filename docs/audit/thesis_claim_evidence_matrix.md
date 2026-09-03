# Thesis Claim Evidence Matrix

Audit: `AUDIT01 CLAIM-EVIDENCE MATRIX`

Amendment: `AUDIT01 CLAIM-EVIDENCE MATRIX DETERMINISTIC AMENDMENT ONLY`

Audit date: 2026-09-03

Repository: `D:\CS\KLTN\gnnexplainer-groundtruth-benchmark`

Audited start commit: `ed636c86af2065950a71728ef0f0486199b8b2f1`

Scope: Audit01 only, governed by `D:\CS\KLTN\docs\before_task28_audit_all_design\THESIS_DESIGN_AUDIT\00_README_AND_GLOBAL_RULES.md` and `D:\CS\KLTN\docs\before_task28_audit_all_design\THESIS_DESIGN_AUDIT\01_RESEARCH_QUESTIONS_AND_CLAIMS.md`.

No experiment was rerun. No model checkpoint, dataset, metric value, target list, config, or result artifact was changed.

## Status And Verification Legend

| Label | Meaning |
|---|---|
| SUPPORTED | The claim is supported by current repository evidence, experiment artifacts, or governing protocol within the stated scope. |
| CONDITIONAL | The claim is valid only with explicit boundaries or pending audit dependencies. |
| UNSUPPORTED | The claim should not be made from current evidence. |
| UNKNOWN | The claim cannot be traced to evidence in this audit. |
| PROTOCOL_ASSERTED | A governing document requires the property, but artifact-level equivalence is not verified in this audit. |
| ARTIFACT_VERIFIED | An existing artifact, hash, report, or code symbol was checked in this audit. |
| PENDING_AUDIT | A later audit must verify the property before final comparability claims. |
| NOT_APPLICABLE | The property does not apply to the dataset or method. |

Severity scale follows Audit00: L1 = documentation only, L2 = deterministic audit/re-aggregation, L3 = protocol amendment before an unrun experiment, L4 = existing-result validity blocker, L5 = thesis-design blocker or unsupported core claim.

## Evidence Index

| Evidence ID | Evidence type | Exact source | Location | Commit/hash | Supported claim | Scope/limitation |
|---|---|---|---|---|---|---|
| EV-GP-001 | GOVERNING_PROTOCOL | `D:\CS\KLTN\docs\before_task28_audit_all_design\THESIS_DESIGN_AUDIT\00_README_AND_GLOBAL_RULES.md` | `# Cross-task methodological audit - execution index`; `## Mandatory explanation and evidence rule`; `## Severity` | external design file, not repo-tracked here | Audit taxonomy, severity definitions, one-file-per-turn boundary. | Governs audit wording, not experiment correctness. |
| EV-GP-002 | GOVERNING_PROTOCOL | `D:\CS\KLTN\docs\before_task28_audit_all_design\THESIS_DESIGN_AUDIT\01_RESEARCH_QUESTIONS_AND_CLAIMS.md` | `# Audit 01 - Research questions, hypotheses, and claim boundaries`; `## Deliverable` | external design file, not repo-tracked here | Required claim-evidence matrix and term audit. | Does not itself verify experiment artifacts. |
| EV-GP-003 | GOVERNING_PROTOCOL | `D:\CS\KLTN\docs\super_plan_19082026\SUPER_PLAN_REVISED_TASK20_TO_33_V3\SUPER_PLAN_REVISED_TASK20_TO_33_V3\05_TASK20_BA_HOUSE_2X2_FREEZE.md` | `## Interpretation hierarchy - mandatory`; `## Statistics discipline`; `## Acceptance` | external design file, not repo-tracked here | BA-House within-backbone primary comparison, cross-backbone non-causal boundary, Random outside 2x2. | Protocol assertion only unless paired with BA-House freeze/report artifacts. |
| EV-GP-004 | GOVERNING_PROTOCOL | `D:\CS\KLTN\docs\super_plan_19082026\SUPER_PLAN_REVISED_TASK20_TO_33_V3\SUPER_PLAN_REVISED_TASK20_TO_33_V3\08_TASK23_SHAPEGGEN_GNNEXPLAINER_RANDOM.md` | `## Goal`; `## Random sanity baseline`; `## GNNExplainer stability` | external design file, not repo-tracked here | ShapeGGen GNNExplainer and Random protocol, 20 percent budget, random aggregation rule. | Candidate/evaluator equivalence remains pending Audit05/Audit06. |
| EV-GP-005 | GOVERNING_PROTOCOL | `D:\CS\KLTN\docs\super_plan_19082026\SUPER_PLAN_REVISED_TASK20_TO_33_V3\SUPER_PLAN_REVISED_TASK20_TO_33_V3\09_TASK24_SHAPEGGEN_PGEXPLAINER.md` | `## Goal`; `## Stability`; `## Acceptance` | external design file, not repo-tracked here | ShapeGGen PGExplainer common train/validation protocol and stability scope. | Artifact-level training-pool and evaluator equivalence remains pending later audits. |
| EV-GP-006 | GOVERNING_PROTOCOL | `D:\CS\KLTN\docs\super_plan_19082026\SUPER_PLAN_REVISED_TASK20_TO_33_V3\SUPER_PLAN_REVISED_TASK20_TO_33_V3\10_TASK25_SHAPEGGEN_2X2_CROSS_SYNTHETIC.md` | `## Interpretation hierarchy`; `## Cross-synthetic rule`; `## Generalization wording` | external design file, not repo-tracked here | Cross-synthetic pattern-consistency boundary and no pooled cross-dataset pseudo-summary. | Does not prove artifact-level equivalence inside each source task. |
| EV-GP-007 | GOVERNING_PROTOCOL | `D:\CS\KLTN\docs\super_plan_19082026\SUPER_PLAN_REVISED_TASK20_TO_33_V3\SUPER_PLAN_REVISED_TASK20_TO_33_V3\11_TASK26_ELLIPTIC_HANDOFF_PARITY_PROTOCOL.md` | `## Explanation evaluation protocol`; `## Elliptic ground-truth boundary` | external design file, not repo-tracked here | Elliptic has no explanatory-edge ground truth, Edge AUROC/AP disabled. | Full PGExplainer Elliptic final evidence is still pending. |
| EV-GP-008 | GOVERNING_PROTOCOL | `D:\CS\KLTN\docs\super_plan_19082026\SUPER_PLAN_REVISED_TASK20_TO_33_V3\SUPER_PLAN_REVISED_TASK20_TO_33_V3\12_TASK27_ELLIPTIC_GNNEXPLAINER.md` | `## Goal`; `## No GT explanation metrics`; `## Stability`; `## Cross-backbone interpretation boundary` | external design file, not repo-tracked here | Elliptic GNNExplainer scope, no Edge AUROC/AP, stability summary unit, no causal cross-backbone wording. | Governs Task27, not Task28 PGExplainer final result. |
| EV-GP-009 | GOVERNING_PROTOCOL | `D:\CS\KLTN\docs\super_plan_19082026\SUPER_PLAN_REVISED_TASK20_TO_33_V3\SUPER_PLAN_REVISED_TASK20_TO_33_V3\13_TASK28_ELLIPTIC_PGEXPLAINER.md` | `## Goal`; `## No GT explanation metrics`; `## Cross-backbone fairness audit`; `## Runtime`; `## STOP` | external design file, not repo-tracked here | Elliptic PGExplainer intended protocol, common pools, no Edge AUROC/AP, runtime decomposition. | Step04 and final training were not started in this audit. |
| EV-ART-001 | EXPERIMENT_ARTIFACT | `docs/ba_house_2x2_evaluation_report.md` | `## Scope And Freeze Status`; `## A. Primary 2x2 Ground-Truth Agreement`; `## C. Stability`; `## Statistical Discipline`; `## Acceptance Checklist` | SHA256 `A6AE7C896AEB018128D8AA673EB2FE0470A3CAB7CCEA98C7B185148177E15A86`; commit `ed636c86` | BA-House frozen summaries, target hash status, top-20 percent budget, node-level stability aggregation. | Artifact-level candidate/evaluator equivalence across all methods is deferred to Audit05/Audit06. |
| EV-ART-002 | EXPERIMENT_ARTIFACT | `docs/ba_house_2x2_freeze.json` | JSON keys `target_hash_verified`, `target_hash`, `evaluation_protocol`, `source_hashes` | repo artifact at commit `ed636c86`; target hash `2b1fa0cb70a4aaa107dfbfe864042ddb488372a6bcfd9ea6fd4c04bd1894526f` | BA-House frozen targets and protocol metadata. | This audit did not recompute every source artifact from raw result files. |
| EV-ART-003 | EXPERIMENT_ARTIFACT | `docs/shapeggen_2x2_report.md` | `## Scope`; `## Protocol`; `## Stability`; `## Thesis-Ready Findings`; `## Stop Boundary` | SHA256 `6B1924222DD92F3DB38E2E68F9BC063E49B60728691CFC9AC3E3335D008CC357`; commit `ed636c86` | ShapeGGen frozen 2x2 result summaries and wording boundaries. | Candidate/evaluator equivalence remains pending Audit05/Audit06. |
| EV-ART-004 | EXPERIMENT_ARTIFACT | `docs/cross_synthetic_ba_house_shapeggen_report.md` | `## Dataset Context`; `## Conclusion-Consistency Matrix`; `## Generalization Statement`; `## Methodological Guardrails` | SHA256 `FD5060A48C2744D548BBBB630AB2EEAF60939173FCF3A5A0EBC01939915914C3`; commit `ed636c86` | Cross-synthetic pattern claims and no-pooling boundary. | Supports pattern consistency only, not universal generalization. |
| EV-ART-005 | EXPERIMENT_ARTIFACT | `docs/cross_synthetic_freeze.json` | JSON keys `frozen_dataset_config_checkpoint_target_hashes`, `metric_semantics`, `random_aggregation_rule`, `stability_sources_and_pair_counts` | repo artifact at commit `ed636c86` | ShapeGGen and BA-House cross-synthetic freeze metadata. | Later audits must verify lower-level source artifact equivalence. |
| EV-ART-006 | EXPERIMENT_ARTIFACT | `docs/elliptic_xai_protocol.md` | `## Dataset Semantics`; `## Explanation Metrics`; `## Frozen Targets And PG Pools`; `## Stop Boundary` | SHA256 `9D9274B26FD5DCBD5A0E305729F6C2531959A54D93214B93E1A1E2678C61190C`; commit `ed636c86` | Elliptic class/explanation boundary, selected-edge ratio, final target and PG pool counts. | Protocol/report evidence, not causal explanation ground truth. |
| EV-ART-007 | EXPERIMENT_ARTIFACT | `docs/elliptic_gnnexplainer_report.md` | `## Optimization Target Semantics`; `## Candidate Scope And Budget`; `## Stability Audit`; `## Interpretation Boundary And Limitations` | SHA256 `5780C5909602BEEB485363977809976425D7E6B73E9B36DC561E1B2D10025B42`; commit `ed636c86` | Elliptic GNNExplainer completed model-importance evaluation and stability unit hierarchy. | Does not include Elliptic PGExplainer final comparison. |
| EV-ART-008 | EXPERIMENT_ARTIFACT | `docs/elliptic_pgexplainer_step03_evidence_expansion_report.md` | `## Scope`; `## Reviewed Tolerance Amendment`; `## Pool And Leakage Audit`; `## Verdict` | SHA256 `95BE451926861781D6ECBADD73536B0F76C9377351D6AF215DA4270EB2695150`; commit `ed636c86` | Task28 Step03 pilot evidence pass and final-target isolation. | Step04 remains unauthorized; no final PGExplainer Elliptic result exists. |
| EV-CODE-001 | REPOSITORY_EVIDENCE | `src/gnnexplainer_benchmark/evaluation/budget.py` | `selected_edge_count_for_ratio`; `select_top_ratio_edges` | commit `ed636c86` | 20 percent selected-edge budget code symbol and rounding rule. | Code presence does not prove all historical artifacts used it. |
| EV-CODE-002 | REPOSITORY_EVIDENCE | `src/gnnexplainer_benchmark/evaluation/edge_ground_truth.py` | `candidate_edge_scores_and_labels` | commit `ed636c86` | Candidate-edge score/label construction symbol. | Cross-method equivalence remains pending Audit05. |
| EV-CODE-003 | REPOSITORY_EVIDENCE | `src/gnnexplainer_benchmark/evaluation/fidelity.py` | `compute_fidelity` | commit `ed636c86` | Remove/retain perturbation metric symbol. | Evaluator equivalence across methods remains pending Audit06. |
| EV-CODE-004 | REPOSITORY_EVIDENCE | `src/gnnexplainer_benchmark/evaluation/stability.py` | `pairwise_jaccard_rows`; `summarize_stability_rows` | commit `ed636c86` | Stability pair and per-node aggregation symbols. | Full stability artifact audit remains pending Audit07/Audit08. |
| EV-CODE-005 | REPOSITORY_EVIDENCE | `scripts/run_shapeggen_gnnexplainer_random.py` | `run_random_once`; `run_random_quality_and_stability`; `aggregate_random_rows_by_node` | commit `ed636c86` | ShapeGGen Random implementation and node-level aggregation symbols. | Sampling semantics need later artifact/code-level audit before final Random comparability claims. |
| EV-CODE-006 | REPOSITORY_EVIDENCE | `scripts/build_gcn_explainer_comparison.py` | `build_random_aggregation_audit_rows`; `runtime_semantics` | commit `ed636c86` | BA-House GCN three-way comparison aggregation and runtime wording. | Applies to GCN BA-House reporting scope only. |
| EV-REPO-001 | REPOSITORY_EVIDENCE | `README.md` | lines containing planned scaffold and `faithfulness` wording found by repository search | SHA256 `44A6C62785CB259201EC6AD14E4B86211030EA650D1092488B8C158DBC97016B`; commit `ed636c86` | Documentation debt finding. | README was not edited in this turn. |
| EV-THESIS-001 | REPOSITORY_EVIDENCE | `D:\CS\KLTN\Luận\Luận.docx` | Chapter 1 draft sections `1.5`, `1.6`, and Elliptic scope paragraph extracted read-only | local thesis draft, not repo-tracked | Thesis scope separates synthetic GT evaluation from Elliptic model-behavior analysis. | Page numbers were not stable in this audit. |
| EV-THESIS-002 | REPOSITORY_EVIDENCE | `D:\CS\KLTN\Luận\Chuong_1_2_Thesis_Hoan_Chinh.docx` | Chapter 2 discussion of GNNExplainer, PGExplainer, GraphXAI, fidelity/faithfulness | local thesis draft, not repo-tracked | Literature framing and terminology debt. | DOCX was not edited in this turn. |
| EV-THESIS-003 | REPOSITORY_EVIDENCE | `D:\CS\KLTN\Luận\Chuong_3_Phuong_phap_nghien_cuu.docx` | Section `3.1`; dataset and metric protocol paragraphs | local thesis draft, not repo-tracked | Current methodology chapter separates synthetic GT from Elliptic no-GT setting. | DOCX was not edited in this turn. |
| EV-PAPER-001 | EXTERNAL_SOURCE | Ying, Bourgeois, You, Zitnik, and Leskovec, `GNNExplainer: Generating Explanations for Graph Neural Networks` | NeurIPS 2019; arXiv submitted 2019-03-10, revised 2019-11-13; URL https://arxiv.org/abs/1903.03894 | DOI: `10.48550/arXiv.1903.03894`; accessed 2026-09-03 | Supports the general definition of GNNExplainer as instance-level subgraph/feature explanation optimization for GNN predictions. | Does not prove repository implementation correctness or metric comparability. |
| EV-PAPER-002 | EXTERNAL_SOURCE | Luo, Cheng, Xu, Yu, Zong, Chen, and Zhang, `Parameterized Explainer for Graph Neural Network` | NeurIPS 2020; arXiv submitted 2020-11-09; URL https://arxiv.org/abs/2011.04573 | DOI: `10.48550/arXiv.2011.04573`; accessed 2026-09-03 | Supports the general definition of PGExplainer as a parameterized explainer trained for multiple instances. | Does not prove this repository's PG training, checkpointing, or candidate-edge protocol. |
| EV-PAPER-003 | EXTERNAL_SOURCE | Kipf and Welling, `Semi-Supervised Classification with Graph Convolutional Networks` | ICLR 2017; arXiv submitted 2016-09-09, revised 2017-02-22; URL https://arxiv.org/abs/1609.02907 | DOI: `10.48550/arXiv.1609.02907`; accessed 2026-09-03 | Supports the general GCN backbone definition. | Does not prove local checkpoint quality, fairness, or implementation parity. |
| EV-PAPER-004 | EXTERNAL_SOURCE | Hamilton, Ying, and Leskovec, `Inductive Representation Learning on Large Graphs` | NeurIPS/NIPS 2017; arXiv submitted 2017-06-07, revised 2018-09-10; URL https://arxiv.org/abs/1706.02216 | DOI: `10.48550/arXiv.1706.02216`; accessed 2026-09-03 | Supports the general GraphSAGE backbone definition and aggregation-based inductive framework. | Does not prove local GraphSAGE checkpoint fairness or comparability. |
| EV-PAPER-005 | EXTERNAL_SOURCE | Agarwal, Queen, Lakkaraju, and Zitnik, `Evaluating Explainability for Graph Neural Networks` | 2022 arXiv submission, 2023 revision | DOI: `10.48550/arXiv.2208.09339`; URL https://arxiv.org/abs/2208.09339; accessed 2026-09-03 | Supports ShapeGGen/GraphXAI motivation for synthetic datasets with ground-truth explanations. | Does not prove this repository's ShapeGGen artifact or ground-truth mapping without local audits. |

## Control Verification Matrix

| Control | Current state | Evidence IDs | Audit dependency |
|---|---|---|---|
| Frozen BA-House targets and target hash. | ARTIFACT_VERIFIED | EV-ART-001, EV-ART-002 | None for claim-boundary level. |
| Frozen ShapeGGen final targets. | PROTOCOL_ASSERTED plus report-level evidence | EV-GP-004, EV-GP-005, EV-ART-003, EV-ART-005 | Audit04 for target leakage and target selection details. |
| Same candidate-edge set within each method/backbone/dataset comparison. | PROTOCOL_ASSERTED | EV-GP-003, EV-GP-004, EV-GP-005, EV-CODE-002 | PENDING_AUDIT05. |
| Continuous-score Edge AUROC/AP over full candidate-edge set. | PROTOCOL_ASSERTED plus report-level evidence | EV-ART-001, EV-ART-003, EV-ART-004, EV-CODE-002 | PENDING_AUDIT05 and PENDING_AUDIT06 before final artifact-level comparability. |
| Remove/retain evaluator and reference-class equivalence. | PROTOCOL_ASSERTED | EV-CODE-003, EV-ART-001, EV-ART-003, EV-ART-006, EV-ART-007 | PENDING_AUDIT06. |
| Fixed 20 percent selected-edge budget. | ARTIFACT_VERIFIED at report/config level | EV-ART-001, EV-ART-003, EV-ART-006, EV-ART-007, EV-CODE-001 | Audit06 should verify evaluator use in raw artifacts. |
| Stability repetition and stochastic-source equivalence. | PROTOCOL_ASSERTED plus report-level evidence | EV-GP-003, EV-GP-006, EV-GP-008, EV-GP-009, EV-CODE-004 | PENDING_AUDIT07 and PENDING_AUDIT08. |
| Random sanity baseline candidate/evaluator equivalence. | PROTOCOL_ASSERTED plus code-symbol evidence | EV-GP-003, EV-GP-004, EV-CODE-005, EV-CODE-006 | PENDING_AUDIT05 and PENDING_AUDIT06. |
| Elliptic final-target isolation before Task28 Step04. | ARTIFACT_VERIFIED for Step03 evidence | EV-ART-008 | None for Step03 claim. Step04 still unauthorized. |
| Elliptic Edge AUROC/AP disabled. | ARTIFACT_VERIFIED at protocol/report level | EV-GP-007, EV-GP-008, EV-GP-009, EV-ART-006, EV-ART-007 | None for Audit01 claim boundary. |

## Research Question And Claim Matrix

| ID | RQ, hypothesis, or claim | Status | Dataset/backbone/explainer/metric | Independent variable | Controls and current verification state | Uncontrolled variables | Unit of analysis | Evidence | Valid conclusion | Forbidden conclusion | Severity/remediation |
|---|---|---|---|---|---|---|---|---|---|---|---|
| RQ1 | Compare GNNExplainer and PGExplainer on synthetic datasets with ground-truth explanation edges. | CONDITIONAL - pending Audit05 candidate-edge equivalence and Audit06 perturbation/metric semantics. | BA-House and ShapeGGen; GCN and GraphSAGE; GNNExplainer and PGExplainer; Edge AUROC/AP, Fid+, Fid-, 1-Fid-, sparsity, stability, runtime. | Explainer method within a fixed dataset and backbone. | Frozen targets: ARTIFACT_VERIFIED at report level for BA-House and report-level/protocol-level for ShapeGGen. Candidate scope: PENDING_AUDIT05. Evaluator semantics: PENDING_AUDIT06. Budget: report-level ARTIFACT_VERIFIED. | Dataset generator, target count, candidate-edge distribution, GT prevalence, backbone predictive quality. | Target node for primary quality summaries. | EV-GP-003, EV-GP-004, EV-GP-005, EV-ART-001, EV-ART-003, EV-CODE-001, EV-CODE-002, EV-CODE-003. | Current evidence supports completed within-dataset reports. Final comparability status remains pending Audit05/Audit06. | Do not claim universal explainer superiority or final artifact-level method equivalence before Audit05/Audit06. | L2 dependency: deterministic audit of candidate/evaluator equivalence. |
| RQ2 | Assess whether ground-truth agreement and perturbation-based fidelity answer different questions. | CONDITIONAL - pending Audit05/Audit06. | Synthetic datasets only; Edge AUROC/AP versus Fid+/Fid-/1-Fid-. | Metric family. | GT mapping: PENDING_AUDIT05. Continuous-score candidate alignment: PENDING_AUDIT05. Remove/retain semantics and reference class: PENDING_AUDIT06. | Model may learn signals different from synthetic generator; perturbations may create out-of-distribution graphs. | Target node, then aggregate over target nodes. | EV-ART-001, EV-ART-003, EV-ART-004, EV-CODE-002, EV-CODE-003, EV-PAPER-005. | Frozen reports record that ranking agreement and perturbation behavior are distinct metric families. Full confidence depends on Audit05/Audit06. | Do not use AUROC/AP as a substitute for fidelity or Fid+/Fid- as proof of ground-truth correctness. | L2 dependency: metric identity and evaluator audit. |
| RQ3 | Evaluate stability of explanations. | CONDITIONAL - pending Audit07 seed/stability semantics and Audit08 statistical aggregation. | BA-House, ShapeGGen, Elliptic GNNExplainer; Jaccard stability; GNNExplainer, PGExplainer, Random where available. | Repeated explainer seeds, repeated PG checkpoints, or repeated random selections. | Stability metric: PROTOCOL_ASSERTED. Pair nesting and node-level aggregation: report-level evidence. Full repetition equivalence: PENDING_AUDIT07. Statistical aggregation: PENDING_AUDIT08. | Method-specific stochastic sources differ; stability is not adversarial robustness. | Raw measurement unit: unordered seed pair within a target node. Intermediate aggregation: mean pairwise Jaccard within each target node. Final statistical/summary unit: target node. Seed-pair rows are nested within target nodes and are not treated as independent node-level observations. | EV-GP-003, EV-GP-006, EV-GP-008, EV-GP-009, EV-ART-001, EV-ART-004, EV-ART-007, EV-CODE-004. | Frozen reports record higher mean Jaccard for PGExplainer under their respective protocols; equivalence of repetitions, stochastic sources, and aggregation remains pending Audit07/Audit08. | Do not claim Random is universally lowest or that higher Jaccard means complete robustness. | L2 dependency: deterministic stability and aggregation audit. |
| RQ4 | Study cross-synthetic pattern consistency. | CONDITIONAL. | BA-House versus ShapeGGen, GCN and GraphSAGE, GNNExplainer and PGExplainer. | Dataset context. | No pooling across datasets: report-level ARTIFACT_VERIFIED. Source task comparability: PENDING_AUDIT05/AUDIT06/AUDIT08. | Dataset size, graph construction, target count, candidate-edge distribution, GT prevalence. | Dataset-level pattern, not pooled target rows. | EV-GP-006, EV-ART-004, EV-ART-005. | Some patterns are consistent across the two synthetic datasets, while others are mixed. | Do not claim broad generalization from two synthetic datasets. | L1 wording boundary plus pending lower-level audits. |
| RQ5 | Compare GCN and GraphSAGE as backbone contexts. | CONDITIONAL. | BA-House, ShapeGGen, Elliptic. | Backbone architecture. | Same targets within dataset: report-level evidence. Backbone fairness: PENDING_AUDIT03. Target leakage/selection: PENDING_AUDIT04. | Predictive competence and learned representations differ. | Backbone-explainer cell summary. | EV-GP-003, EV-GP-006, EV-GP-008, EV-GP-009, EV-ART-001, EV-ART-003, EV-ART-004, EV-ART-007. | Differences may be reported as backbone-context observations. | Do not claim a pure causal effect of changing GCN to GraphSAGE. | L1 wording boundary; Audit03 dependency. |
| RQ6 | Apply explainers to Elliptic Bitcoin transaction classification. | CONDITIONAL. | Elliptic; GCN and GraphSAGE; GNNExplainer complete; PGExplainer Step03 pilot evidence only; Fid+, Fid-, 1-Fid-, sparsity, stability, runtime. | Explainer method after frozen model prediction. | Temporal split and no-GT explanation boundary: report-level/protocol-level ARTIFACT_VERIFIED. Task28 PG full training/final evaluation: PENDING_EXECUTION_DEPENDENCY. | No explanatory-edge ground truth; real-world observational setting. | Target transaction node. | EV-GP-007, EV-GP-008, EV-GP-009, EV-ART-006, EV-ART-007, EV-ART-008. | Elliptic explanations can be framed as model-importance signals for the predicted class. | Do not claim explanation correctness, fraud cause, legal/AML evidence, or final Elliptic GNNExplainer-vs-PGExplainer result. | Pending execution dependency, not a methodological severity escalation. |
| RQ7 | Use Random explanation as a sanity baseline. | CONDITIONAL - pending Audit05 candidate equivalence and Audit06 evaluator equivalence. | BA-House and ShapeGGen Random baseline. | Random edge scoring/selection versus learned explainers. | Same targets and budget: report-level evidence. Same candidate set, exact selected-edge budget, sampling without replacement, evaluator/reference class: PENDING_AUDIT05/AUDIT06. | Random is not learned and has different runtime semantics. | Raw random seed row, then node-level mean, then target-node summary. | EV-GP-003, EV-GP-004, EV-ART-001, EV-ART-004, EV-CODE-005, EV-CODE-006. | Random can serve as a chance/sanity baseline after candidate/evaluator equivalence is verified. | Do not treat Random as a trained explainer or as universally lower on every possible stability setting. | L2 dependency: deterministic Random/evaluator audit. |
| RQ8 | Report runtime and efficiency trade-offs. | CONDITIONAL. | Synthetic reports and Elliptic Step03 pilot. | Method execution mode: per-target optimization, upfront training, inference, random selection/evaluation. | Runtime naming: report-level evidence. Hardware/runtime comparability: PENDING_AUDIT09. | Timer boundaries and overhead differ by method. | Runtime stage, not always target-level equivalent. | EV-GP-003, EV-GP-006, EV-GP-008, EV-GP-009, EV-ART-001, EV-ART-003, EV-ART-007, EV-ART-008. | Runtime values are descriptive when stage semantics are named. | Do not collapse all methods into a single identical runtime stage. | L1 wording boundary; Audit09 dependency. |
| RQ9 | Claim that the thesis has ground-truth explanations. | CONDITIONAL. | BA-House and ShapeGGen only. | Dataset type. | Synthetic GT scope: PROTOCOL_ASSERTED plus report-level evidence. Local GT mapping correctness: PENDING_AUDIT05. | Synthetic GT is generator-based and may not reflect real-world causality. | Candidate edge. | EV-PAPER-005, EV-GP-003, EV-GP-004, EV-ART-001, EV-ART-003, EV-ART-006. | Synthetic branch has ground-truth explanation masks for benchmark evaluation. | Do not say Elliptic has explanatory-edge ground truth. | L1 wording boundary; Audit05 dependency. |
| RQ10 | Claim explanations reveal causes of illicit Bitcoin transactions. | UNSUPPORTED. | Elliptic. | None valid for causal inference. | No intervention, no explanatory-edge ground truth, no causal identification design. | Observational transaction graph and model-dependent explanations. | Transaction node or candidate edge. | EV-GP-007, EV-ART-006, EV-ART-007, EV-THESIS-001, EV-THESIS-003. | The thesis may analyze prediction-supporting subgraphs under the frozen model. | Do not claim fraud cause, legal evidence, AML proof, intention, or true causal transaction mechanism. | L1 wording guardrail. L5 only if the thesis insists on causal fraud claims, which current thesis drafts do not. |

## Design Choice Audit

| Design choice | Status | Chronology classification | Evidence ID for chronology | Governing rationale | Benefit | Trade-off | Valid interpretation | Forbidden interpretation |
|---|---|---|---|---|---|---|---|---|
| Continuous Edge AUROC/AP over the full candidate-edge set. | CONDITIONAL | REVIEW_AMENDMENT_FROZEN_BEFORE_SUBSEQUENT_RUN | EV-ART-001, EV-ART-003, EV-CODE-002 | Ranking metrics should use continuous candidate-edge scores rather than only the selected top-20 percent edges. | Preserves full score-distribution information. | Only valid when GT explanation edges exist. | Synthetic edge-ranking agreement after Audit05 confirms candidate mapping. | Elliptic explanation correctness. |
| Fixed selected-edge ratio of 0.2 with `ceil(candidate_edge_count * 0.2)`, minimum 1. | SUPPORTED at protocol/report level | REVIEW_AMENDMENT_FROZEN_BEFORE_SUBSEQUENT_RUN | EV-ART-001, EV-ART-003, EV-ART-006, EV-CODE-001 | Candidate-edge counts vary, so ratio budget avoids fixed top-k distortion. | Comparable selected-edge fraction across targets. | Still an evaluation budget, not a natural explanation size. | Frozen perturbation/sparsity budget. | Universally optimal explanation size. |
| Fid+, Fid-, and 1-Fid- binary prediction-change metrics. | CONDITIONAL | REVIEW_AMENDMENT_FROZEN_BEFORE_SUBSEQUENT_RUN | EV-ART-001, EV-ART-003, EV-CODE-003 | Separates removal-side and retention-side behavior. | Easier thesis interpretation than one generic probability delta. | Binary outcomes can hide probability magnitude. | Model-behavior metrics under frozen perturbation protocol. | Ground-truth correctness or causality. |
| Legacy `fidelity = remove_delta` compatibility field. | CONDITIONAL | POST_RESULT_DOCUMENTATION | EV-CODE-003, EV-ART-001 | Kept for backward compatibility with older Phase 4 schema. | Keeps older outputs readable. | Can be confused with Fid+. | Probability-delta diagnostic. | Main fidelity metric or GraphFramEx Fid+. |
| Common PGExplainer train/validation pools. | CONDITIONAL | REVIEW_AMENDMENT_FROZEN_BEFORE_SUBSEQUENT_RUN for ShapeGGen and Elliptic; POST_RESULT_DOCUMENTATION sensitivity for BA-House Task19B. | EV-GP-003, EV-GP-005, EV-GP-009, EV-ART-004, EV-ART-008 | Reduces explainer-training-pool membership confound. | Fairer PG comparison. | Does not remove architecture, representation, or predictive-competence confounds. | Better controlled PG train/validation setting. | Causal backbone effect. |
| No Edge AUROC/AP on Elliptic. | SUPPORTED | ORIGINALLY_PRE_SPECIFIED for Elliptic protocol before final Elliptic explanations. | EV-GP-007, EV-GP-008, EV-GP-009, EV-ART-006, EV-ART-007 | Elliptic has class labels but no explanatory-edge ground truth. | Avoids manufacturing invalid correctness labels. | Leaves Elliptic with behavior metrics and qualitative interpretation only. | No-GT model-importance evaluation. | Elliptic explanation accuracy. |
| Frozen target lists before explanation evaluation. | CONDITIONAL | ORIGINALLY_PRE_SPECIFIED at protocol level, artifact details pending later audits. | EV-GP-003, EV-GP-004, EV-GP-005, EV-GP-007, EV-ART-002, EV-ART-008 | Avoid target shopping and post-result selection. | Improves reproducibility and claim discipline. | Target set size limits statistical breadth. | Results apply to frozen target sets. | Population-wide proof without qualification. |
| Cross-dataset reporting without pooling target rows. | SUPPORTED at report level | REVIEW_AMENDMENT_FROZEN_BEFORE_SUBSEQUENT_RUN | EV-GP-006, EV-ART-004, EV-ART-005 | BA-House and ShapeGGen have different data-generating processes and target counts. | Avoids pseudo-replication and target-count domination. | Produces pattern comparison rather than one global score. | Cross-synthetic pattern consistency. | Direct pooled superiority or causal generalization. |
| Task28 PGExplainer full training must wait after Step03 evidence and review. | SUPPORTED | ORIGINALLY_PRE_SPECIFIED plus reviewed Step03 evidence. | EV-GP-009, EV-ART-008 | Full PGExplainer Elliptic training is expensive and needs parity, resume, timing, and leakage evidence first. | Reduces risk before long training. | Does not produce final PGExplainer Elliptic metrics. | Step03 evidence readiness. | Step04 authorization. |

## Term And Wording Audit

| Term | Status | Evidence | Required wording discipline |
|---|---|---|---|
| better | CONDITIONAL | EV-GP-002, EV-ART-001, EV-ART-003, EV-ART-004 | Use only with dataset, backbone, target set, and metric, for example “higher Edge AUROC on ShapeGGen with GCN fixed”. |
| robust | CONDITIONAL | EV-GP-002, EV-ART-001, EV-ART-004, EV-CODE-004 | Prefer “stable under repeated seeds by Jaccard”. Do not imply adversarial or distributional robustness. |
| stable | CONDITIONAL | EV-ART-001, EV-ART-004, EV-ART-007, EV-CODE-004 | Valid as Jaccard stability only after Audit07/Audit08 confirm repetition and aggregation details. |
| faithful | CONDITIONAL | EV-THESIS-002, EV-REPO-001 | Replace with or define narrowly as fidelity, Fid+, Fid-, 1-Fid-, or perturbation-based model-behavior metric. Wording-only debt is L1. |
| fidelity | CONDITIONAL | EV-CODE-003, EV-ART-001, EV-ART-003 | Split into Fid+, Fid-, 1-Fid-, and probability diagnostics. Do not collapse all into one score. |
| generalizable | CONDITIONAL | EV-GP-006, EV-ART-004 | Use “pattern consistency across two synthetic datasets”; avoid broad generalization. |
| causal | UNSUPPORTED | EV-GP-003, EV-GP-006, EV-GP-007, EV-ART-006, EV-ART-007 | Avoid for backbone effects, explainer effects, and Elliptic transaction behavior. |
| ground truth | CONDITIONAL | EV-PAPER-005, EV-ART-006, EV-THESIS-003 | Valid for synthetic explanation masks only; invalid for Elliptic explanatory edges. |
| fraud cause | UNSUPPORTED | EV-GP-007, EV-ART-006, EV-ART-007 | Replace with “model-importance signal for the predicted class”, “prediction-supporting subgraph under the frozen model”, or “subgraph associated with the model prediction”. |

## Proposed Claim Wording Corrections

| Risky wording | Safer thesis wording |
|---|---|
| “GNNExplainer is better than PGExplainer.” | “On the specified dataset, frozen backbone, target set, and metric, one explainer achieved a higher recorded score than the other under the frozen protocol.” |
| “PGExplainer is more robust.” | “PGExplainer showed higher recorded Jaccard stability under repeated seeds/checkpoints in the frozen reports. Full stability comparability remains pending Audit07/Audit08.” |
| “The explanation is faithful.” | “The selected edges affected the frozen model prediction according to Fid+, Fid-, 1-Fid-, and probability diagnostics under the perturbation protocol.” |
| “GraphSAGE causes better explanations.” | “Explanation metrics differ between GCN and GraphSAGE backbone contexts. This is descriptive and not a causal estimate.” |
| “The Elliptic explanation identifies fraud causes.” | “The Elliptic explanation is a model-importance signal for the predicted class under the frozen model. It is not legal, AML, ground-truth, or causal evidence.” |
| “Results generalize.” | “Some metric-ordering patterns are consistent across BA-House and ShapeGGen, while others are mixed.” |

## Audit Findings

| Finding | Severity | Evidence | Why it matters | Remediation |
|---|---|---|---|---|
| Current thesis chapters separate synthetic ground-truth evaluation from Elliptic model-importance analysis. | L1 positive | EV-THESIS-001, EV-THESIS-003, EV-ART-006 | This protects the thesis from invalid Elliptic AUROC/AP and causal claims. | Preserve this boundary in results and conclusion chapters. |
| Current thesis chapters include “faithfulness” in conceptual discussion. | L1 documentation wording debt | EV-THESIS-002 | The project now standardizes on fidelity, Fid+, Fid-, and 1-Fid-. | Replace or define narrowly in a thesis-writing pass. |
| Repository `README.md` is outdated and still includes scaffold/planned language and “faithfulness”. | L1 documentation debt | EV-REPO-001 | External readers may infer an outdated project state. | Update README in a later documentation task, not this amendment. |
| Elliptic PGExplainer final evaluation is not available yet. | PENDING EXECUTION DEPENDENCY | EV-GP-009, EV-ART-008 | A final Elliptic 2x2 comparison would be premature. | Keep Task28 Step04/full training blocked until explicitly authorized. |
| Cross-backbone claims remain non-causal. | L1 wording guardrail | EV-GP-003, EV-GP-006, EV-ART-001, EV-ART-004, EV-ART-007 | Predictive competence, representations, and architecture differ together. | Use “backbone-context observation”, not “backbone effect”. |
| Candidate-edge and evaluator equivalence are not fully proven by Audit01. | L2 pending deterministic audit | EV-CODE-002, EV-CODE-003 | Protocol assertions cannot be upgraded into artifact-level proof without later audits. | Leave RQ1/RQ2/RQ7 conditional until Audit05/Audit06 pass. |
| Stability aggregation must remain nested by target node. | L2 pending deterministic/statistical audit | EV-CODE-004, EV-ART-001, EV-ART-007 | Treating seed-pair rows as independent target observations would inflate evidence. | Audit07/Audit08 must verify pair counts, nesting, and aggregation. |

## External Citation Metadata Audit

| Evidence ID | Authors | Full title | Year and venue | DOI | URL | Access date | Supports | Does not prove |
|---|---|---|---|---|---|---|---|---|
| EV-PAPER-001 | Rex Ying, Dylan Bourgeois, Jiaxuan You, Marinka Zitnik, Jure Leskovec | GNNExplainer: Generating Explanations for Graph Neural Networks | 2019, NeurIPS 2019 | `10.48550/arXiv.1903.03894` | https://arxiv.org/abs/1903.03894 | 2026-09-03 | GNNExplainer method definition and optimization-based local explanation framing. | Local implementation correctness, candidate alignment, or metric validity. |
| EV-PAPER-002 | Dongsheng Luo, Wei Cheng, Dongkuan Xu, Wenchao Yu, Bo Zong, Haifeng Chen, Xiang Zhang | Parameterized Explainer for Graph Neural Network | 2020, NeurIPS 2020 | `10.48550/arXiv.2011.04573` | https://arxiv.org/abs/2011.04573 | 2026-09-03 | PGExplainer as parameterized multi-instance explanation generator. | Local PGExplainer training fairness, checkpoints, or evaluator equivalence. |
| EV-PAPER-003 | Thomas N. Kipf, Max Welling | Semi-Supervised Classification with Graph Convolutional Networks | 2017, ICLR 2017 | `10.48550/arXiv.1609.02907` | https://arxiv.org/abs/1609.02907 | 2026-09-03 | GCN backbone concept. | Local GCN checkpoint quality or causal backbone comparison. |
| EV-PAPER-004 | William L. Hamilton, Rex Ying, Jure Leskovec | Inductive Representation Learning on Large Graphs | 2017, NeurIPS/NIPS 2017 | `10.48550/arXiv.1706.02216` | https://arxiv.org/abs/1706.02216 | 2026-09-03 | GraphSAGE backbone concept and neighborhood aggregation. | Local GraphSAGE checkpoint quality or causal backbone comparison. |
| EV-PAPER-005 | Chirag Agarwal, Owen Queen, Himabindu Lakkaraju, Marinka Zitnik | Evaluating Explainability for Graph Neural Networks | 2022 arXiv submission, 2023 revision | `10.48550/arXiv.2208.09339` | https://arxiv.org/abs/2208.09339 | 2026-09-03 | ShapeGGen/GraphXAI motivation for synthetic datasets with ground-truth explanations. | Local ShapeGGen artifact correctness or repository metric implementation. |

## Stop-Condition Review

| Stop condition | Result |
|---|---|
| A claim cannot be traced to evidence. | Not triggered after amendment. Major claims now cite evidence IDs or pending audit dependencies. |
| External source metadata cannot be verified. | Not triggered. Primary arXiv metadata was verified; no unverified DOI was invented. |
| A claimed control is contradicted by artifacts. | Not discovered in Audit01. Some controls remain PENDING_AUDIT rather than ARTIFACT_VERIFIED. |
| A primary thesis claim becomes unsupported. | Not triggered if primary claims remain scoped to synthetic within-backbone reports and Elliptic model-importance analysis. |
| Candidate-edge/evaluator equivalence failure is discovered. | Not discovered. Equivalence is deferred to Audit05/Audit06. |
| Result values would need to change. | Not triggered. No metric values were changed. |
| Experiment rerun appears necessary. | Not triggered. No rerun required. |

## Audit01 Review Amendments

Original Audit01 verdict: `PASS WITH CLAIM BOUNDARIES`.

External review verdict: `AUDIT01: CONDITIONAL PASS - DOCUMENTATION AMENDMENT ONLY`.

Amendments made in this version:

1. RQ1 changed from `SUPPORTED` to `CONDITIONAL - pending Audit05 candidate-edge equivalence and Audit06 perturbation/metric semantics`.
2. RQ2 changed from `SUPPORTED` to `CONDITIONAL - pending Audit05/Audit06`.
3. RQ3 changed from `SUPPORTED` to `CONDITIONAL - pending Audit07 seed/stability semantics and Audit08 statistical aggregation`.
4. RQ7 changed from `SUPPORTED` to `CONDITIONAL - pending Audit05 candidate equivalence and Audit06 evaluator equivalence`.
5. Added `Evidence Index` with `EV-GP`, `EV-ART`, `EV-CODE`, `EV-REPO`, `EV-THESIS`, and `EV-PAPER` IDs.
6. Added control-verification states: `PROTOCOL_ASSERTED`, `ARTIFACT_VERIFIED`, `PENDING_AUDIT`, `NOT_APPLICABLE`, and `UNKNOWN`.
7. Added external-source metadata, source roles, and source limitations.
8. Replaced vague chronology wording with `ORIGINALLY_PRE_SPECIFIED`, `REVIEW_AMENDMENT_FROZEN_BEFORE_SUBSEQUENT_RUN`, `POST_RESULT_DOCUMENTATION`, and `UNKNOWN_CHRONOLOGY`.
9. Corrected severity semantics: `faithfulness` wording and outdated README are L1 documentation debt; unfinished Elliptic PGExplainer final evaluation is a pending execution dependency, not automatically L3.
10. Tightened Elliptic wording to “model-importance signal for the predicted class” and “prediction-supporting subgraph under the frozen model”.

No experiment rerun was performed. No result, metric, checkpoint, dataset, target list, or config value was changed.

Pending dependencies retained:

| Dependency | Purpose |
|---|---|
| Audit03 | Backbone fairness and predictive-quality comparability. |
| Audit04 | Target selection and leakage audit. |
| Audit05 | Candidate-edge mapping, directedness, and candidate equivalence. |
| Audit06 | Perturbation, evaluator, metric identity, and reference-class semantics. |
| Audit07 | Explainer training, seeds, and stability stochastic-source semantics. |
| Audit08 | Statistical aggregation, nesting, and summary unit verification. |
| Audit09 | Runtime and compute semantics. |

## Final Verdict

AUDIT01 VERDICT: PASS WITH EXPLICIT PENDING-AUDIT DEPENDENCIES.

Audit02 authorization: YES, after user review.

Task28 Step04: NOT AUTHORIZED.

Full Task28 training: NOT AUTHORIZED.

Task29: NOT AUTHORIZED.

STOP FOR REVIEW.
