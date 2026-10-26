"""Finalize audited TASK29 tie-v1 scientific reports from persisted outputs only."""

import csv
import io
import json
import math
import statistics
import subprocess
import time
import unittest

import numpy as np

import evaluate_task29_tie_v1 as ev
from verify_task29_tie_v1_outputs import verify as independent_output_readback

OUT, ROOT = ev.OUT, ev.ROOT
sha, write = ev.frozen.sha, ev.write


def md(name, text):
    (OUT / name).write_text(text.strip() + "\n", encoding="utf-8")


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def main():
    evidence = load("TASK29_FINAL_EVALUATION_INTEGRITY.json")
    assert evidence["verdict"] == ev.VERDICT
    assert sha(ROOT / "scripts/evaluate_task29_tie_v1.py") == evidence["evaluation_source_sha256"]
    test_stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_task29*.py")
    result = unittest.TextTestRunner(stream=test_stream, verbosity=2).run(suite)
    assert result.wasSuccessful() and result.testsRun == 30 and not result.skipped
    write(
        "TEST_RESULTS.json",
        {
            "tests": result.testsRun,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "skipped": len(result.skipped),
            "transcript": test_stream.getvalue(),
            "scope": "Synthetic evaluation/archive/cache/lexicographic-ranking/aggregation unit tests; no model training",
        },
    )
    results = load("TASK29_FINAL_STABILITY_RESULTS.json")
    sensitivity = load("TASK29_TIE_SENSITIVITY_ANALYSIS.json")
    audit = load("TASK29_RAW_LOGIT_RANKING_AUDIT.json")
    summaries = {r["backbone"]: r for r in results["results"]}
    s = sensitivity["backbones"]
    diagnostics = audit["records"]
    observed = []
    parity_count = 0
    for b in summaries:
        for node in evidence["target_ids"]:
            path = OUT / "records" / b / f"{node}.json"
            record = json.loads(path.read_text())
            receipt = json.loads(path.with_suffix(".receipt.json").read_text())
            assert sha(path) == receipt["sha256"]
            assert record["evaluation_policy_identity"] == ev.POLICY_ID
            old = ev.OLD / "records" / b / f"{node}.json"
            if old.exists():
                previous = json.loads(old.read_text())
                assert (
                    record["candidate_global_edge_indices"]
                    == previous["candidate_global_edge_indices"]
                )
                assert all(
                    a["raw_logits"] == c["raw_logits"]
                    for a, c in zip(record["runs"], previous["runs"], strict=True)
                )
                parity_count += 5
            for row in record["runs"]:
                primary = set(row["selected_global_edge_indices"])
                alternative = set(row["alternative_selected_global_edge_indices"])
                if primary != alternative:
                    other = [r for r in record["runs"] if r["seed"] != row["seed"]]
                    observed.append(
                        {
                            "backbone": b,
                            "node_id": node,
                            "seed": row["seed"],
                            "only_primary": sorted(primary - alternative),
                            "only_alternative": sorted(alternative - primary),
                            "boundary_raw": row["diagnostics"]["boundary_value"],
                            "pair_intersection_comparison": [
                                {
                                    "other_seed": r["seed"],
                                    "primary_intersection": len(
                                        primary & set(r["selected_global_edge_indices"])
                                    ),
                                    "alternative_intersection": len(
                                        alternative
                                        & set(r["alternative_selected_global_edge_indices"])
                                    ),
                                }
                                for r in other
                            ],
                        }
                    )
    assert parity_count == 90
    sensitivity["changed_selection_details"] = observed
    sensitivity["materiality_assessment"] = {
        "headline_or_pairwise_changes_observed": False,
        "reason": "Every one of4000 ascending/descending-ID pairwise Jaccards is identical in this evaluation",
        "not_a_general_tie_invariance_theorem": True,
        "selection_identities_still_changed": len(observed),
    }
    assert all(
        float(r["jaccard"]) == float(r["alternative_jaccard"])
        for r in csv.DictReader((OUT / "TASK29_FINAL_STABILITY_PAIRWISE.csv").open())
    )
    write("TASK29_TIE_SENSITIVITY_ANALYSIS.json", sensitivity)
    audit["backbone_summary"] = {}
    for b in summaries:
        rows = [r for r in diagnostics if r["backbone"] == b]
        audit["backbone_summary"][b] = {
            "denominator": 1000,
            "exact_duplicate_vectors": s[b]["exact_duplicate_vectors"],
            "boundary_crossing_vectors": s[b]["boundary_crossing_vectors"],
            "sigmoid_constant_vectors": s[b]["sigmoid_constant_vectors"],
            "all_identical_raw_vectors": s[b]["all_identical_raw_vectors"],
            "near_boundary_fragile_counts": {
                t: sum(r["near_tie_sensitivity"][t]["boundary_gap_le_2tol"] for r in rows)
                for t in ("1e-07", "1e-06", "1e-05")
            },
            "raw_population_sd_median": float(np.median([r["raw_std_population"] for r in rows])),
            "raw_unique_fraction_mean": statistics.mean(
                r["unique_raw_float32"] / r["candidate_edge_count"] for r in rows
            ),
        }
    write("TASK29_RAW_LOGIT_RANKING_AUDIT.json", audit)
    # Verify the historical GNN aggregation/coverage from archived per-node and seed CSVs.
    historical = json.loads(
        (
            ROOT / "docs/audit/elliptic_comprehensive_results/elliptic_results_summary.json"
        ).read_text()
    )
    comparison = {
        "verdict": "DESCRIPTIVE_PROTOCOL_LIMITED_NOT_CONTROLLED_SUPERIORITY",
        "matching_checks": [
            "same frozen200 targets/order",
            "same2-hop candidate count and k",
            "same seeds0..4",
            "same target-level mean10 pairs and sample SD unit",
            "same frozen backbone hashes in configs",
        ],
        "candidate_exact_id_matching_evidence": "Existing comprehensive audit02 verifies2400 GNN records against historical PG canonical IDs; current PG full200 matches those same IDs",
        "new_raw_GNN_mask_rerun_or_reaudit": False,
        "differences": [
            "GNN5 epochs per target vs independently trained PG shared MLP",
            "GNN common feature mask plus edge mask vs PG edge logits",
            "PG RAW inference and post-observation tie addendum; do not claim identical training/scoring protocol",
        ],
        "backbones": {},
    }
    for b, key in [("gcn", "GCN_GNN"), ("graphsage", "GraphSAGE_GNN")]:
        path = ROOT / f"results/metrics/elliptic/{b}/gnnexplainer/stability_seed_metrics.csv"
        rows = list(csv.DictReader(path.open()))
        assert len(rows) == 1000
        for node in evidence["target_ids"]:
            selected = [r for r in rows if int(r["node_id"]) == node]
            assert sorted(int(r["seed"]) for r in selected) == list(range(5))
            pg = json.loads((OUT / "records" / b / f"{node}.json").read_text())
            n = len(pg["candidate_global_edge_indices"])
            assert all(
                int(r["candidate_edge_count"]) == n
                and int(r["selected_edge_count"]) == max(1, math.ceil(0.2 * n))
                for r in selected
            )
        comparison["backbones"][b] = {
            "GNN_historical": historical["final_metrics"][key]["stability"],
            "PG_current": summaries[b],
            "seed_csv_sha256": sha(path),
        }
    write("TASK29_HISTORICAL_GNN_COMPARISON.json", comparison)
    g, gs = summaries["gcn"], summaries["graphsage"]
    delta = gs["mean_jaccard"] - g["mean_jaccard"]
    table = f"""| Backbone | Seeds | Targets | Mean Jaccard | Sample SD (ddof1) |
|---|---|---:|---:|---:|
| GCN (diagnostic) | 0–4 | 200 | {g["mean_jaccard"]:.10f} | {g["sample_sd"]:.10f} |
| GraphSAGE | 0–4 | 200 | {gs["mean_jaccard"]:.10f} | {gs["sample_sd"]:.10f} |"""
    sens_table = "| Backbone | Exact duplicate vectors | Boundary-crossing | Changed selections | Affected target Jaccards | ASC mean | DESC mean | Absolute Δ |\n|---|---:|---:|---:|---:|---:|---:|---:|\n"
    for b, label in [("gcn", "GCN"), ("graphsage", "GraphSAGE")]:
        v = s[b]
        sens_table += f"| {label} | {v['exact_duplicate_vectors']}/1000 ({v['exact_duplicate_vectors_percent']}%) | {v['boundary_crossing_vectors']}/1000 | {v['changed_seed_target_selections']}/1000 | {v['targets_with_affected_pairwise_jaccard']}/200 | {v['official_mean']:.10f} | {v['alternative_mean']:.10f} | {v['absolute_difference']:.10f} |\n"
    md(
        "TASK29_TIE_SENSITIVITY_ANALYSIS.md",
        f"""
# TASK29 Tie Sensitivity Analysis

Ngày2026-10-15. Evaluation policy identity: `{ev.POLICY_ID}`. Official rule là RAW float32 giảm dần, exact-equal global directed ID tăng dần. ID giảm dần chỉ là diagnostic, không chọn theo Jaccard.

{sens_table}

Hai selections thay đổi thuộc GCN seed3/node159576 và GCN seed0/node174138. Chúng tương ứng2/200 targets có thay đổi ít nhất một seed selection; GraphSAGE0/200. Tất cả4000 pairwise Jaccards,400 target means và hai headline means không đổi giữa hai chính sách. Chi tiết các ID đổi và intersection counts với từng seed còn lại nằm trong JSON. Đây là kiểm tra trên dữ liệu đã đo, không phải định lý mọi tie resolution đều không ảnh hưởng Jaccard.

Near-tie thresholds1e-7/1e-6/1e-5 giữ riêng: số vectors có boundary gap≤2tol là2/1000 cho GCN và0/1000 cho GraphSAGE ở cả ba thresholds, trùng các exact-crossing cases trong lượt này. Không dùng tolerance để gộp score hoặc thay official selection.

Rule tăng dần được user approve sau quan sát blocker và được áp dụng cho toàn2000 vectors. Không đổi RAW scores; không claim ID nhỏ quan trọng hơn ID lớn. Không cần chọn alternative dù hai headline bằng nhau. Post-observation addendum và original BLOCKED evidence đều được giữ nguyên.
""",
    )
    md(
        "TASK29_RAW_LOGIT_RANKING_AUDIT.md",
        f"""
# TASK29 RAW-Logit Ranking Audit: Tie V1

Full coverage2000 vectors,1000/backbone,200 targets×5 seeds. Policy `{ev.POLICY_ID}`. Không sigmoid trước ranking; no score changes.

{sens_table}

Sigmoid float32 vectors hằng số: GCN1000/1000; GraphSAGE973/1000. RAW vectors hoàn toàn bằng nhau:0/1000 ở cả hai. Do đó saturation của sigmoid không đồng nghĩa RAW-logit collapse. RAW có variation nhưng exact duplicates vẫn xuất hiện:61 GCN vectors và7 SAGE vectors; chỉ2 GCN vectors có tie cắt top-k boundary. Duplicate-vector count là số vectors có ít nhất một duplicate RAW score; khác với duplicate-value excess và fraction bên trong mỗi vector.

Các records giữ finite count, unique float32, duplicate-value excess/fraction, RAW min/max/float64 descriptive mean/population SD, boundary value/multiplicity/gap/crossing, sigmoid unique count/std và near-tie diagnostics. Float64 chỉ dùng descriptive reduction sau inference, không tính lại model logits. Tất cả scores để ranking vẫn RAW float32 nguyên vẹn.

Full-edge MLP có468710 entries, candidate universe canonical giữ thứ tự. Near thresholds absolute logit units1e-7,1e-6,1e-5 chỉ diagnostic; không epsilon, rounding hay tolerance-based tie merging. Smoke lặp40 target-seed vectors (cả10 checkpoints×4 targets, gồm known blocker) có RAW và ranking exact.90 vectors cũ được đối chiếu với fresh inference và trùng RAW từng giá trị; old outputs không được dùng làm cache mới.

Stability đo overlap selections giữa seeds. Nó không đo Fidelity hay causal importance. Sigmoid saturation vẫn là một interpretive limitation quan trọng: đây là kết quả RAW-ranked PG case study, không trình bày như masks sigmoid có độ phân giải tốt. GCN giữ vai trò diagnostic.
""",
    )
    md(
        "TASK29_FINAL_EVALUATION_INTEGRITY.md",
        f"""
# TASK29 Final Evaluation Integrity: Tie V1

**{ev.VERDICT}**. Ngày2026-10-15. Policy `{ev.POLICY_ID}`; original full protocol `{ev.frozen.PROTOCOL_ID}` giữ nguyên. Addendum post-observation; original strict-boundary STOP và old BLOCKED folder bất biến.

## Inputs Và Checkpoints

Master A SHA256 `6d8ac8dba1f15b31ba5e883650fe7197c0af3830b4b501e795a24084e859b1be`; master B `53c99474b44dcc7da347ee3ff59454f5af7ac9c71488aa5af6dc1c6f2e591e3c`. Cả hai readable safe ZIP, CRC, exact member coverage và all manifest hashes PASS, nested seed ZIP/checkpoint hashes khớp receipts.10/10 unique backbone/seed pairs,30 epochs và bốn Adam step=60000, finite parameters, exact MLP/Adam schema/config/HARD/frozen inputs được reverify. Không dùng seed42; không optimizer construction/step/backward/training. Original final weights và loaded weights không đổi.

Current_master.json/execution_summary.json external chưa được cung cấp: archive integrity và compatible COMPLETE session được verify, nhưng không tuyên bố platform-selected latest master đã được attest. Hai master local được nhận dạng bằng SHA cố định, không tự suy ra latest từ basename.

## Source, Runtime Và Protocol

Frozen code được import từ source bytes trong Release002. Inputs/backbones hash-match native related repo. Original execution identity7d4706ff1342ae2a7445cd92a7ead06e6200480885fa4a4e6699c92ba2d3e11c, scientific origin6034c13bfab601bef823c07194c70574cd49ec41. Target SHA e414941e6943b06484ecd5028c51daf80ff2bf9f1bbb06e6a5fd816de8dfb9c0; all200 ordered IDs và canonical2-hop candidate IDs khớp historical artifact. Graph203769 nodes/468710 directed columns; full-edge[h_src,h_dst,h_target] float32 MLP arithmetic không đổi.

Windows RTX4060 Laptop; Python3.11.9, torch2.11.0+cu128, PyG2.8.0.post1/CUDA12.8. Không gọi identical training environment vì Linux/T4 khác OS/device/driver. Deterministic algorithms, CUBLAS_WORKSPACE_CONFIG=:4096:8, matmul TF32false, highest float32, threads/interop1; eval, gradients disabled và inference_mode. Smoke gồm known blocker; all10 checkpoints PASS,40 repeated vectors exact; GPU peak{evidence["gpu_peak_allocated_bytes"] / 2**30:.6f}GiB. Inference incl.smoke{evidence["inference_seconds_including_smoke"]:.6f}s; execution incl.input verification/aggregation{evidence["total_evaluation_wall_seconds"]:.6f}s, không gồm viết báo cáo.

## Coverage, Replay Và Independent Verification

400 target records,5 seeds/record,2000 RAW/selected vectors,4000 pairs,400 target means. Per-target atomic JSON+receipt bind source/runtime/checkpoints/candidates và policy identity. Toàn bộ inference được tính mới; không admit selected-edge sets cũ. Fresh RAW khớp90 partial vectors cũ qua read-only comparison. Strict-separated boundaries luôn cùng selected set với original CPU topk; exact boundary ties dùng ASC ID uniformly.

Independent verifier đọc saved global IDs, dùng numpy boolean membership overlaps và explicit fsum/two-pass ddof1 formula, không reuse primary set-intersection/mean implementation. All4000 numerator/denominator/pair values và400 means, cả hai headline means/SD PASS tại absolute tolerance1e-14; max pair difference0. Báo cáo JSON giữ toàn bộ recomputed metrics.

30 synthetic evaluation unit tests PASS, gồm15 tests cho approved tie policy. Source hash, test transcript và lint/check outputs được giữ trong output directory. Các CSV/JSON được parse/read-back, finite metrics, complete matrix/cardinality/policy consistency và hash inventory verify. {evidence["all_input_old_evidence_files_after_verified"]} protected paths gồm old outputs, Release002/runtime và frozen inputs được hash-check trước/sau, không đổi. Không sửa old reports, preflight STOP hay thesis chapters.

Không tune checkpoint/hyperparameters/temperature/budget dựa final200; user phê duyệt addendum sau quan sát tie, được disclose rõ. Final200 đã được xem trong TASK28 và blocked run, nên là replication chứ không untouched holdout. Sensitivity không dùng để chọn policy thuận lợi. Chỉ evaluation scope hoàn tất, không thêm training hoặc remote upload.
""",
    )
    md(
        "TASK29_THESIS_READY_STABILITY_REPORT.md",
        f"""
# Cross-Seed Stability Của PGExplainer Trên Elliptic

## Mục Tiêu Và Thiết Kế

Thực nghiệm TASK29 đánh giá mức nhất quán của tập cạnh do PGExplainer lựa chọn khi thay đổi seed huấn luyện explainer, trong điều kiện dữ liệu và backbone cố định. Hai backbone là GCN và GraphSAGE. Với mỗi backbone, năm checkpoint được huấn luyện độc lập theo các seed 0, 1, 2, 3 và 4. Mỗi checkpoint hoàn thành 30 epochs trên cùng 2.000 training targets, tương ứng 60.000 optimizer updates. Tổng cộng 10 checkpoints hoàn tất 600.000 updates. Seed 42 lịch sử không thay thế cho bất kỳ seed mới nào.

Các checkpoints, frozen backbone models, processed dataset và target pools được kiểm tra bằng SHA256 và receipt. Session A cung cấp các seed 0, 1 và 2 cho hai backbone; session B cung cấp các seed 3 và 4. Đánh giá sử dụng cùng 200 final targets theo thứ tự đóng băng. GCN tiếp tục là diagnostic case study; kết quả năm seed mới của GraphSAGE không tự tạo thành bằng chứng mọi checkpoint đều validation-supported.

## Inference Protocol Và Addendum

Explanation scores là RAW float32 edge logits từ MLP đã huấn luyện. Inputs của MLP là [h_source || h_destination || h_target], dùng embeddings cuối message passing của frozen backbone. Logits được tính trên toàn bộ 468.710 directed edge entries, rồi lấy canonical 2-hop candidate edges cho từng target. Không áp dụng sigmoid trước ranking, không tính lại model logits bằng float64, normalization, clipping, epsilon hay thay đổi temperature. Budget là k=max(1,ceil(0.20×candidate_count)). Một edge identity là global directed edge-index column, không phải unordered endpoint pair.

Lượt đánh giá trước dừng đúng protocol tại GCN seed 3, target 159576 do exact score tie cắt top-k boundary. Sau khi đã quan sát blocker, một **evaluation-only post-observation addendum** được phê duyệt: score giảm dần; khi RAW float32 bằng nhau chính xác, global directed edge ID nhỏ hơn được ưu tiên. Quy tắc này được áp dụng đồng nhất cho 2.000 target-seed vectors, không chỉ target gây blocker. Đây là quy ước tái lập, không phải bằng chứng cạnh có ID nhỏ quan trọng hơn.

Original scientific protocol/execution identifiers và STOP policy giữ nguyên làm provenance. Evaluation policy identity mới là `{ev.POLICY_ID}`. Thư mục blocked cũ không bị sửa; amended evaluation nằm trong version mới. Không trình bày addendum như đã được preregister trước khi nhìn thấy final200. Không có training bổ sung, checkpoint selection hay optimization against Jaccard.

## Định Nghĩa Và Tổng Hợp Jaccard

Với tập cạnh A và B của hai seed trên cùng target, J(A,B)=|A∩B|/|A∪B|. Năm seeds tạo 10 cặp không thứ tự. Trước hết lấy trung bình 10 Jaccard trong mỗi target, sau đó lấy trung bình đều trên 200 target means. Sample SD dùng ddof=1 trên chính 200 target means. SD không phải biến thiên giữa backbone models, không phải SE, CI hay significance test; các seed pairs dùng chung seeds và không được xem như 2.000 observations độc lập.

## Kết Quả Chính Thức

{table}

GraphSAGE có mean cao hơn GCN **{delta:.10f}** theo cùng aggregation trong hai frozen model contexts này. Đây là khác biệt mô tả, không phải kiểm định ý nghĩa thống kê hoặc kết luận nhân quả rằng architecture GraphSAGE luôn tạo explanation ổn định hơn. Dataset, input features và target set được giữ chung, nhưng embedding dimensions, frozen model states và lịch sử huấn luyện của hai backbone khác nhau.

GCN có median{g["median"]:.6f}, Q1={g["q1"]:.6f}, Q3={g["q3"]:.6f}, range[{g["min"]:.6f}, {g["max"]:.6f}]. GraphSAGE có median{gs["median"]:.6f}, Q1={gs["q1"]:.6f}, Q3={gs["q3"]:.6f}, range[{gs["min"]:.6f}, {gs["max"]:.6f}]. Quartiles dùng linear interpolation. Những thống kê này cho thấy overlap khác nhau giữa targets; mean cao không có nghĩa mọi target đều giống nhau giữa seeds.

## Tie Sensitivity

{sens_table}

Hai boundary-crossing vectors của GCN tương ứng hai target-seed selections. Khi đổi secondary key thành ID giảm dần, các ID được chọn thay đổi nhưng intersection/union với các seed khác giữ nguyên trong dữ liệu thực đo; tất cả4000 pairwise Jaccards không đổi. Vì vậy hai headline means và kết luận chênh lệch giữa backbone không nhạy với hai policy được so sánh ở lượt này. Không suy rộng thành mọi tie policy hoặc mọi dataset đều bất biến. ASC ID vẫn là official rule đã approve, không được chọn vì đạt mean bằng hay cao hơn alternative.

## Sigmoid Saturation Và RAW Ranking

Ở GCN, 1.000/1.000 sigmoid vectors có đúng một giá trị float32; ở GraphSAGE, số tương ứng là 973/1.000. Tuy nhiên, không RAW vector nào có toàn bộ logits bằng nhau. Điều này phân biệt sự mất độ phân giải sau sigmoid với RAW-logit collapse. RAW vectors vẫn biến thiên, trong khi exact duplicate scores xuất hiện trong 61/1.000 GCN vectors và 7/1.000 GraphSAGE vectors; chỉ 2 GCN vectors có tie cắt selection boundary.

Các near-tie diagnostics tại absolute logit tolerances1e-7,1e-6,1e-5 không thay official ranking. Tại các thresholds đã khai báo, GCN có2 vectors với boundary gap≤2tol và GraphSAGE không có; trong lượt này chúng trùng exact-crossing cases. Không xem diagnostic tolerances như uncertainty calibrated hoặc tune chúng để cải thiện Jaccard. Descriptive mean/SD của RAW logits được tính trên float64 copy sau inference, không thay arithmetic tạo logits.

## Stability Không Đồng Nghĩa Fidelity

Stability phản ánh tính nhất quán của selections giữa các seed explainer. Nó không chứng minh selections bảo toàn quyết định của model, không đo fidelity bằng hard perturbation và không cung cấp causal importance của giao dịch. Lượt này không tái tính Fidelity. Vì vậy, mean Jaccard0.7673/0.7910 không được diễn giải thành explanation accuracy hoặc tỷ lệ cạnh gian lận được phát hiện đúng.

Tương tự, targets có overlap thấp không chứng minh frozen classifier bất ổn. Thiết kế chỉ thay explainer training seed với một classifier cố định cho mỗi backbone, không thay classifier seed hay thực hiện controlled graph perturbation. Elliptic không có edge-level explanation ground truth, nên không tự tạo Edge AUROC/AP từ node labels.

## Đối Chiếu GNNExplainer Có Điều Kiện

Evidence lịch sử đã xác minh GNNExplainer mean Jaccard GCN0.5011678974282419 và GraphSAGE0.5747609353498806. Audit hiện tại kiểm lại1000 seed rows/backbone về cùng200 target IDs, seeds0..4, candidate counts/k; comprehensive audit trước đã kiểm exact candidate IDs. Cùng frozen backbone hashes và target-level mean10-pair aggregation được mô tả trong configs/audit nguồn.

Có thể đối chiếu **mô tả, có giới hạn protocol** rằng mean overlap PG RAW-amended trong case study này cao hơn các historical GNN means. Không gọi đó là controlled superiority hoặc chất lượng explanation tốt hơn: GNNExplainer tối ưu mask riêng từng target,5 epochs, có common feature mask; PG dùng shared edge-logit MLP được huấn luyện upfront. Scoring và tie disclosure của PG khác. Không rerun GNN raw-mask experiment hoặc tạo significance test trong lượt này. Supplementary JSON ghi matching factors, source hashes và khác biệt, không âm thầm coi mọi điều kiện tối ưu hóa giống nhau.

## Tái Lập, Giới Hạn Và Kết Luận

Inference chạy local Windows/RTX4060 với Python3.11.9, torch2.11.0+cu128 và PyG2.8.0.post1. Training chạy Linux/T4; versions khớp không đồng nghĩa execution environment giống hệt. Deterministic inference smoke trên cả10 checkpoints và4 targets, gồm blocker cũ, có RAW/ranking exact khi lặp trong cùng runtime. Toàn2000 vectors được tính mới; không reuse old selected sets. Canonical candidates toàn200 targets khớp evidence lịch sử.

Independent recomputation từ saved global edge IDs khớp4000 pairwise values,400 target means và cả hai means/SD với tolerance1e-14. Toàn bộ10 checkpoints và old evidence không đổi. Việc tái lập không cần một chu kỳ training10 models mới. Tuy vậy, repeated local parity không chứng minh bitwise equality trên mọi GPU/backend. Final200 đã được xem ở TASK28, blocked run và addendum approval, nên đây là replication có disclosure chứ không untouched holdout.

TASK29 hoàn tất phép đo cross-seed Stability với verdict **{ev.VERDICT}**. Có thể đóng task ở phạm vi metric/evaluation đã nêu, đồng thời giữ giới hạn sigmoid saturation, diagnostic scope của GCN và post-observation addendum. Hướng nghiên cứu tiếp theo gồm kiểm soát classifier-seed variation, robustness của rankings theo numerical backend, và đánh giá Fidelity riêng. Các hướng này không được xem là kết quả đã có của TASK29.

## Đoạn Tổng Kết Cho Chương 4–5

“Trên 200 targets Elliptic cố định, PGExplainer với năm seed huấn luyện độc lập cho mỗi backbone đạt mean pairwise Jaccard {g["mean_jaccard"]:.4f} ± {g["sample_sd"]:.4f} cho GCN và {gs["mean_jaccard"]:.4f} ± {gs["sample_sd"]:.4f} cho GraphSAGE. Sample SD được tính trên 200 target-level means, mỗi mean gồm 10 cặp seed. RAW float32 logits được xếp hạng theo budget top20%; exact ties được giải bằng global directed edge ID tăng dần theo addendum phê duyệt sau quan sát blocker. Toàn bộ kết quả được tính lại độc lập từ saved edge selections.”

“Đổi secondary key sang ID giảm dần làm thay đổi hai selections của GCN nhưng không đổi pairwise Jaccard hoặc headline means trong lượt đo này. Tuy nhiên, phần lớn sigmoid vectors bị mất độ phân giải, dù RAW logits vẫn biến thiên. Kết quả vì vậy mô tả tính nhất quán của RAW-ranked selections trong một frozen-model case study, không tự chứng minh Fidelity hoặc causal relevance của explanations.”

## Nguồn Evidence

Các bảng lấy từ TASK29_FINAL_STABILITY_RESULTS.json/CSV cùng directory; tie counts và alternative overlap từ TASK29_TIE_SENSITIVITY_ANALYSIS.json; full diagnostics từ TASK29_RAW_LOGIT_RANKING_AUDIT.json. Input/checkpoint/source/runtime hashes và independent recomputation nằm trong TASK29_FINAL_EVALUATION_INTEGRITY.json. Protocol gốc: `D:/CS/KLTN/gnnexplainer-groundtruth-benchmark/docs/audit/elliptic_pg_stability/preflight_manifest.json`; historical comparison: `D:/CS/KLTN/gnnexplainer-groundtruth-benchmark/docs/audit/elliptic_comprehensive_results/02_BACKBONE_AND_PROTOCOL_AUDIT.md` và `03_EXPLANATION_METRICS.md`. Không overwrite các chương luận văn hoặc historical reports.
""",
    )
    md(
        "TASK29_FINAL_HANDOFF.md",
        f"""
# TASK29 Final Handoff: Tie V1

**{ev.VERDICT} — PASS**, đủ coverage và independent verification, có interpretive limitations.

{table}

SD theo200 target-level means, mỗi mean của10 unordered seed-pair Jaccards. Đủ400 target cases,2000 RAW/selected seed-target vectors,4000 pairwise rows. Independent reconstruction từ saved global IDs PASS, tolerance1e-14; max pair difference=0. Có30 unit tests PASS, trong đó15 tests dành cho exact-float32 tie rule.

Original scientific protocol/execution identity không đổi. Post-observation evaluation-only tie addendum mới, policy `{ev.POLICY_ID}`. Original strict STOP và blocked directory giữ nguyên; không sửa Release002/manifests/scientific code. All10 checkpoints không đổi; **không thêm training**. Không reuse old RAW/selected caches; fresh90-vector parity với partial evidence cũ đạt.

GCN2/1000 vectors có boundary ties; alternative DESC ID đổi2 selections nhưng không đổi Jaccard; GraphSAGE không boundary tie. Sigmoid saturation1000/1000 GCN và973/1000 SAGE vectors; RAW all-identical0 ở cả hai. Không claim Stability chứng minh Fidelity/causality, không nâng GCN khỏi diagnostic scope.

Đầu ra chính gồm addendum.md/.json; final results.json; summary/per-target/pairwise.csv; records/400 atomic target JSON+receipts (mỗi record5 seeds); RAW ranking audit.md/.json; tie sensitivity.md/.json; integrity.md/.json; thesis-ready Vietnamese report; historical comparison.json; TEST_RESULTS.json; OUTPUT_SHA256.json. Mọi output bind policy mới qua manifest/records/receipts. Full precision giữ trong JSON/CSV; reports làm tròn để đọc.

Có thể trích mean/SD kèm RAW/top20%/five-seed/200-target/post-observation-addendum qualifiers và giới hạn case study. Không trích như untouched-holdout hoặc evidence explainer causal importance. Historical GNN comparison chỉ descriptive/protocol-limited. Current_master/execution_summary external chưa được cung cấp, nên không attest latest platform-selected master; archive COMPLETE/hash provenance vẫn verify.

**TASK29 có thể đóng trong scope official amended Stability evaluation.** Không cần training thêm hoặc user approval tiếp theo để hoàn tất addendum đã được duyệt. Các hướng Fidelity, cross-platform numerical parity và classifier-seed variation là task riêng, không phải phần đã đo.
""",
    )
    evidence["tests"] = {"count": 30, "tie_policy_tests": 15, "failures": 0, "errors": 0}
    evidence["old_partial_fresh_raw_parity"] = {
        "vectors": parity_count,
        "exact": True,
        "old_caches_admitted": False,
    }
    evidence["historical_comparison"] = {
        "path": "TASK29_HISTORICAL_GNN_COMPARISON.json",
        "verdict": comparison["verdict"],
    }
    evidence["report_sources"] = {
        str(p): sha(p)
        for p in (
            ROOT / "scripts/report_task29_tie_v1.py",
            ROOT / "tests/test_task29_tie_v1.py",
            ROOT / "scripts/verify_task29_tie_v1_outputs.py",
        )
    }
    expected = {**evidence["old_evaluation_snapshot_before"], **evidence["protected_files_before"]}
    assert all(sha(p) == h for p, h in expected.items())
    evidence["final_input_hash_verification"] = {
        "files": len(expected),
        "unchanged": True,
        "at_unix": time.time(),
    }
    for name in (
        "TASK29_FINAL_STABILITY_SUMMARY.csv",
        "TASK29_FINAL_STABILITY_PER_TARGET.csv",
        "TASK29_FINAL_STABILITY_PAIRWISE.csv",
    ):
        rows = list(csv.DictReader((OUT / name).open(encoding="utf-8")))
        assert (
            len(rows)
            == (
                {
                    "TASK29_FINAL_STABILITY_SUMMARY.csv": 2,
                    "TASK29_FINAL_STABILITY_PER_TARGET.csv": 400,
                    "TASK29_FINAL_STABILITY_PAIRWISE.csv": 4000,
                }[name]
            )
        )
        assert all(r["evaluation_policy_identity"] == ev.POLICY_ID for r in rows)
    evidence["output_readback"] = {
        "summary_rows": 2,
        "target_rows": 400,
        "pair_rows": 4000,
        "policy_ids_consistent": True,
    }
    write("TASK29_FINAL_EVALUATION_INTEGRITY.json", evidence)
    results["integrity_receipt_sha256"] = sha(OUT / "TASK29_FINAL_EVALUATION_INTEGRITY.json")
    results["limitations"] = [
        "Post-observation exact-tie addendum",
        "Sigmoid saturation; RAW ranking case study",
        "GCN diagnostic",
        "Single frozen backbone per architecture",
        "Not untouched holdout; no Fidelity/causal conclusion",
    ]
    write("TASK29_FINAL_STABILITY_RESULTS.json", results)
    checks = subprocess.run(
        [
            str(ROOT / ".venv/Scripts/python.exe"),
            "-m",
            "ruff",
            "check",
            "--select",
            "F",
            str(ROOT / "scripts/evaluate_task29_tie_v1.py"),
            str(ROOT / "scripts/report_task29_tie_v1.py"),
            str(ROOT / "tests/test_task29_tie_v1.py"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert checks.returncode == 0, checks.stdout
    diff = subprocess.run(["git", "diff", "--check"], cwd=ROOT, capture_output=True, text=True)
    assert diff.returncode == 0
    write("STATIC_CHECKS.json", {"ruff_f": checks.stdout, "git_diff_check_exit": diff.returncode})
    write("INDEPENDENT_OUTPUT_READBACK.json", independent_output_readback())
    inventory = {
        p.relative_to(OUT).as_posix(): {"sha256": sha(p), "size_bytes": p.stat().st_size}
        for p in OUT.rglob("*")
        if p.is_file()
        and p.name != "OUTPUT_SHA256.json"
        and "__pycache__" not in p.parts
        and p.suffix not in (".pyc", ".tmp")
    }
    write(
        "OUTPUT_SHA256.json",
        {
            "verdict": ev.VERDICT,
            "original_scientific_protocol_identity": ev.frozen.PROTOCOL_ID,
            "files": inventory,
        },
    )
    assert all(sha(OUT / n) == r["sha256"] for n, r in inventory.items())
    print(
        json.dumps(
            {
                "verdict": ev.VERDICT,
                "results": results["results"],
                "output_files": len(inventory),
                "input_unchanged": len(expected),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
