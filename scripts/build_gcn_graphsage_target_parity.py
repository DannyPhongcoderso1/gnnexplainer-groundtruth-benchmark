from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from gnnexplainer_benchmark.data.serialization import load_data_artifact
from gnnexplainer_benchmark.data.validation import validate_ba_house_data
from gnnexplainer_benchmark.explain.subgraph_mapping import computation_subgraph_edge_mask
from gnnexplainer_benchmark.training.checkpointing import (
    build_model_from_checkpoint,
    load_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GCN/GraphSAGE target parity table.")
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/data/ba_house.pt"))
    parser.add_argument(
        "--gcn-checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/ba_house_gcn_best.pt"),
    )
    parser.add_argument(
        "--graphsage-checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/ba_house_graphsage_best.pt"),
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=Path("results/metrics/explainer_test_targets.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/metrics/model_target_parity_gcn_graphsage.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_target_parity(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    eligible = sum(row["eligible_for_ba_house_2x2"] for row in rows)
    print(
        "GCN/GraphSAGE target parity complete | "
        f"targets={len(rows)} | eligible_for_ba_house_2x2={eligible}"
    )
    print(f"output={args.output}")


def build_target_parity(args: argparse.Namespace) -> list[dict[str, int | bool]]:
    data = load_data_artifact(args.artifact)
    validate_ba_house_data(data)
    gcn_logits = predict(data, args.gcn_checkpoint)
    graphsage_logits = predict(data, args.graphsage_checkpoint)
    gcn_predictions = gcn_logits.argmax(dim=-1)
    graphsage_predictions = graphsage_logits.argmax(dim=-1)
    targets = read_targets(args.targets)

    rows = []
    for target in targets:
        node_id = int(target["node_id"])
        label = int(data.y[node_id].item())
        gt_eligible = bool(data.node_mask[node_id].item()) and label > 0
        gcn_mapping_valid = mapping_is_valid(data, node_id, num_hops=3)
        graphsage_mapping_valid = mapping_is_valid(data, node_id, num_hops=3)
        gcn_prediction = int(gcn_predictions[node_id].item())
        graphsage_prediction = int(graphsage_predictions[node_id].item())
        gcn_correct = gcn_prediction == label
        graphsage_correct = graphsage_prediction == label
        rows.append(
            {
                "node_id": node_id,
                "label": label,
                "gcn_prediction": gcn_prediction,
                "graphsage_prediction": graphsage_prediction,
                "gcn_correct": gcn_correct,
                "graphsage_correct": graphsage_correct,
                "gt_explanation_eligible": gt_eligible,
                "candidate_mapping_valid_for_gcn": gcn_mapping_valid,
                "candidate_mapping_valid_for_graphsage": graphsage_mapping_valid,
                "eligible_for_ba_house_2x2": bool(
                    gcn_correct
                    and graphsage_correct
                    and gt_eligible
                    and gcn_mapping_valid
                    and graphsage_mapping_valid
                ),
            }
        )
    return rows


def predict(data: torch.nn.Module, checkpoint_path: Path) -> torch.Tensor:
    checkpoint = load_checkpoint(checkpoint_path)
    model = build_model_from_checkpoint(
        checkpoint,
        in_channels=int(data.x.size(1)),
        out_channels=int(data.y.max().item()) + 1,
    )
    model.eval()
    with torch.no_grad():
        return model(data.x, data.edge_index).cpu()


def mapping_is_valid(data: torch.nn.Module, node_id: int, num_hops: int) -> bool:
    try:
        edge_mask = computation_subgraph_edge_mask(data, node_id, num_hops=num_hops)
    except ValueError:
        return False
    return bool(edge_mask.any().item())


def read_targets(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    main()
