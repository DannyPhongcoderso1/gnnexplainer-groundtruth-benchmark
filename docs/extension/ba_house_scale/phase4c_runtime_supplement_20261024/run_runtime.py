from __future__ import annotations
import csv, hashlib, json, os, platform, statistics, sys, time
from copy import deepcopy
from pathlib import Path
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"src"))
import torch
import torch_geometric
from gnnexplainer_benchmark.explain.gnnexplainer_runner import run_gnnexplainer_for_node
from gnnexplainer_benchmark.explain.node_selection import TargetNode
from gnnexplainer_benchmark.training.checkpointing import build_model_from_checkpoint
from gnnexplainer_benchmark.explainers.pgexplainer_training import build_pgexplainer_from_config,wrap_pgexplainer,materialize_pgexplainer

REPEATS=3
def read(p): return json.loads((ROOT/p).read_text(encoding="utf-8"))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return torch.load(ROOT/p,map_location="cpu",weights_only=False)
def write(name,value):(OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")
def save_csv(name,rows):
    with (OUT/name).open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def emit(x):print(json.dumps(x),flush=True)
def state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
def unchanged(model,before):return all(torch.equal(v,model.state_dict()[k].detach().cpu()) for k,v in before.items())

def main():
    if (OUT/"results.json").exists():raise RuntimeError("Completed run exists; refusing overwrite")
    torch.set_num_threads(1)
    if not torch.cuda.is_available():raise RuntimeError("CUDA required; no silent CPU fallback")
    device=torch.device("cuda:0")
    protected={};cells=[]
    def protect(p):
        p=Path(p);p=p if p.is_absolute() else ROOT/p
        protected[p.relative_to(ROOT).as_posix()]=sha(p)
    # Protect all existing result reports/metrics in these relevant trees, plus every used input.
    for tree in ["results/metrics/ba_house_2x2","docs/extension/ba_house_scale/phase4a_gnn_random","docs/extension/ba_house_scale/phase4b1_pg_primary","docs/extension/ba_house_scale/master_report"]:
        for p in (ROOT/tree).rglob("*"):
            if p.is_file() and p.suffix in [".json",".csv",".md",".py"]:protect(p)
    for p in ["configs/explainer/gnnexplainer.yaml","configs/explainer/pgexplainer.yaml","src/gnnexplainer_benchmark/explain/gnnexplainer_runner.py","src/gnnexplainer_benchmark/explainers/pgexplainer_training.py","src/gnnexplainer_benchmark/training/checkpointing.py"]:protect(p)
    protect(Path(__file__))
    for scale in ["ba1000_ext","ba1500_ext"]:
        mp=f"artifacts/extensions/ba_house_scale/targets/{scale}/target_manifest.json"
        tp=f"artifacts/extensions/ba_house_scale/targets/{scale}/final_test_targets.json"
        m=read(mp);ids=read(tp)["selected_ids"]
        assert len(ids)==(24 if scale=="ba1000_ext" else 36) and len(set(ids))==len(ids)
        for p in [mp,tp,m["dataset_path"]]:protect(p)
        assert sha(ROOT/m["dataset_path"])==m["dataset_sha256"]
        for bb in ["gcn","graphsage"]:
            gdir=f"artifacts/extensions/ba_house_scale/xai/gnnexplainer/{scale}/{bb}/seed_42"
            pdir=f"artifacts/extensions/ba_house_scale/xai/pgexplainer/{scale}/{bb}/seed_42"
            pm=read(pdir+"/manifest.json")
            assert pm["seed"]==42 and set(pm["test_node_ids"])==set(ids)
            for p in [m["checkpoints"][bb]["path"],pdir+"/manifest.json",pdir+"/final_pg.pt"]:protect(p)
            assert sha(ROOT/m["checkpoints"][bb]["path"])==m["checkpoints"][bb]["sha256"]
            assert sha(ROOT/(pdir+"/final_pg.pt"))==pm["final_checkpoint"]["sha256"]
            for n in ids:
                protect(gdir+f"/node_{n}.pt");protect(pdir+f"/node_{n}.pt")
            cells.append((scale,bb,m,ids,gdir,pdir,pm))
    write("protected_inputs_before.json",protected)
    protocol={
        "date_local":"2026-10-24","timezone":"Asia/Bangkok","purpose":"Supplementary runtime only; frozen quality results unchanged",
        "device":torch.cuda.get_device_name(0),"torch":torch.__version__,"pyg":torch_geometric.__version__,
        "python":sys.version,"platform":platform.platform(),"cuda":torch.version.cuda,
        "num_threads":torch.get_num_threads(),"seed":42,"repeats":REPEATS,
        "warmup":"one discarded generation per method/backbone/scale on first frozen target",
        "timer":"time.perf_counter; CUDA synchronize immediately before and after; wall time seconds",
        "gnn_scope":"existing run_gnnexplainer_for_node: seed, setup, 5-epoch mask optimization and backbone immutability check",
        "pg_scope":"existing extension inference: full graph embedding, src/dst/target concatenation, frozen MLP, sigmoid, zero outside candidate; no embedding cache",
        "excluded":"disk load/save, candidate mask construction, model load, AUROC/AP/fidelity/stability evaluation, score parity comparisons",
        "aggregation":"arithmetic mean over per-target means across 3 repeats; equally weighted targets; sample SD over target means reported separately",
        "training":"historical seed42 training times copied from manifests; no PG/backbone retraining",
        "historical_BA500":"unchanged runtime.csv; historical timer lacks explicit CUDA synchronization; comparison not a controlled scaling experiment",
        "pg_note":"uses exact extension score-generation path, not a replacement masked-graph evaluator",
        "torch_deterministic_algorithms":torch.are_deterministic_algorithms_enabled(),
    }
    write("protocol.json",protocol)
    rows=[];summary=[];parity=[]
    for scale,bb,m,ids,gdir,pdir,pm in cells:
        cpu=load(m["dataset_path"]);data=cpu.clone().to(device)
        ckpt=load(m["checkpoints"][bb]["path"])
        model=build_model_from_checkpoint(ckpt,10,4).to(device).eval()
        for p in model.parameters():p.requires_grad_(False)
        model_before=state(model)
        originals_g={n:load(gdir+f"/node_{n}.pt") for n in ids}
        originals_p={n:load(pdir+f"/node_{n}.pt") for n in ids}
        masks={}
        for n in ids:
            idx=originals_g[n]["candidate_global_indices"]
            mask=torch.zeros(cpu.edge_index.size(1),dtype=torch.bool,device=device);mask[idx.to(device)]=True;masks[n]=mask
            assert set(idx.tolist())==set(originals_p[n]["candidate_global_indices"].tolist())
        cfg=originals_g[ids[0]]["resolved_config"]
        assert cfg["explainer"]["epochs"]==5 and cfg["explainer"]["seed"]==42
        pgpayload=load(pdir+"/final_pg.pt")
        alg=build_pgexplainer_from_config(pgpayload["explainer_config"]).to(device)
        wrapper=wrap_pgexplainer(model,alg)
        materialize_pgexplainer(alg,model,data.x,data.edge_index,ids[0])
        alg.load_state_dict(pgpayload["algorithm_state_dict"],strict=True)
        alg._curr_epoch=int(pgpayload["algorithm_current_epoch"])
        for p in alg.parameters():p.requires_grad_(False)
        alg.mlp.eval()
        pg_before=state(alg)
        def gnn(n):
            return run_gnnexplainer_for_node(model,data,TargetNode(n,1,1,0.0),deepcopy(cfg)).edge_mask
        @torch.no_grad()
        def pg(n):
            emb=alg._get_embeddings(model,data.x,data.edge_index)
            q=alg.mlp(alg._get_inputs(emb,data.edge_index,n)).view(-1).sigmoid()
            q[~masks[n]]=0
            return q
        for method,fn,old,key in [("GNNExplainer",gnn,originals_g,"edge_scores"),("PGExplainer",pg,originals_p,"global_edge_scores")]:
            fn(ids[0]);torch.cuda.synchronize()
            maxdiff=0.;allclose=True
            for repeat in range(1,REPEATS+1):
                times=[]
                for n in ids:
                    torch.cuda.synchronize()
                    started=time.perf_counter()
                    scores=fn(n)
                    torch.cuda.synchronize()
                    elapsed=time.perf_counter()-started
                    q=scores.detach().cpu();ref=old[n][key]
                    assert q.shape==ref.shape and bool(torch.isfinite(q).all())
                    diff=float((q-ref).abs().max());close=bool(torch.allclose(q,ref,atol=1e-6,rtol=1e-5))
                    maxdiff=max(maxdiff,diff);allclose=allclose and close
                    assert elapsed>0
                    rows.append(dict(scale=scale,backbone=bb,method=method,node_id=n,seed=42,repeat=repeat,generation_runtime_seconds=elapsed,score_max_absolute_difference=diff,score_allclose=close))
                    times.append(elapsed)
                save_csv("per_target_runtime.csv",rows)
                emit({"scale":scale,"backbone":bb,"method":method,"repeat":repeat,"targets":len(ids),"mean_seconds":statistics.mean(times)})
            rr=[x for x in rows if x["scale"]==scale and x["backbone"]==bb and x["method"]==method]
            means=[statistics.mean(x["generation_runtime_seconds"] for x in rr if x["node_id"]==n) for n in ids]
            summary.append(dict(configuration=scale,backbone=bb,method=method,target_count=len(ids),repeats=REPEATS,pg_training_seconds=pm["training_runtime_seconds"] if method=="PGExplainer" else "",explanation_seconds_per_node=statistics.mean(means),target_mean_sample_sd_seconds=statistics.stdev(means),repeat_min_seconds=min(x["generation_runtime_seconds"] for x in rr),repeat_max_seconds=max(x["generation_runtime_seconds"] for x in rr),timing_source="SUPPLEMENTARY_20261024_CUDA_SYNCHRONIZED",pg_training_source=pdir+"/manifest.json" if method=="PGExplainer" else "not_applicable"))
            parity.append(dict(scale=scale,backbone=bb,method=method,max_score_absolute_difference=maxdiff,allclose_to_archived_scores=allclose))
        assert unchanged(model,model_before),"Backbone changed"
        assert unchanged(alg,pg_before),"PG parameters changed"
        del wrapper,alg,model,data
        torch.cuda.empty_cache()
    after={p:sha(ROOT/p) for p in protected}
    assert after==protected,"Protected input modified"
    write("protected_inputs_after.json",after)
    assert len(rows)==720 and all(x["score_allclose"] for x in rows),"Score parity failed"
    base=list(csv.DictReader((ROOT/"results/metrics/ba_house_2x2/runtime.csv").open(encoding="utf-8-sig")))
    table=[]
    for x in base:
        table.append(dict(configuration="BA500",backbone=x["backbone"],method="PGExplainer" if x["explainer"].startswith("PGExplainer") else x["explainer"],pg_training_seconds=float(x["upfront_training"].split()[0]) if x["upfront_training"]!="none" else "",explanation_seconds_per_node=float(x["per_target_generation_or_inference_mean_seconds"]),target_count=12,repeats="historical",generation_source="results/metrics/ba_house_2x2/runtime.csv"))
    for x in summary:
        table.append(dict(configuration="BA1000" if x["configuration"]=="ba1000_ext" else "BA1500",backbone="GCN" if x["backbone"]=="gcn" else "GraphSAGE",method=x["method"],pg_training_seconds=x["pg_training_seconds"],explanation_seconds_per_node=x["explanation_seconds_per_node"],target_count=x["target_count"],repeats=REPEATS,generation_source="per_target_runtime.csv"))
    save_csv("extension_runtime_summary.csv",summary);save_csv("runtime_three_configurations.csv",table)
    result={"status":"PASS","protocol":protocol,"summary":summary,"table":table,"score_parity":parity,"integrity":{"protected_files":len(protected),"changed_files":[],"in_memory_backbone_and_pg_state_unchanged":True},"execution":{"PG_training_updates":0,"backbone_training_updates":0,"GNN_measured_generations":360,"PG_measured_generations":360,"discarded_warmup_generations":8},"completed_at_utc":datetime.now(timezone.utc).isoformat()}
    write("results.json",result)
    def num(v):return "–" if v=="" else f"{v:.4f}".replace(".",",")
    lines=["# BA-House — runtime đủ ba cấu hình","","Bổ sung ngày 2026-10-24; seed42, GPU "+protocol["device"]+". Chỉ đo bổ sung BA1000/BA1500; BA500 giữ bản ghi lịch sử.","","| Cấu hình | Backbone | Phương pháp | Huấn luyện PG (giây) | Tạo giải thích (giây/đỉnh) |","|---|---|---|---:|---:|"]
    lines += [f"| {x['configuration']} | {x['backbone']} | {x['method']} | {num(x['pg_training_seconds'])} | {num(x['explanation_seconds_per_node'])} |" for x in table]
    lines += ["","## Giao thức và giới hạn","","BA1000:24 đỉnh; BA1500:36 đỉnh, đúng frozen final targets, cả hai backbone. Mỗi đỉnh đo3 lần sau1 lượt warmup mỗi cell/method; mean qua target và repeat. GNNExplainer tối ưu mask5epochs như run gốc. PG sử dụng frozen checkpoint, tính lại full-graph embeddings mỗi đỉnh; không cache. Timer wall-clock có CUDA synchronize trước/sau; không gồm I/O, nạp model, tính candidate hay quality metrics.","","Score tái tạo khớp artifact gốc theo atol1e−6/rtol1e−5; chi tiết max difference ở results.json.score_parity. Không sửa saved explanations, official quality metrics, backbone hoặc PG weights. Training time lấy nguyên từ manifests lịch sử; không retrain PG/backbone. Timing mới và BA500 lịch sử khác phiên đo và quy tắc đồng bộ GPU; không quy chênh lệch trực tiếp cho graph size.","","Nguồn: runtime_three_configurations.csv (12rows); per_target_runtime.csv (720rows); extension_runtime_summary.csv (8cells); protocol.json; results.json; protected_inputs_before/after.json. Bản ghi PG training chỉ cộng training loops có instrumentation, loại diagnostics ở ngoài epochs; không phải tổng wall time pipeline. BA500 natural pools48/106 khác extension common pools94/141.","","Script run_runtime.py tái lập đo mới, không ghi đè input gốc; cố tình từ chối nếu results.json đã tồn tại. Không dùng số runtime đo mới để thay đổi model selection hoặc bảng quality đã đóng băng."]
    (OUT/"RUNTIME_TABLE.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    emit({"status":"PASS","rows":len(rows),"protected_files":len(protected),"table":table})
if __name__=="__main__":main()
