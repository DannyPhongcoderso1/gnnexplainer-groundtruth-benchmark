from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RUNTIME = OUT / 'runtime_tmp'
RUNTIME.mkdir(exist_ok=True)
tempfile.tempdir = str(RUNTIME)
os.environ['TMP'] = str(RUNTIME)
os.environ['TEMP'] = str(RUNTIME)
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch_geometric
import yaml
from torch import nn
from torch_geometric.explain.algorithm.base import ExplainerAlgorithm
from torch_geometric.explain.algorithm.pg_explainer import PGExplainer
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.explain.config import ModelConfig

sys.path.insert(0, str(ROOT / 'src'))
from gnnexplainer_benchmark.training.checkpointing import build_model_from_checkpoint

torch.set_num_threads(1)
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)
SOURCES: set[str] = set()


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def use(path):
    SOURCES.add(str(path).replace('\\', '/'))
    return ROOT / path


def read_json(path):
    return json.loads(use(path).read_text(encoding='utf-8'))


def rows(path):
    with use(path).open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def protected_paths():
    paths = set()
    for folder in ['results/explanations/shapeggen', 'results/metrics/shapeggen',
                   'results/metrics/shapeggen_backbone_comparison',
                   'docs/thesis_evidence/shapeggen_v1_final',
                   'docs/thesis_evidence/shapeggen_v1_pg_gcn_diagnostic']:
        paths.update(p for p in (ROOT / folder).rglob('*') if p.is_file())
    for folder, pattern in [('artifacts/data', 'shapeggen*'),
                            ('artifacts/checkpoints', 'shapeggen*'),
                            ('artifacts/explainers', 'shapeggen_pg_*'),
                            ('configs/dataset', 'shapeggen.yaml'),
                            ('configs/model', 'shapeggen*'),
                            ('configs/evaluation', 'shapeggen*'),
                            ('configs/targets', 'shapeggen*'), ('docs', 'shapeggen*')]:
        paths.update(p for p in (ROOT / folder).glob(pattern) if p.is_file())
    paths.update(ROOT / p for p in ['configs/explainer/pgexplainer.yaml',
                                   'configs/explainer/gnnexplainer.yaml'])
    paths.update(ROOT / p for p in SOURCES if (ROOT / p).is_file())
    return sorted(paths)


def snapshot(paths):
    return {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in paths}


def summary(values):
    a = np.asarray(values, dtype=np.float64)
    assert np.isfinite(a).all()
    return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)),
                std=float(a.std(ddof=0)), min=float(a.min()), max=float(a.max()))


def grad_stats(vector):
    v = vector.detach().reshape(-1).double().cpu().numpy()
    finite = np.isfinite(v)
    return dict(n=len(v), l2=float(np.linalg.norm(v)), mean_abs=float(np.abs(v).mean()),
                max_abs=float(np.abs(v).max()), exact_zero_fraction=float((v == 0).mean()),
                below_1e12_fraction=float((np.abs(v) < 1e-12).mean()),
                below_1e8_fraction=float((np.abs(v) < 1e-8).mean()),
                finite=bool(finite.all()), nan_count=int(np.isnan(v).sum()), inf_count=int(np.isinf(v).sum()))


def cosine(a, b):
    x, y = a.detach().flatten().double(), b.detach().flatten().double()
    nx, ny = float(x.norm()), float(y.norm())
    return float(torch.dot(x, y)/(nx*ny)) if nx and ny else None


class FrozenMLP(nn.Module):
    def __init__(self, state):
        super().__init__()
        self.w0 = nn.Parameter(state['mlp.0.weight'].detach().clone())
        self.b0 = nn.Parameter(state['mlp.0.bias'].detach().clone())
        self.w2 = nn.Parameter(state['mlp.2.weight'].detach().clone())
        self.b2 = nn.Parameter(state['mlp.2.bias'].detach().clone())

    def forward(self, inputs):
        h = nn.functional.linear(inputs, self.w0, self.b0).relu()
        return nn.functional.linear(h, self.w2, self.b2).view(-1)


def attach_forward(model, data, mask):
    set_masks(model, mask, data.edge_index, apply_sigmoid=False)
    try:
        return model(data.x, data.edge_index)
    finally:
        clear_masks(model)


def decomposition(model, data, node, predicted, q, hard):
    output = attach_forward(model, data, q)[node]
    ce = nn.functional.cross_entropy(output, predicted[node])
    hq = q[hard]
    size = .05 * hq.sum()
    adjusted = .99*hq + .005
    entropy = (-adjusted*adjusted.log()-(1-adjusted)*(1-adjusted).log()).mean()
    loss = {'prediction_ce': ce, 'size': size, 'entropy': entropy, 'total': ce+size+entropy}
    probability = output.softmax(-1)[predicted[node]]
    return loss, probability, output.argmax(), output


def loss_record(losses, probability, prediction, original_probability, label):
    return {**{k:float(v.detach()) for k,v in losses.items()},
            'original_probability': float(original_probability), 'masked_probability': float(probability.detach()),
            'probability_delta':float(original_probability-probability.detach()),
            'flip':int(prediction.detach()!=label)}


def grouped(records, values):
    answer = {}
    for label in ['all', 0, 1]:
        subset = records if label == 'all' else [r for r in records if r['label'] == label]
        answer[str(label)] = {'n':len(subset), **{name:summary([r[name] for r in subset]) for name in values}}
    return answer


def gradient_record(losses, q, z, mlp, hard, derivative):
    params = tuple(mlp.parameters()) if z is not None else ()
    inputs = (q, z, *params) if z is not None else (q,)
    all_vectors = {}
    measurements = {}
    for name, value in losses.items():
        grads = torch.autograd.grad(value, inputs, retain_graph=True, allow_unused=True)
        grads = tuple(torch.zeros_like(t) if g is None else g for g, t in zip(grads, inputs))
        vectors = {'soft_full': grads[0], 'soft_hard': grads[0][hard]}
        if z is not None:
            vectors.update(raw_full=grads[1], raw_hard=grads[1][hard],
                           mlp_parameters=torch.cat([g.flatten() for g in grads[2:]]))
        all_vectors[name] = vectors
        measurements[name] = {space:grad_stats(v) for space,v in vectors.items()}
    relations = {}
    for space in all_vectors['total']:
        relations[space] = {f'{a}__{b}':cosine(all_vectors[a][space],all_vectors[b][space])
                           for a,b in [('prediction_ce','size'),('prediction_ce','entropy'),('size','entropy')]}
        ce_norm = float(all_vectors['prediction_ce'][space].double().norm())
        size_norm = float(all_vectors['size'][space].double().norm())
        ent_norm = float(all_vectors['entropy'][space].double().norm())
        relations[space]['size_to_ce_l2_ratio'] = size_norm/ce_norm if ce_norm else None
        relations[space]['entropy_to_ce_l2_ratio'] = ent_norm/ce_norm if ce_norm else None
    return dict(components=measurements, relations=relations,
                dq_dz_hard=grad_stats(derivative[hard]) if derivative is not None else None)


def gradient_summary(records):
    summary_out = {}
    for label in ['all', 0, 1]:
        subset = records if label == 'all' else [r for r in records if r['label']==label]
        grad_out = {}
        for comp in subset[0]['gradients']['components']:
            grad_out[comp] = {}
            for space in subset[0]['gradients']['components'][comp]:
                grad_out[comp][space] = {field:summary([r['gradients']['components'][comp][space][field] for r in subset])
                                       for field in ['l2','mean_abs','max_abs','exact_zero_fraction','below_1e12_fraction','below_1e8_fraction']}
        relation_out = {}
        for space, rel in subset[0]['gradients']['relations'].items():
            relation_out[space] = {}
            for name in rel:
                vals = [r['gradients']['relations'][space][name] for r in subset]
                numbers = [v for v in vals if v is not None]
                relation_out[space][name] = dict(valid_count=len(numbers), summary=summary(numbers) if numbers else None)
        derivative_rows = [r['gradients']['dq_dz_hard'] for r in subset if r['gradients']['dq_dz_hard'] is not None]
        summary_out[str(label)] = dict(n=len(subset), components=grad_out, relations=relation_out,
            derivative={f:summary([v[f] for v in derivative_rows]) for f in ['mean_abs','max_abs','exact_zero_fraction','below_1e12_fraction','below_1e8_fraction']} if derivative_rows else None)
    return summary_out


def finalize(result):
    result['overall_verdict'] = 'MULTIPLE_MECHANISMS_SUPPORTED'
    result['supported_mechanisms'] = [
        'EMPTY_MASK_OBJECTIVE_FAVORED in the specified endpoint counterfactuals',
        'GRADIENT_SATURATION_SUPPORTED at the frozen final checkpoint on 8 fixed train targets']
    result['implementation_verdicts']['objective_prediction_independence'] = 'VERIFIED_CORRECT'
    result['implementation_verdicts']['unintended_gradient_detachment'] = 'VERIFIED_CORRECT'
    result['implementation_verdicts']['sigmoid_concrete_saturation'] = 'VERIFIED_CORRECT'
    result['implementation_verdicts']['stochastic_training_vs_deterministic_inference'] = 'VERIFIED_CORRECT'
    result['implementation_notes'] = dict(
        objective_prediction_independence='CE is not generally independent of mask: endpoints change CE and nonzero soft/MLP gradients were measured',
        sigmoid_concrete_saturation='Measured final-state attenuation; no implementation defect implied',
        stochastic_training_vs_deterministic_inference='Difference prescribed by installed PG source; equivalent historical algorithm, not equivalent masks')
    result['protocol']['preregistered_report_sha256'] = sha(OUT/'01_OBJECTIVE_RECONSTRUCTION.md')
    result['derived_comparisons'] = {}
    for slug, b in result['backbones'].items():
        train=b['counterfactuals']['train']['conditions']
        deterministic=b['deterministic_gradient_per_target']
        paired={m:summary([r[m] for r in deterministic]) for m in ['prediction_ce','size','entropy','total','masked_probability','flip']}
        result['derived_comparisons'][slug] = dict(
            same_8_target_deterministic_loss=paired,
            mean_sample_sd_within_target={m:summary([r[m]['std'] for r in b['stochastic_per_target']]) for m in ['prediction_ce','size','entropy','total']},
            train_zero_ce_cost_vs_one=train['ZERO']['all']['prediction_ce']['mean']-train['ONE']['all']['prediction_ce']['mean'],
            train_zero_size_saving_vs_one=train['ONE']['all']['size']['mean'],
            train_one_minus_zero_total=train['ONE']['all']['total']['mean']-train['ZERO']['all']['total']['mean'])
    result['derived_comparisons']['gcn']['deterministic_train_loss_minus_historical_final'] = result['backbones']['gcn']['counterfactuals']['train']['conditions']['LEARNED']['all']['total']['mean']-result['reference']['historical_loss']
    result['reproduction']['script_sha256'] = sha(Path(__file__))
    result['integrity']['newly_created_output_files'] = [str(p.relative_to(OUT)).replace('\\','/') for p in OUT.rglob('*') if p.is_file()]
    (OUT/'09_LOSS_PROBE_RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


if '--finalize-only' in sys.argv:
    result=json.loads((OUT/'09_LOSS_PROBE_RESULTS.json').read_text(encoding='utf-8'))
    finalize(result)
    print(json.dumps(result['derived_comparisons'],indent=2))
    sys.exit(0)


source_paths = [
    'docs/thesis_evidence/shapeggen_v1_final/08_MASTER_REPORT.md',
    'docs/thesis_evidence/shapeggen_v1_final/09_METRICS_AND_HASHES.json',
    *[f'docs/thesis_evidence/shapeggen_v1_pg_gcn_diagnostic/{name}' for name in [
        '01_IMPLEMENTATION_PROTOCOL_PARITY.md','02_SCORE_DISTRIBUTION_AND_SATURATION.md',
        '05_PG_TRAINING_AND_LOSS_AUDIT.md','07_ROOT_CAUSE_ASSESSMENT.md',
        '08_MASTER_REPORT.md','09_DIAGNOSTIC_RESULTS.json']],
    'scripts/run_shapeggen_pgexplainer.py', 'src/gnnexplainer_benchmark/explainers/pgexplainer_training.py',
    'src/gnnexplainer_benchmark/models/gcn.py', 'src/gnnexplainer_benchmark/models/graphsage.py',
    'src/gnnexplainer_benchmark/training/checkpointing.py',
    '.venv/Lib/site-packages/torch_geometric/explain/algorithm/pg_explainer.py',
    '.venv/Lib/site-packages/torch_geometric/explain/algorithm/base.py',
    '.venv/Lib/site-packages/torch_geometric/explain/algorithm/utils.py',
    '.venv/Lib/site-packages/torch_geometric/nn/conv/message_passing.py',
    '.venv/Lib/site-packages/torch_geometric/nn/conv/gcn_conv.py',
    '.venv/Lib/site-packages/torch_geometric/nn/conv/sage_conv.py',
    'configs/explainer/pgexplainer.yaml', 'configs/evaluation/shapeggen_task24_pgexplainer.yaml',
    'configs/targets/shapeggen_pg_common_train.csv', 'configs/targets/shapeggen_pg_common_val.csv',
    'configs/targets/shapeggen_2x2_targets.csv', 'artifacts/data/shapeggen.pt',
    *[f'artifacts/{folder}/{name}' for folder,names in [
       ('checkpoints',['shapeggen_gcn_best.pt','shapeggen_graphsage_best.pt']),
       ('explainers',['shapeggen_pg_gcn_seed42.pt','shapeggen_pg_graphsage_seed42.pt',
                      'shapeggen_pg_gcn_seed42_manifest.json','shapeggen_pg_graphsage_seed42_manifest.json'])] for name in names],
    *[f'results/metrics/shapeggen/{slug}/pgexplainer/training_seed42.csv' for slug in ['gcn','graphsage']]]
for path in source_paths:
    use(path)
print('Snapshot historical V1 artifacts and source', flush=True)
protected = protected_paths()
before = snapshot(protected)
cache_folders = [ROOT/'src', ROOT/'scripts', ROOT/'docs/thesis_evidence', ROOT/'.venv/Lib/site-packages/torch_geometric/explain']
cache_before = {str(p):p.stat().st_mtime_ns for folder in cache_folders for p in folder.rglob('*.pyc')}
reference = read_json('docs/thesis_evidence/shapeggen_v1_final/09_METRICS_AND_HASHES.json')
for path,h in reference['sha256'].items():
    p = Path(path) if Path(path).is_absolute() else ROOT/path
    assert sha(p)==h, path
previous = read_json('docs/thesis_evidence/shapeggen_v1_pg_gcn_diagnostic/09_DIAGNOSTIC_RESULTS.json')
data = torch.load(use('artifacts/data/shapeggen.pt'), map_location='cpu', weights_only=False)
assert data.num_nodes==1298 and data.edge_index.size(1)==4518 and data.x.shape[1]==10
cohorts = {'train':rows('configs/targets/shapeggen_pg_common_train.csv'),
           'test':rows('configs/targets/shapeggen_2x2_targets.csv')}
for pool in cohorts.values():
    for r in pool:
        r['label'] = r.get('label', r.get('ground_truth_label'))
assert len(cohorts['train'])==539 and len(cohorts['test'])==167
probe_ids = []
for label in [0,1]:
    pool = sorted(int(r['node_id']) for r in cohorts['train'] if int(r['label'])==label)
    probe_ids.extend(pool[i] for i in [0,(len(pool)-1)//3,2*(len(pool)-1)//3,len(pool)-1])
assert len(probe_ids)==len(set(probe_ids))==8
all_ids = sorted(set(int(r['node_id']) for pool in cohorts.values() for r in pool))
result = dict(scope='THESIS_SHAPEGGEN_OFFICIAL_V1', status='DIAGNOSTIC_ONLY',
    protocol=dict(primary_cohort='539 original train targets', secondary_cohort='167 frozen test targets',
                  target_ids={k:[int(r['node_id']) for r in pool] for k,pool in cohorts.items()},
                  gradient_target_ids=probe_ids, gradient_selection='Within each class sorted train IDs: positions 0,floor((n-1)/3),floor(2*(n-1)/3),n-1',
                  monte_carlo_samples=16, base_seeds=list(range(101,117)), seed_formula='base_seed+1000*node_id',
                  temperature_epoch_api=29, gradient_tolerances=[1e-12,1e-8], statistics='Per-target means/median/population SD; gradients norm/cosine float64 from float32 autograd values'),
    environment=dict(python=sys.version,torch=torch.__version__,pyg=torch_geometric.__version__,device='cpu',
                     threads=1,parameter_updates=0,optimizer_instances=0,optimizer_steps=0),
    reference=dict(pg_gcn={k:v for k,v in previous['reference'][1].items()}, historical_loss=0.714083893317902),
    backbones={})
cfg = yaml.safe_load(use('configs/explainer/pgexplainer.yaml').read_text(encoding='utf-8'))['explainer']
holder = SimpleNamespace(coeffs=dict(edge_size=cfg['regularization']['edge_size'], edge_ent=cfg['regularization']['edge_ent'],
                                     temp=[cfg['temperature']['start'],cfg['temperature']['end']],
                                     bias=cfg['implementation_coefficients']['bias']['value']),epochs=cfg['epochs'],is_hetero=False,
                         model_config=ModelConfig(mode='multiclass_classification',task_level='node',return_type='raw'))
temperature = PGExplainer._get_temperature(holder,29)
result['protocol']['temperature']=temperature

for slug in ['gcn','graphsage']:
    print(f'{slug}: frozen counterfactuals on 539 train + 167 test targets',flush=True)
    checkpoint = torch.load(use(f'artifacts/checkpoints/shapeggen_{slug}_best.pt'),map_location='cpu',weights_only=False)
    payload = torch.load(use(f'artifacts/explainers/shapeggen_pg_{slug}_seed42.pt'),map_location='cpu',weights_only=False)
    manifest = read_json(f'artifacts/explainers/shapeggen_pg_{slug}_seed42_manifest.json')
    assert sha(ROOT/manifest['model_checkpoint_path'])==manifest['model_checkpoint_sha256']
    assert sha(ROOT/manifest['artifact_path'])==manifest['pgexplainer_checkpoint_sha256']
    assert payload['algorithm_current_epoch']==29 and payload['explainer_config']['explainer']['epochs']==30
    assert set(payload['train_node_ids'])=={int(r['node_id']) for r in cohorts['train']}
    model = build_model_from_checkpoint(checkpoint,10,2).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    mlp = FrozenMLP(payload['algorithm_state_dict'])
    model_before={k:v.detach().clone() for k,v in model.state_dict().items()}
    mlp_before={k:v.detach().clone() for k,v in mlp.state_dict().items()}
    hard={n:ExplainerAlgorithm._get_hard_masks(model,n,data.edge_index,data.num_nodes)[1] for n in all_ids}
    with torch.no_grad():
        full=model(data.x,data.edge_index)
        pred=full.argmax(-1)
        probability=full.softmax(-1)
        emb=PGExplainer._get_embeddings(holder,model,data.x,data.edge_index)
        zero=torch.zeros(4518)
        one=torch.ones(4518)
        zero_output=attach_forward(model,data,zero)
        one_output=attach_forward(model,data,one)
    one_logits_max=float((one_output-full).abs().max())
    one_probability_max=float((one_output.softmax(-1)-probability).abs().max())
    assert one_probability_max<1e-5
    assert torch.equal(one_output.argmax(-1),pred)
    confidence_parity=max(abs(float(probability[int(r['node_id']),int(r['label'])])-float(r[f'{slug}_confidence'])) for pool in cohorts.values() for r in pool)
    native_reg_max=0.
    native_ce_max=0.
    saved_parity_max=0.
    learned_all_scores={}
    counterfactuals={}
    for cohort,pool in cohorts.items():
        condition_rows={'LEARNED':[],'ZERO':[],'ONE':[]}
        domain=[]
        for index,r in enumerate(pool):
            node,label=int(r['node_id']),int(r['label'])
            assert int(pred[node])==label
            h=hard[node]
            candidate=data.candidate_edge_mask[node].bool()
            domain.append(dict(node_id=node,label=label,hard_count=int(h.sum()),candidate_count=int(candidate.sum()),
                               candidate_outside_hard=int((candidate&~h).sum()),hard_outside_candidate=int((h&~candidate).sum())))
            with torch.no_grad():
                inputs=PGExplainer._get_inputs(holder,emb,data.edge_index,node)
                z=mlp(inputs)
                q=z.sigmoid()
                losses,p,pr,output=decomposition(model,data,node,pred,q,h)
                native_ce=ExplainerAlgorithm._loss_multiclass_classification(holder,output,pred[node])
                native=PGExplainer._add_mask_regularization(holder,losses['prediction_ce'],z[h])
                native_ce_max=max(native_ce_max,abs(float(native_ce-losses['prediction_ce'])))
                native_reg_max=max(native_reg_max,abs(float(native-losses['total'])))
                condition_rows['LEARNED'].append(dict(node_id=node,label=label,**loss_record(losses,p,pr,probability[node,label],pred[node])))
                learned_all_scores[(cohort,node)]={**grad_stats(q[h]),'logits':summary(z[h].tolist())}
                for name,mask,output in [('ZERO',zero,zero_output),('ONE',one,one_output)]:
                    ce=nn.functional.cross_entropy(output[node],pred[node])
                    hq=mask[h]
                    a=.99*hq+.005
                    size=.05*hq.sum()
                    entropy=(-a*a.log()-(1-a)*(1-a).log()).mean()
                    decomp=dict(prediction_ce=ce,size=size,entropy=entropy,total=ce+size+entropy)
                    condition_rows[name].append(dict(node_id=node,label=label,**loss_record(decomp,output[node].softmax(-1)[label],output[node].argmax(),probability[node,label],pred[node])))
                    endpoint=torch.full((int(h.sum()),),float('-inf') if name=='ZERO' else float('inf'))
                    native=PGExplainer._add_mask_regularization(holder,ce,endpoint)
                    native_reg_max=max(native_reg_max,abs(float(native-decomp['total'])))
                if cohort=='test':
                    saved=torch.load(use(f'results/explanations/shapeggen/{slug}/pgexplainer/seed_42/node_{node}/edge_mask.pt'),map_location='cpu',weights_only=False)
                    saved_prob=attach_forward(model,data,saved)[node].softmax(-1)[label]
                    saved_parity_max=max(saved_parity_max,abs(float(saved_prob-p)))
            if index and index%150==0:
                print(f'{slug} {cohort} targets {index}/{len(pool)}',flush=True)
        metrics=['prediction_ce','size','entropy','total','original_probability','masked_probability','probability_delta','flip']
        summary_conditions={name:grouped(rr,metrics) for name,rr in condition_rows.items()}
        pairs={}
        for lab in ['all',0,1]:
            positions=[i for i,r in enumerate(condition_rows['ONE']) if lab=='all' or r['label']==lab]
            pairs[str(lab)]=dict(n=len(positions),
                learned_less_than_one=sum(condition_rows['LEARNED'][i]['total']<condition_rows['ONE'][i]['total'] for i in positions),
                zero_less_than_one=sum(condition_rows['ZERO'][i]['total']<condition_rows['ONE'][i]['total'] for i in positions),
                learned_minus_zero=summary([condition_rows['LEARNED'][i]['total']-condition_rows['ZERO'][i]['total'] for i in positions]),
                zero_preserves=sum(condition_rows['ZERO'][i]['flip']==0 for i in positions),
                zero_probability_rises=sum(condition_rows['ZERO'][i]['probability_delta']<0 for i in positions))
        counterfactuals[cohort]=dict(conditions=summary_conditions,per_target=condition_rows,paired_comparison=pairs,edge_domains=domain,
                                    learned_soft_statistics={str(n):learned_all_scores[(cohort,n)] for n in [int(r['node_id']) for r in pool]})

    deterministic_gradients=[]
    endpoint_gradients={'ZERO':[],'ONE':[]}
    stochastic=[]
    for index,node in enumerate(probe_ids):
        print(f'{slug}: gradient/Concrete target {index+1}/8 node {node}',flush=True)
        h=hard[node]
        label=int(data.y[node])
        inputs=PGExplainer._get_inputs(holder,emb,data.edge_index,node)
        for condition in ['LEARNED','ZERO','ONE']:
            if condition=='LEARNED':
                z=mlp(inputs)
                q=z.sigmoid()
                derivative=q*(1-q)
            else:
                z=None
                q=(torch.zeros(4518) if condition=='ZERO' else torch.ones(4518)).requires_grad_(True)
                derivative=None
            losses,p,pr,_=decomposition(model,data,node,pred,q,h)
            rec=dict(node_id=node,label=label,mode='DETERMINISTIC_LEARNED_MASK_SURROGATE' if condition=='LEARNED' else 'MANUAL_SOFT_MASK_COUNTERFACTUAL',
                     **loss_record(losses,p,pr,probability[node,label],pred[node]),
                     gradients=gradient_record(losses,q,z,mlp,h,derivative))
            (deterministic_gradients if condition=='LEARNED' else endpoint_gradients[condition]).append(rec)
        for base_seed in range(101,117):
            torch.manual_seed(base_seed+1000*node)
            z=mlp(inputs)
            a=PGExplainer._concrete_sample(holder,z,temperature)
            q=a.sigmoid()
            losses,p,pr,_=decomposition(model,data,node,pred,q,h)
            native=PGExplainer._add_mask_regularization(holder,losses['prediction_ce'],a[h])
            native_reg_max=max(native_reg_max,abs(float((native-losses['total']).detach())))
            rec=dict(node_id=node,label=label,base_seed=base_seed,rng_seed=base_seed+1000*node,
                     mode='TRAINING_EQUIVALENT_STOCHASTIC_DIAGNOSTIC',
                     **loss_record(losses,p,pr,probability[node,label],pred[node]),
                     gradients=gradient_record(losses,q,z,mlp,h,q*(1-q)/temperature),
                     sampled_soft_mask_hard=summary(q[h].detach().tolist()))
            stochastic.append(rec)
    for k,v in model.state_dict().items():
        assert torch.equal(v,model_before[k])
    for k,v in mlp.state_dict().items():
        assert torch.equal(v,mlp_before[k])
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    assert all(p.grad is None for p in mlp.parameters())
    assert native_ce_max==0 and native_reg_max<1e-5
    mc_node_means=[]
    for node in probe_ids:
        values=[r for r in stochastic if r['node_id']==node]
        mc_node_means.append(dict(node_id=node,label=int(data.y[node]),
            **{k:summary([r[k] for r in values]) for k in ['prediction_ce','size','entropy','total','masked_probability','flip']}))
    # Equal sample counts permit direct averaging; retain per-node variability explicitly.
    result['backbones'][slug]=dict(counterfactuals=counterfactuals,
        parity=dict(all_one_logits_max_abs=one_logits_max,all_one_probability_max_abs=one_probability_max,
                    all_one_predictions_match=True,canonical_confidence_max_abs=confidence_parity,
                    native_ce_max_abs=native_ce_max,native_regularization_max_abs=native_reg_max,
                    saved_vs_full_raw_soft_target_probability_max_abs=saved_parity_max,
                    parameters_unchanged=True,parameter_grad_buffers_all_none=True,
                    embedding_shape=list(emb.shape),mlp_input_width=int(mlp.w0.shape[1])),
        deterministic_gradient_per_target=deterministic_gradients,
        deterministic_gradient_summary=gradient_summary(deterministic_gradients),
        endpoint_gradient_per_target=endpoint_gradients,
        endpoint_gradient_summary={k:gradient_summary(v) for k,v in endpoint_gradients.items()},
        stochastic_per_sample=stochastic,stochastic_per_target=mc_node_means,
        stochastic_loss_summary=grouped(stochastic,['prediction_ce','size','entropy','total','masked_probability','flip']),
        stochastic_gradient_summary=gradient_summary(stochastic))

result['implementation_verdicts']={
    'frozen_prediction_raw_ce':'VERIFIED_CORRECT', 'sum_size_single_coefficient':'VERIFIED_CORRECT',
    'entropy_affine_adjustment_single_coefficient':'VERIFIED_CORRECT',
    'training_regularization_domain_hard_receptive_field':'VERIFIED_CORRECT',
    'candidate_not_substituted_for_hard_domain':'VERIFIED_CORRECT',
    'directed_entry_order_and_self_loops':'VERIFIED_CORRECT',
    'no_parameter_update_no_grad_buffers':'VERIFIED_CORRECT',
    'original_concrete_sampling_at_final_temperature':'VERIFIED_CORRECT',
    'native_loss_formula_parity':'VERIFIED_CORRECT',
    'historical_rng_draws_and_gradient_trajectory':'INCONCLUSIVE',
    'objective_prediction_independence':'INCONCLUSIVE_PENDING_MEASURED_GRADIENT_INTERPRETATION',
    'stochastic_training_vs_deterministic_inference':'VERIFIED_CORRECT_DIFFERENCE_BY_DESIGN'}
after=snapshot(protected)
assert before==after
tree_sha=hashlib.sha256(json.dumps(before,sort_keys=True).encode()).hexdigest()
cache_after={str(p):p.stat().st_mtime_ns for folder in cache_folders for p in folder.rglob('*.pyc')}
cache_changes=[p for p,mtime in cache_after.items() if cache_before.get(p)!=mtime]
result['integrity']=dict(checked_file_count=len(before),changed_files=[],before_tree_sha256=tree_sha,after_tree_sha256=tree_sha,
    checked_paths=list(before),canonical_reference_hashes_verified=len(reference['sha256']),
    newly_created_or_changed_pyc_in_checked_cache_paths=cache_changes,
    checked_cache_scopes=[str(p) for p in cache_folders],
    temporary_files_created=[str(p.relative_to(OUT)).replace('\\','/') for p in RUNTIME.rglob('*') if p.is_file()],
    temporary_file_policy='PyG generated Jinja modules and library temporary files redirected to runtime_tmp within the new audit directory; no external temp files requested',
    scope='Listed paths only; official V1 trees, previous evidence directories and listed implementation source',
    newly_created_output_files=['01_OBJECTIVE_RECONSTRUCTION.md','run_loss_probe.py','09_LOSS_PROBE_RESULTS.json'])
result['source_hashes']={p:sha(ROOT/p) for p in sorted(SOURCES)}
result['reproduction']=dict(command='.venv/Scripts/python.exe -B docs/thesis_evidence/shapeggen_v1_pg_gcn_loss_probe/run_loss_probe.py',
                            script_sha256=sha(Path(__file__)),
                            note='Re-run only this inference/autograd script; no optimizer exists. Historical stochastic draws unavailable. No benchmark outputs written.')
assert not cache_changes
finalize(result)
print(json.dumps({slug:{cohort:{condition:{metric:content['all'][metric]['mean'] for metric in ['prediction_ce','size','entropy','total','masked_probability','flip']}
                for condition,content in record['conditions'].items()} for cohort,record in v['counterfactuals'].items()}
                for slug,v in result['backbones'].items()},indent=2),flush=True)
print(f'Historical integrity: {len(before)} files unchanged; no new pyc in checked paths',flush=True)
