import os
os.environ['JAX_PLATFORMS']='cpu';os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,json,hashlib,dataclasses,random,argparse
P=Path(__file__).resolve().parent;R=P.parents[1];sys.path.insert(0,str(R/'src'));sys.path.insert(0,str(P))
import numpy as np,torch,pyarrow.parquet as pq,pyarrow.compute as pc,pyarrow as pa
import safetensors.torch
from openpi.shared import normalize
from openpi.training import config as cfg
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from train_pytorch_warmstart import normalize_state_dict_for_loading
parser=argparse.ArgumentParser()
parser.add_argument('--base-checkpoint',type=Path,required=True)
parser.add_argument('--data-root',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
O=args.output.expanduser().resolve();O.mkdir(parents=True,exist_ok=True)
if (O/'common_init').exists():raise FileExistsError('Use a fresh preparation directory; existing initialization is never overwritten')
DATA=args.data_root.expanduser().resolve();split=json.loads((P/'episode_split.json').read_text());ids=split['train_episodes']
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def arrays(arm):
 tabs=[pq.read_table(f,columns=['episode_index','frame_index','state','actions']) for f in sorted((DATA/arm/'data').glob('*/*.parquet'))]
 t=pa.concat_tables(tabs);t=t.filter(pc.is_in(t['episode_index'],value_set=pa.array(ids)));t=t.take(pc.sort_indices(t,sort_keys=[('episode_index','ascending'),('frame_index','ascending')]))
 return np.array(t['episode_index']),np.array(t['frame_index']),np.asarray(t['state'].to_pylist(),np.float32),np.asarray(t['actions'].to_pylist(),np.float32)
a=arrays('single');assert len(a[0])==7547
for arm in ['multi','multi_fixed']:
 b=arrays(arm);assert all(np.array_equal(x,y) for x,y in zip(a,b)),arm
stats={k:normalize.RunningStats() for k in ['state','actions']}
for ep in ids:
 m=a[0]==ep;states=a[2][m];actions=a[3][m];assert np.array_equal(a[1][m],np.arange(len(states)))
 chunks=actions[np.minimum(np.arange(len(actions))[:,None]+np.arange(50),len(actions)-1)].copy();chunks[:,:,:6]-=states[:,None,:6]
 stats['state'].update(states);stats['actions'].update(chunks)
N=O/'norm_assets/pudding_train30';normalize.save(N,{k:v.get_statistics() for k,v in stats.items()})
(O/'norm_receipt.json').write_text(json.dumps({'frames':7547,'episodes':ids,'all_three_raw_state_action_equal':True,'action_horizon':50,'delta_first6':True,'episode_end_padding':True,'sha256':sha(N/'norm_stats.json')},indent=2));print('NORM_READY',flush=True)
meta=json.loads((P/'checkpoint_metadata.json').read_text())['multi']['config']['model'];base=cfg.get_config('pi0_libero_object').model;fields={f.name for f in dataclasses.fields(base)};mc=dataclasses.replace(base,**{k:v for k,v in meta.items() if k in fields})
torch.manual_seed(42);np.random.seed(42);random.seed(42);torch.set_num_threads(8)
model=PI0Pytorch(dataclasses.replace(mc,control_attention_enabled=False,object_use_control=False,depth_use_control=False,skill_use_control=False));del model.paligemma_with_expert.gemma_expert.lm_head
src=args.base_checkpoint.expanduser().resolve()/'model.safetensors';state=normalize_state_dict_for_loading(safetensors.torch.load_file(src),source_label='original pi0 base')
state.pop('paligemma_with_expert.gemma_expert.lm_head.weight',None)
missing,unexpected=model.load_state_dict(state,strict=False)
assert not unexpected and not [k for k in missing if not k.startswith(('depth','skill_head.'))],(missing,unexpected)
for k,v in state.items():assert torch.equal(model.state_dict()[k],v),k
del state
model.config=mc
model.enable_control_attention(num_control_heads=8,copy_weights=False,freeze_origin=False,use_headwise_gate=None)
for name,t in model.state_dict().items():
 if 'zero_conv' in name:assert torch.count_nonzero(t)==0,name
out=O/'common_init';out.mkdir(exist_ok=False)
safetensors.torch.save_file({k:v.detach().cpu().contiguous().clone() for k,v in model.state_dict().items()},str(out/'model.safetensors'))
manifest={'source_checkpoint':str(src),'source_sha256':sha(src),'model_sha256':sha(out/'model.safetensors'),'base_loaded_strictly':True,'shared_ca_initialization':True,'ca_seed':42,'ca_copy_weights':False,'zero_conv_zero':True,'model_config':dataclasses.asdict(mc),'num_tensors':len(model.state_dict())}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print('COMMON_INIT_READY',manifest['model_sha256'],flush=True)
