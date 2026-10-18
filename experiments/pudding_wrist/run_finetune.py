"""Default: CPU config inspection only. --execute explicitly enables training."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import argparse,json,pathlib,sys,dataclasses,importlib.util
P=pathlib.Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--arm',required=True,choices=['single_task_ft','multi_task_ft','multi_wrist_fixed_task_ft'])
parser.add_argument('--execute',action='store_true')
parser.add_argument('--check-data',action='store_true')
parser.add_argument('--data-root',type=pathlib.Path,required=True)
parser.add_argument('--single-checkpoint',type=pathlib.Path,required=True)
parser.add_argument('--multi-checkpoint',type=pathlib.Path,required=True)
parser.add_argument('--tokenizer',type=pathlib.Path,required=True)
parser.add_argument('--output',type=pathlib.Path,default=pathlib.Path('checkpoints/pudding_wrist_ablation'))
args=parser.parse_args();c=json.loads((P/(args.arm+'.json')).read_text());split=json.loads((P/'episode_split.json').read_text());R=P.parents[1];c['dataset_root']=str(args.data_root.resolve()/({'single_task_ft':'single','multi_task_ft':'multi','multi_wrist_fixed_task_ft':'multi_fixed'}[args.arm]));c['checkpoint']=str((args.single_checkpoint if args.arm=='single_task_ft' else args.multi_checkpoint).resolve());sys.path.insert(0,str(R/'src'))
os.environ['OPENPI_PALIGEMMA_TOKENIZER_PATH']=str(args.tokenizer.resolve())
if not args.tokenizer.is_file():raise FileNotFoundError(args.tokenizer)
from openpi.training import config as cfg, data_loader as dl
from openpi.training import optimizer as opt
base=cfg.get_config('pi0_libero_object')
meta=json.loads((P/'checkpoint_metadata.json').read_text())['single' if args.arm.startswith('single') else 'multi']['config']
model_fields={x.name for x in dataclasses.fields(base.model)}
mc=dataclasses.replace(base.model,**{k:v for k,v in meta['model'].items() if k in model_fields})
dc=cfg.LeRobotLiberoDataConfig(repo_id=c['dataset_repo_id'],assets=cfg.AssetsConfig(assets_dir=str(args.single_checkpoint.resolve()/'assets'),asset_id='ybwowen/libero'),base_config=cfg.DataConfig(prompt_from_task=True,use_object_loss=True,local_root_dir=c['dataset_root'],train_val_split=30/35,split_seed=42),extra_delta_transform=True)
conf=dataclasses.replace(base,name='pudding_wrist_ablation',exp_name=args.arm+'_b16_s6000_seed42',model=mc,data=dc,pytorch_weight_path=c['checkpoint'],pytorch_training_precision='float32',use_gradient_checkpointing=True,backbone_lr_scale=1.,batch_size=16,num_train_steps=6000,num_workers=0,seed=42,lr_schedule=opt.CosineDecaySchedule(warmup_steps=100,peak_lr=5e-6,decay_steps=6000,decay_lr=5e-7),optimizer=opt.AdamW(b1=.9,b2=.95,eps=1e-8,weight_decay=1e-10,clip_gradient_norm=1.),object_loss_weight=.001,save_interval=1000,keep_period=1000,val_interval=250,val_max_batches=None,log_interval=20,checkpoint_base_dir=str(args.output.resolve()),resume=False,overwrite=False,wandb_enabled=False)
# Explicit filtering: this installed LeRobot v3 loader ignores episodes for local data.
class SelectedEpisodes(dl.lerobot_dataset.LeRobotDataset):
    def load_hf_dataset(self):
        import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
        import datasets
        from lerobot.datasets.utils import get_hf_features_from_features, hf_transform_to_torch
        manifest=json.loads((P/'task9_rows.json').read_text())
        files=sorted({r['file'] for r in manifest if r['episode'] in self.episodes})
        tables=[]
        for rel in files:
            table=pq.read_table(self.root/rel)
            table=table.filter(pc.is_in(table['episode_index'],value_set=pa.array(self.episodes)))
            tables.append(table)
        table=pa.concat_tables(tables)
        table=table.take(pc.sort_indices(table,sort_keys=[('episode_index','ascending'),('frame_index','ascending')]))
        episodes=table['episode_index'].to_pylist(); frames=table['frame_index'].to_pylist()
        self.selected_bounds={}
        for ep in self.episodes:
            positions=[i for i,x in enumerate(episodes) if x==ep]
            assert positions and [frames[i] for i in positions]==list(range(len(positions)))
            self.selected_bounds[ep]=(positions[0],positions[-1]+1)
        features=get_hf_features_from_features(self.features)
        ds=datasets.Dataset.from_dict(table.to_pydict(),features=features)
        ds.set_transform(hf_transform_to_torch)
        return ds
    def _get_query_indices(self,idx,ep_idx):
        import torch
        start,end=self.selected_bounds[ep_idx]
        queries={k:[max(start,min(end-1,idx+d)) for d in ds] for k,ds in self.delta_indices.items()}
        padding={k+'_is_pad':torch.BoolTensor([idx+d<start or idx+d>=end for d in ds]) for k,ds in self.delta_indices.items()}
        return queries,padding
def episode_dataset(data_config,action_horizon,model_config,*,split='all'):
    ids=globals()['split']['train_episodes'] if split=='train' else globals()['split']['validation_episodes'] if split=='val' else sorted(globals()['split']['train_episodes']+globals()['split']['validation_episodes'])
    md=dl.lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id,root=data_config.local_root_dir)
    ds=SelectedEpisodes(data_config.repo_id,root=data_config.local_root_dir,episodes=ids,delta_timestamps={k:[t/md.fps for t in range(action_horizon)] for k in data_config.horizon_sequence_keys})
    actual=set(int(x) for x in ds.hf_dataset['episode_index'])
    assert actual==set(ids),(actual,ids)
    assert set(int(x) for x in ds.hf_dataset['task_index'])=={9}
    expected=globals()['split']['train_frames'] if split=='train' else globals()['split']['validation_frames'] if split=='val' else 8726
    assert len(ds)==expected,(len(ds),expected)
    for ep,(start,end) in ds.selected_bounds.items():
        q,pad=ds._get_query_indices(end-1,ep)
        assert all(all(x==end-1 for x in v) for v in q.values())
        assert all(bool(v[1:].all()) and not bool(v[0]) for v in pad.values())
    if data_config.prompt_from_task:ds=dl.TransformedDataset(ds,[dl._transforms.PromptFromLeRobotTask(md.tasks)])
    return ds
dl.create_torch_dataset=episode_dataset
if __name__ == '__main__':
    if args.execute:
        if int(os.environ.get('WORLD_SIZE','1'))!=4:raise RuntimeError('Use torchrun with 4 processes on the intended A100 host.')
        if not (pathlib.Path(c['dataset_root'])/'subset_manifest.json').exists():raise RuntimeError('Exported subset manifest required.')
        if conf.checkpoint_dir.exists() and any(conf.checkpoint_dir.iterdir()):raise RuntimeError('Output already contains data; choose a new --output or implement explicit resume.')
        spec=importlib.util.spec_from_file_location('pudding_trainer',P/'train_pytorch_warmstart.py');mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
        mod.init_logging()
        mod.train_loop(conf,lambda_object=.001,lambda_skill=0.)
        if int(os.environ.get('RANK','0'))==0:
            for name in ['model.safetensors','optimizer.pt','metadata.pt']:
                assert (conf.checkpoint_dir/'6000'/name).is_file(),name
    else:
        data=conf.data.create(conf.assets_dirs,conf.model)
        print(json.dumps({'arm':args.arm,'checkpoint':conf.pytorch_weight_path,'dataset_root':data.local_root_dir,'batch_size_global':conf.batch_size,'steps':conf.num_train_steps,'norm_stats_loaded':data.norm_stats is not None,'train_demos':30,'val_demos':5,'training_started':False},indent=2))
        if args.check_data:
            for sp in ['train','val']:
                ds=episode_dataset(data,50,conf.model,split=sp)
                sample=ds[0]
                print(sp,len(ds),'actions',sample['actions'].shape,'has_weighted', 'wrist_object_masks' in sample)
