from pathlib import Path
import argparse,json,shutil,hashlib
import pyarrow as pa,pyarrow.parquet as pq,pyarrow.compute as pc
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);a=p.parse_args()
rows=json.loads(a.manifest.read_text());eps=sorted({x['episode'] for x in rows});files=sorted({x['file'] for x in rows});report={}
for arm,ds in [('single','ybwowen_libero'),('multi','libero_sam2_weighted'),('multi_fixed','libero_sam2_weighted_pudding_wrist_fixed_20261018')]:
 src=a.repo/'data/datasets'/ds;dest=a.out/arm
 if dest.exists():raise RuntimeError(f'Refusing overwrite {dest}')
 dest.mkdir(parents=True);shutil.copytree(src/'meta',dest/'meta')
 # Original IDs and metadata retained for provenance; subset_manifest is authoritative for selection.
 count=0;hashes={}
 for rel in files:
  t=pq.read_table(src/rel);t=t.filter(pc.is_in(t['episode_index'],value_set=pa.array(eps)));assert set(t['task_index'].to_pylist())=={9}
  q=dest/rel;q.parent.mkdir(parents=True,exist_ok=True);pq.write_table(t,q,compression='zstd');count+=len(t);hashes[rel]=hashlib.sha256(q.read_bytes()).hexdigest()
 assert count==8726
 manifest={'format':'GuidedVLA sparse episode subset; use included selected-episode loader','source_dataset':ds,'task_index':9,'episodes':eps,'num_episodes':35,'num_frames':count,'metadata_note':'Source metadata retained for original IDs/provenance. Manifest defines actual subset; do not use default unfiltered LeRobot loader.','sha256':hashes}
 (dest/'subset_manifest.json').write_text(json.dumps(manifest,indent=2));size=sum(f.stat().st_size for f in dest.rglob('*') if f.is_file());report[arm]={'bytes':size,'GiB':size/2**30,'GB':size/1e9,'num_episodes':35,'num_frames':count};print(arm,report[arm],flush=True)
report['total_bytes']=sum(x['bytes'] for x in report.values());(a.out/'sizes.json').write_text(json.dumps(report,indent=2))
