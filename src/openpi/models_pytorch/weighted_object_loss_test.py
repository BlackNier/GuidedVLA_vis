import math
import unittest
import torch
from openpi.models_pytorch.weighted_object_loss import weighted_object_mass_loss
from openpi.shared.weighted_object_targets import pack_targets


class WeightedLossTest(unittest.TestCase):
    def data(self):
        maps=torch.zeros(1,3,2,256);maps[0,0,0,0]=1;maps[0,0,1,1]=1
        weights=torch.tensor([[[.7,.3],[0,0],[0,0]]])
        valid=weights>0
        attn=torch.zeros(1,2,3,768);attn[:,:,:,0]=.8;attn[:,:,:,1]=.2
        return attn,maps,weights,valid

    def test_numeric_and_weight_rescaling(self):
        a,m,w,v=self.data()
        loss=weighted_object_mass_loss(a,1,(0,1),m,w,v)
        self.assertAlmostEqual(loss.item(),-.7*math.log(.8)-.3*math.log(.2),places=6)
        self.assertAlmostEqual(loss.item(),weighted_object_mass_loss(a,1,(0,1),m,w*20,v).item(),places=6)
        self.assertAlmostEqual(loss.item(),weighted_object_mass_loss(a,1,(0,1),m,w,v,'per_head').item(),places=6)

    def test_neglected_object_has_penalty_and_gradient(self):
        a,m,w,v=self.data();a.requires_grad_()
        loss=weighted_object_mass_loss(a,1,(0,1),m,w,v);loss.backward()
        self.assertLess(a.grad[0,0,1,0],0);self.assertLess(a.grad[0,0,1,1],0)
        ignored=a.detach().clone();ignored[:,:,:,0]=.999;ignored[:,:,:,1]=.001
        self.assertGreater(weighted_object_mass_loss(ignored,1,(0,1),m,w,v).item(),loss.item())

    def test_padding_invalid_and_empty_batch(self):
        a,m,w,v=self.data();v[:,:,1]=False
        self.assertAlmostEqual(weighted_object_mass_loss(a,1,(0,),m,w,v).item(),-math.log(.8),places=6)
        v[:]=False;a.requires_grad_()
        loss=weighted_object_mass_loss(a,1,(0,),m,w,v);self.assertEqual(loss.item(),0)
        loss.backward();self.assertTrue(torch.isfinite(a.grad).all());self.assertEqual(a.grad.abs().sum().item(),0)

    def test_pack_allowed_views(self):
        data={}
        for prefix in ['agentview','wrist']:
            data[prefix+'_object_masks']=torch.ones(2,4,256)
            data[prefix+'_object_weights']=torch.ones(2,4)
            data[prefix+'_object_valid']=torch.ones(2,4,dtype=torch.bool)
        out=pack_targets(data,{'base_0'})
        self.assertEqual(out['per_object_maps'].shape,(2,3,4,256))
        self.assertTrue(out['per_object_valid'][:,0].all())
        self.assertFalse(out['per_object_valid'][:,1:].any())
        data['agentview_object_weights'][0,0]=-1
        with self.assertRaises(ValueError):pack_targets(data)

    def test_training_target_transport_keeps_groups(self):
        import ast
        from pathlib import Path
        source=Path(__file__).resolve().parents[3]/'scripts/train_pytorch.py'
        tree=ast.parse(source.read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='prepare_object_targets')
        ns={'torch':torch,'_OBJECT_TARGET_KEYS':('object_maps','object_masks'),
            'get_empty_object_targets':lambda b,d:{'object_maps':torch.zeros(b,3,256),'object_masks':torch.zeros(b,3,dtype=torch.bool)}}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),ns)
        _,m,w,v=self.data()
        targets=dict(object_maps=m.amax(2),object_masks=v.any(2),per_object_maps=m,per_object_weights=w,per_object_valid=v)
        result=ns['prepare_object_targets'](targets,torch.device('cpu'),1,use_object_loss=True)
        self.assertEqual(set(result),set(targets))
        self.assertEqual(result['per_object_valid'].dtype,torch.bool)

    def test_loader_selects_weighted_path_without_union_map(self):
        import ast,types,logging
        from pathlib import Path
        source=Path(__file__).resolve().parents[1]/'training/data_loader.py'
        tree=ast.parse(source.read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and any(isinstance(f,ast.FunctionDef) and f.name=='_pack_maps_to_tensor' for f in n.body))
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__iter__')
        ns={'logging':logging,'_model':types.SimpleNamespace(Observation=types.SimpleNamespace(from_dict=lambda b,**kw:b))}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),ns)
        data={}
        for prefix in ['agentview','wrist']:
            data[prefix+'_object_masks']=torch.ones(1,2,256)
            data[prefix+'_object_weights']=torch.tensor([[.7,.3]])
            data[prefix+'_object_valid']=torch.ones(1,2,dtype=torch.bool)
        fake=types.SimpleNamespace(_data_loader=[{'weighted_objects':data,'actions':torch.zeros(1,2,7)}],
            _use_object_loss=True,_allowed_object_views_set={'base_0','left_wrist_0'},_logged_object_map_keys=False,_framework='pytorch')
        _,_,targets=next(ns['__iter__'](fake))
        self.assertEqual(targets['per_object_maps'].shape,(1,3,2,256))

if __name__=='__main__':unittest.main()
