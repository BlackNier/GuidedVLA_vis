"""Dependency-free regression test for AssetsConfig local /gs path handling."""
import ast
import pathlib
import types
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]

class PathTests(unittest.TestCase):
    def test_local_mount_and_explicit_cloud_uri(self):
        tree = ast.parse((ROOT / 'src/openpi/training/config.py').read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_load_norm_stats')
        method.decorator_list = []
        method.returns = None
        for arg in method.args.args:
            arg.annotation = None
        calls = []
        class CloudPath:
            def __init__(self, raw):
                assert raw.startswith('gs://'), raw
                self.raw = raw
            def __truediv__(self, child): return CloudPath(self.raw.rstrip('/') + '/' + child)
            def __str__(self): return self.raw
        scope = {'pathlib': pathlib, 'epath': types.SimpleNamespace(Path=CloudPath),
                 '_download': types.SimpleNamespace(maybe_download=lambda path: calls.append(path) or path),
                 '_normalize': types.SimpleNamespace(load=lambda path: path),
                 'logging': types.SimpleNamespace(info=lambda *args: None)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), '<method>', 'exec'), scope)
        fn = scope['_load_norm_stats']
        for root in ['/gs/bs/work/checkpoint/assets', '/gs/bs/tga-shinoda/checkpoint/assets', '/tmp/assets']:
            self.assertEqual(fn(None, root, 'pudding_train30'), root + '/pudding_train30')
        self.assertEqual(fn(None, 'gs://openpi-assets/assets', 'x'), 'gs://openpi-assets/assets/x')
        self.assertIsNone(fn(None, '/gs/bs/assets', None))
        # Caller must pass the raw local path; epath conversion here recreates the bug.
        create = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'create_base_config')
        call = next(n for n in ast.walk(create) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == '_load_norm_stats')
        self.assertIsInstance(call.args[0], ast.BoolOp)

if __name__ == '__main__': unittest.main()
