"""CPU regressions: python -B tests/test_cache_semantic_lora.py.

This test baseline has no segment LoRA or FFV1 implementation. Test artifacts
are retained in the OS temp directory; no cache deletion is exercised.
"""

import importlib
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = '_cache_semantic_tests'
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
folder_paths = types.ModuleType('folder_paths')
folder_paths.get_input_directory = lambda: str(ROOT)
folder_paths.get_output_directory = lambda: str(ROOT)
sys.modules.setdefault('folder_paths', folder_paths)
comfy = types.ModuleType('comfy')
comfy.__path__ = []
utils = types.ModuleType('comfy.utils')
utils.common_upscale = lambda tensor, width, height, *_: torch.nn.functional.interpolate(
    tensor, size=(height, width), mode='nearest')
comfy.utils = utils
sys.modules.setdefault('comfy', comfy)
sys.modules.setdefault('comfy.utils', utils)

cache = importlib.import_module(f'{PACKAGE}.director.segment_cache')
bridge = importlib.import_module(f'{PACKAGE}.director.semantic_bridge')
gen = importlib.import_module(f'{PACKAGE}.director.gen_timeline')

# Import the real Director class; stub only execution/UI infrastructure.
node_package = types.ModuleType(f'{PACKAGE}.nodes')
node_package.__path__ = [str(ROOT / 'nodes')]
node_stubs = {f'{PACKAGE}.nodes': node_package,
              'comfy.samplers': types.ModuleType('comfy.samplers')}
for module_name, names in {
        'director.executor_core': ['execute_director_plan_core'],
        'nodes.director_common': ['finalize_director_outputs', 'prepare_director_plan',
                                  'timeline_required_inputs', 'director_perf_inputs'],
        'nodes.director_refine': ['director_face_refine_widget_inputs',
                                  'director_refine_widget_inputs', 'director_selflift_widget_inputs',
                                  'director_semantic_bridge_widget_inputs']}.items():
    module = types.ModuleType(f'{PACKAGE}.{module_name}')
    for name in names:
        setattr(module, name, lambda *args, **kwargs: None)
    node_stubs[module.__name__] = module
with patch.dict(sys.modules, node_stubs):
    director_node = importlib.import_module(f'{PACKAGE}.nodes.director')


def make_plan():
    return gen.build_gen_director_plan(
        dict(timelineMode='prompt_batch', global_={}, frameRate=24,
             output=dict(mode='fixed', width=64, height=64, continuityEnabled=False),
             segments=[dict(taskType='t2v', frameCount=5, prompt='test')]),
        global_task_type='mixed', global_prompt='', total_frames=5,
        frame_rate=24, width=64, height=64, ref_max_size=64, load_media=False)


class CacheSemanticTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='director-cache-regressions-'))
        self.plan = make_plan()
        self.seg = self.plan.segments[0]
        self.patches = [patch.object(cache, '_cache_root', return_value=self.root),
                        patch.object(cache, 'h3_output_path', return_value=self.root)]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def save(self, value=1, **kwargs):
        latent = dict(samples=(torch.tensor([float(value)]),))
        frames = torch.full((5, 2, 2, 3), value / 10)
        cache.save_first_pass_cache('node', self.seg, self.plan, av_latent=latent,
                                    frames=frames, **kwargs)

    def meta(self):
        return json.loads((self.root / 'seg_0000.pre.meta.json').read_text())

    def assert_miss(self):
        self.assertIsNone(cache.load_first_pass_cache('node', self.seg, self.plan))
        self.assertIsNone(cache.load_first_pass_av_latent(
            'node', self.seg, self.plan, allow_stale=True))
        self.assertIsNone(cache.load_first_pass_frames_stale('node', self.seg, self.plan))
        self.assertFalse(cache.inspect_first_pass_cache('node', self.plan)['matches'])

    def test_complete_generation_roundtrip_and_panel(self):
        self.save(handoff=dict(export_frames=3), low_carry=dict(samples=(torch.ones(1),)))
        result = cache.load_first_pass_cache('node', self.seg, self.plan)
        self.assertEqual(result['av_latent']['samples'][0].item(), 1)
        self.assertEqual(result['frames'].shape[0], 5)
        self.assertEqual(result['handoff'], dict(export_frames=3))
        self.assertIsNotNone(result['low_carry'])
        self.assertEqual(cache.load_first_pass_frames_stale(
            'node', self.seg, self.plan).shape[0], 3)
        self.assertTrue(cache.inspect_first_pass_cache('node', self.plan)['matches'])
        self.assertTrue(cache._cache_fingerprint_matches(
            self.meta(), cache.first_pass_cache_fingerprint(self.seg, self.plan)))

    def test_write_failure_at_each_artifact_and_meta_is_miss(self):
        real_save = torch.save
        real_publish = cache._publish_first_pass_json
        real_write = Path.write_text
        for failure in ['av.pt', 'low.pt', 'handoff', 'frames', 'meta']:
            self.save()
            old_meta = self.meta()
            def save_or_fail(payload, path):
                if (failure in ['av.pt', 'low.pt'] and str(path).endswith(failure)) or (
                        failure == 'frames' and str(path).endswith('.pt') and
                        not str(path).endswith(('.av.pt', '.low.pt'))):
                    raise OSError('injected payload failure')
                return real_save(payload, path)
            def publish_or_fail(path, payload):
                if failure == 'meta' and path.name.endswith('.meta.json'):
                    raise OSError('injected commit failure')
                return real_publish(path, payload)
            def write_or_fail(path, *args, **kwargs):
                if failure == 'handoff' and path.name.endswith('.handoff.json'):
                    raise OSError('injected handoff failure')
                return real_write(path, *args, **kwargs)
            with self.subTest(failure=failure), patch.object(torch, 'save', side_effect=save_or_fail), \
                    patch.object(cache, '_publish_first_pass_json', side_effect=publish_or_fail), \
                    patch.object(cache, '_safe_unlink', side_effect=AssertionError('deletion')), \
                    patch.object(Path, 'write_text', write_or_fail):
                self.save(2, handoff=dict(export_frames=3), low_carry=dict(samples=(torch.ones(1),)))
            # Old meta remains unchanged but the pending witness rejects it.
            self.assertEqual(old_meta, self.meta())
            self.assert_miss()
            self.save(3)
            self.assertIsNotNone(cache.load_first_pass_cache('node', self.seg, self.plan))

    def test_optional_artifacts_do_not_leak_across_generations(self):
        self.save(handoff=dict(export_frames=3), low_carry=dict(samples=(torch.ones(1),)))
        legacy = self.root / 'seg_0000.pre.low.pt'
        torch.save(dict(samples=(torch.ones(1),)), legacy)
        cache.save_first_pass_cache('node', self.seg, self.plan,
                                    av_latent=dict(samples=(torch.ones(1),)))
        result = cache.load_first_pass_cache('node', self.seg, self.plan)
        self.assertIsNone(result['frames'])
        self.assertIsNone(result['low_carry'])
        self.assertEqual(result['handoff'], {})
        self.assertTrue(legacy.is_file())

    def test_changed_or_missing_generation_artifact_rejected(self):
        self.save()
        paths = cache._first_pass_paths(self.root, 0, self.meta())
        paths['frames'].write_bytes(b'broken')
        self.assert_miss()
        self.save()
        paths = cache._first_pass_paths(self.root, 0, self.meta())
        paths['latent'].rename(paths['latent'].with_suffix('.retained'))
        self.assert_miss()

    def test_legacy_no_bridge_cache_still_matches(self):
        fp = cache.first_pass_cache_fingerprint(self.seg, self.plan)
        (self.root / 'seg_0000.pre.meta.json').write_text(json.dumps(fp))
        torch.save(dict(samples=(torch.ones(1),)), self.root / 'seg_0000.pre.av.pt')
        torch.save(torch.zeros((5, 2, 2, 3), dtype=torch.uint8), self.root / 'seg_0000.pre.pt')
        self.assertIsNotNone(cache.load_first_pass_cache('node', self.seg, self.plan))
        self.assertTrue(cache.inspect_first_pass_cache('node', self.plan)['matches'])

    def test_zero_alpha_is_identity_without_load_or_marker(self):
        positive = [[torch.ones((1, 2, bridge.HIDDEN_DIM)), {}]]
        self.plan.semantic_bridge = bridge.pack_semantic_bridge(adapter='missing.pt', alpha=0)
        with patch.object(bridge, 'load_semantic_student', side_effect=AssertionError('load')), \
                patch.object(bridge, 'resolve_semantic_bridge_path', side_effect=AssertionError('resolve')):
            out, note = bridge.apply_semantic_bridge(positive, self.plan)
            self.assertIs(out, positive)
            self.assertIsNone(note)
            self.assertEqual(positive[0][1], {})
            self.assertEqual(bridge.semantic_bridge_fingerprint(self.plan), {})

    def test_real_director_signature_changes_on_same_name_replacement(self):
        path = self.root / 'node_adapter.pt'
        path.write_bytes(b'first')
        widgets = dict(semantic_bridge_enable='on', semantic_bridge_adapter=str(path),
                       semantic_bridge_alpha='0.2', semantic_bridge_magnitude_match='false')
        with patch.object(cache, 'first_pass_cache_disk_signature', return_value='unchanged'):
            before = director_node.H3_D_NEO.IS_CHANGED(unique_id='node', **widgets)
            pack = bridge.pack_semantic_bridge(adapter=str(path), alpha='0.2', magnitude_match=False)
            expected = director_node._director_input_signature(
                widgets, 'unchanged', bridge.semantic_bridge_fingerprint(types.SimpleNamespace(semantic_bridge=pack)))
            self.assertEqual(before, expected)
            path.write_bytes(b'replacement')
            self.assertNotEqual(before, director_node.H3_D_NEO.IS_CHANGED(unique_id='node', **widgets))

    def test_real_director_disabled_and_zero_signatures_keep_old_hash(self):
        def old_hash(widgets):
            inputs = {key: director_node._director_is_changed_value(value)
                      for key, value in widgets.items() if key not in director_node._DIRECTOR_LINKED_INPUTS}
            return hashlib.sha256(json.dumps(dict(inputs=inputs, pre_cache='unchanged'),
                                            ensure_ascii=False, sort_keys=True,
                                            separators=(',', ':'), default=str).encode('utf-8')).hexdigest()
        cases = [dict(timeline_data='old', model=object()),
                 dict(semantic_bridge_enable='false', semantic_bridge_adapter='missing.pt'),
                 dict(semantic_bridge_enable='true', semantic_bridge_adapter='missing.pt',
                      semantic_bridge_alpha=0),
                 dict(semantic_bridge_enable='on', semantic_bridge_adapter='missing.pt',
                      semantic_bridge_alpha='0')]
        with patch.object(cache, 'first_pass_cache_disk_signature', return_value='unchanged'), \
                patch.object(bridge, 'resolve_semantic_bridge_path', side_effect=AssertionError('resolve')), \
                patch.object(bridge, 'weight_file_identity', side_effect=AssertionError('stat')):
            for widgets in cases:
                with self.subTest(widgets=widgets):
                    self.assertEqual(old_hash(widgets), director_node.H3_D_NEO.IS_CHANGED(**widgets))

    def test_legacy_no_meta_carry_and_preview_remain_readable(self):
        torch.save(dict(samples=(torch.ones(1),)), self.root / 'seg_0000.pre.av.pt')
        torch.save(torch.zeros((5, 2, 2, 3), dtype=torch.uint8), self.root / 'seg_0000.pre.pt')
        (self.root / 'seg_0000.pre.handoff.json').write_text(json.dumps(dict(export_frames=3)))
        for allow_stale in [False, True]:
            self.assertIsNotNone(cache.load_first_pass_av_latent(
                'node', self.seg, self.plan, allow_stale=allow_stale))
        self.assertEqual(cache.load_first_pass_frames_stale('node', self.seg, self.plan).shape[0], 3)
        self.assertIsNone(cache.load_first_pass_cache('node', self.seg, self.plan))
        self.assertFalse(cache.inspect_first_pass_cache('node', self.plan)['matches'])

    def test_pending_without_meta_never_exposes_legacy_payloads(self):
        torch.save(dict(samples=(torch.ones(1),)), self.root / 'seg_0000.pre.av.pt')
        torch.save(torch.zeros((5, 2, 2, 3), dtype=torch.uint8), self.root / 'seg_0000.pre.pt')
        # A failed first-ever generation leaves no meta but has a pending marker.
        with patch.object(torch, 'save', side_effect=OSError('first generation failure')):
            self.save(2)
        self.assert_miss()
        marker = self.root / 'seg_0000.pre.pending.json'
        for text in ['{}', 'not JSON']:
            marker.write_text(text)
            self.assert_miss()
            self.assertIsNone(cache.load_first_pass_av_latent('node', self.seg, self.plan))

    def test_concurrent_writers_commit_coherent_generations(self):
        a_pending = threading.Event()
        release_a = threading.Event()
        b_requested = threading.Event()
        real_publish = cache._publish_first_pass_json
        real_lock = cache._first_pass_write_lock
        commits = []

        def observe_lock(root, idx):
            lock = real_lock(root, idx)
            if threading.current_thread().name.endswith('_1'):
                acquired = lock.acquire(blocking=False)
                if acquired:
                    lock.release()
                self.assertFalse(acquired, 'B must block until A commits meta')
                b_requested.set()
            return lock

        def publish(path, data):
            writer = threading.current_thread().name.rsplit('_', 1)[-1]
            real_publish(path, data)
            kind = 'pending' if path.name.endswith('.pending.json') else 'meta'
            commits.append((writer, kind))
            if writer == '0' and kind == 'pending':
                a_pending.set()
                if not release_a.wait(5):
                    raise AssertionError('test release timed out')

        def write(value):
            self.save(value, handoff=dict(writer=value),
                      low_carry=dict(samples=(torch.tensor([float(value)]),)))

        with patch.object(cache, '_publish_first_pass_json', side_effect=publish), \
                patch.object(cache, '_first_pass_write_lock', side_effect=observe_lock), \
                ThreadPoolExecutor(max_workers=2, thread_name_prefix='cache_writer') as pool:
            first = pool.submit(write, 1)
            try:
                self.assertTrue(a_pending.wait(5))
                second = pool.submit(write, 2)
                self.assertTrue(b_requested.wait(5))
            finally:
                release_a.set()
            first.result(timeout=5)
            second.result(timeout=5)
        self.assertEqual(commits, [('0', 'pending'), ('0', 'meta'), ('1', 'pending'), ('1', 'meta')])
        result = cache.load_first_pass_cache('node', self.seg, self.plan)
        self.assertIsNotNone(result)
        self.assertEqual(result['av_latent']['samples'][0].item(), 2)
        self.assertEqual(result['low_carry']['samples'][0].item(), 2)
        self.assertEqual(result['handoff']['writer'], 2)
        self.assertTrue(torch.equal(result['frames'], torch.full((5, 2, 2, 3), 51 / 255)))
        self.assertTrue(cache.inspect_first_pass_cache('node', self.plan)['matches'])

    def test_write_locks_are_keyed_by_normalized_root_and_segment(self):
        lock = cache._first_pass_write_lock(self.root, 0)
        self.assertIs(lock, cache._first_pass_write_lock(self.root / '.', 0))
        self.assertIsNot(lock, cache._first_pass_write_lock(self.root, 1))
        self.assertIsNot(lock, cache._first_pass_write_lock(self.root / 'other', 0))

    def test_weight_identity_replacement_missing_and_json_stability(self):
        path = self.root / 'adapter.pt'
        path.write_bytes(b'first')
        self.plan.semantic_bridge = bridge.pack_semantic_bridge(adapter=str(path))
        before = cache.first_pass_cache_fingerprint(self.seg, self.plan)
        final_before = cache.segment_cache_fingerprint(self.seg, self.plan)
        self.assertEqual(json.loads(json.dumps(before)), before)
        self.assertEqual(before['sb_alpha'], .15)
        path.write_bytes(b'replacement')
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
        after = cache.first_pass_cache_fingerprint(self.seg, self.plan)
        self.assertNotEqual(before, after)
        self.assertNotEqual(final_before, cache.segment_cache_fingerprint(self.seg, self.plan))
        missing = bridge.weight_file_identity(str(self.root / 'absent.pt'))
        self.assertTrue(missing['missing'])
        self.assertEqual(missing['path'], str(self.root / 'absent.pt'))

    def test_student_cache_invalidates_and_stays_bounded(self):
        class TinyStudent(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))
        path = self.root / 'student.pt'
        torch.save(dict(weight=torch.ones(1)), path)
        bridge._MODEL_CACHE.clear()
        with patch.object(bridge, 'SemanticStudent', TinyStudent):
            one = bridge.load_semantic_student(str(path))
            self.assertIs(one, bridge.load_semantic_student(str(path)))
            torch.save(dict(weight=torch.full((1,), 2.)), path)
            stat = path.stat()
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
            two = bridge.load_semantic_student(str(path))
            self.assertIsNot(one, two)
            self.assertEqual(two.weight.item(), 2)
            self.assertEqual(len(bridge._MODEL_CACHE), 1)
        bridge._MODEL_CACHE.clear()


if __name__ == '__main__':
    unittest.main()
