from __future__ import annotations

import json
import tempfile
from pathlib import Path

import h5py
import numpy as np

from icil_jax_query_memory.data.adapter import prepared_tasks_to_host_batch
from icil_jax_query_memory.train.config import QueryMemoryMetaConfig
from icil_metaworld.data.metaworld_cache import MetaWorldEpisodeStore
from icil_metaworld.data.metaworld_task_builder import (
    MetaWorldICILConfig,
    MetaWorldQueryMemoryTaskBuilder,
    prepare_metaworld_query_memory_task_for_meta_step,
)


def _write_synthetic_cache(root: Path) -> None:
    rng = np.random.default_rng(0)
    index = {
        'version': 1,
        'cache_file': 'cache.h5',
        'tasks': {'button-press-v3': {'task_index': 0, 'instances': {'0': [], '1': []}}},
        'episodes': {},
        'obs_model_dim': 36,
        'action_dim': 4,
    }
    with h5py.File(root / 'cache.h5', 'w') as f:
        f.attrs['obs_model_dim'] = 36
        f.attrs['action_dim'] = 4
        episodes = f.create_group('episodes')
        eid = 0
        for instance_id in range(2):
            for _ in range(4):
                T = 32
                group = episodes.create_group(str(eid))
                group.create_dataset('obs_raw', data=rng.normal(size=(T, 39)).astype(np.float32))
                group.create_dataset('obs_model', data=rng.normal(size=(T, 36)).astype(np.float32))
                group.create_dataset('actions', data=rng.normal(size=(T, 4)).astype(np.float32))
                group.create_dataset('rewards', data=rng.normal(size=(T,)).astype(np.float32))
                group.create_dataset('success', data=np.ones((T,), dtype=np.uint8))
                group.create_dataset('done', data=np.zeros((T,), dtype=np.uint8))
                group.create_dataset('trunc', data=np.zeros((T,), dtype=np.uint8))
                group.create_dataset('terminated', data=np.zeros((T,), dtype=np.uint8))
                group.create_dataset('truncated', data=np.zeros((T,), dtype=np.uint8))
                meta = {
                    'episode_id': eid,
                    'task_name': 'button-press-v3',
                    'task_index': 0,
                    'task_instance_id': instance_id,
                    'env_id': 'button-press-v3',
                    'seed': eid,
                    'length': T,
                    'success_any': True,
                    'success_final': True,
                }
                index['episodes'][str(eid)] = meta
                index['tasks']['button-press-v3']['instances'][str(instance_id)].append(eid)
                eid += 1
    (root / 'index.json').write_text(json.dumps(index, indent=2), encoding='utf-8')


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_synthetic_cache(root)
        store = MetaWorldEpisodeStore(root, keep_open_per_worker=False)
        cfg = MetaWorldICILConfig(K=2, T_obs=2, H=8, stride=1)
        meta_cfg = QueryMemoryMetaConfig(inner_steps=2, num_queries_per_step=4, num_query_loss_samples=3)
        builder = MetaWorldQueryMemoryTaskBuilder(store, cfg=cfg, seed=0)
        rng = np.random.default_rng(0)
        tasks = []
        for _ in range(2):
            task = builder.build_task_spec(rng)
            assert task is not None
            tasks.append(
                prepare_metaworld_query_memory_task_for_meta_step(
                    task,
                    task_builder=builder,
                    cfg=meta_cfg,
                    rng=rng,
                )
            )
        host = prepared_tasks_to_host_batch(tasks, inner_steps=2)
        assert host['inner']['query_xyz'].shape == (2, 2, 4, 2, 1, 3)
        assert host['inner']['query_state'].shape == (2, 2, 4, 2, 36)
        assert host['query']['target_action'].shape == (2, 3, 8, 4)
        assert host['meta']['vidx'].tolist() == [0, 0]
        print('MetaWorld synthetic task-builder check passed.')
        store.close()


if __name__ == '__main__':
    main()
