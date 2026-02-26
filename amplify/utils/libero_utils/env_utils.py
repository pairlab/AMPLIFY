import logging
import multiprocessing
import os
import random
from functools import partial

import h5py
import numpy as np
import torch

from LIBERO.libero.libero import benchmark
from LIBERO.libero.libero.envs import OffScreenRenderEnv
from amplify.utils.libero_utils.wrappers import (
    EnvStateWrapper,
    FourDOFWrapper,
    LiberoImageUpsideDownWrapper,
    LiberoObservationWrapper,
    LiberoResetWrapper,
    LiberoSuccessWrapper,
    StackDummyVectorEnv,
    StackSubprocVectorEnv,
)
from amplify.utils.train import get_root_dir

# Required for vectorized envs
if multiprocessing.get_start_method(allow_none=True) != "spawn":  
    multiprocessing.set_start_method("spawn", force=True)


def get_task_emb(task_suite, task_name, dataset_path=None):
    """
    Returns the task embedding for a given task.

    Returns:
        task_emb: torch.Tensor, task embedding
    """
    demo_root: str
    if dataset_path is None:
        root_dir = get_root_dir()
        demo_root = os.path.join(root_dir, "LIBERO/libero/datasets")

        if not os.path.exists(demo_root):
            raise ValueError(f"LIBERO dataset not found at {demo_root}, please set dataset_path in your"
                             " respective config or ensure datasets are in the standard location:"
                             " /LIBERO/libero/datasets")
    else:
        demo_root = os.path.expanduser(dataset_path)
        if not os.path.exists(demo_root):
            raise ValueError(f"Provided dataset_path does not exist: {demo_root}")

    task_suite_path = os.path.join(demo_root, task_suite)
    task_file = f"{task_name}_demo.hdf5"
    task_file_path = os.path.join(task_suite_path, task_file)
    with h5py.File(task_file_path, 'r') as f:
        if 'text_emb' in f:
            task_emb = torch.tensor(f['text_emb'][()])
        else:
            # Fallback to the preprocessed text embeddings if not in the dataset
            root_dir = get_root_dir()
            fallback_path = os.path.join(root_dir, 'preprocessed_data', task_suite, 'text', f'{task_name}.hdf5')
            if not os.path.exists(fallback_path):
                raise ValueError(f"text_emb not found in {task_file_path} and fallback text embedding missing at {fallback_path}. Please run: python -m preprocessing.preprocess_libero mode=text suite='{task_suite}'")
            with h5py.File(fallback_path, 'r') as fallback_f:
                task_emb = torch.tensor(fallback_f['text_emb'][()])
    
    return task_emb


def build_libero_env(task_suite, task_no, img_size, dataset_path, action_dim=7, n_envs=1, use_depth=False, segmentation_level=None, flip_image=True, libero_path="LIBERO/libero/libero", vecenv=True,  **kwargs):
    """
    Builds a libero environment.
    
    Returns:
        env: (vectorized) environment
        task: str, task description
        task_emb: torch.Tensor, task embedding
    """
    assert action_dim in [4, 7], "Only 4 or 7 action dimensions are supported"
    logging.info('Building LIBERO environment...')
    # initialize a benchmark
    benchmark_dict = benchmark.get_benchmark_dict()
    benchmark_instance = benchmark_dict[task_suite]()
    task = benchmark_instance.get_task(task_no)
    
    # Task embedding
    task_emb = get_task_emb(task_suite, task.name, dataset_path)

    env_args = {
        "bddl_file_name": os.path.join(libero_path, 'bddl_files', task.problem_folder, task.bddl_file),
        "camera_heights": img_size,
        "camera_widths": img_size,
        "ignore_done": True,
        "camera_depths": use_depth,
        "camera_segmentations": segmentation_level, # None, 'instance', 'class', 'element'
        "camera_names": ["agentview", "robot0_eye_in_hand"] # ['frontview', 'birdview', 'agentview', 'sideview', 'galleryview', 'robot0_robotview', 'robot0_eye_in_hand'],
    }
    env_args.update(kwargs)
    
    def env_func(init_state_no):
        env = OffScreenRenderEnv(**env_args)
        env = LiberoResetWrapper(
            env, 
            init_states=benchmark_instance.get_task_init_states(task_no),
            init_state_no=init_state_no
        )
        env = EnvStateWrapper(env)
        if action_dim == 4:
            env = FourDOFWrapper(env)
        if flip_image:
            env = LiberoImageUpsideDownWrapper(env)
        env = LiberoSuccessWrapper(env)
        env = LiberoObservationWrapper(env, masks=None, cameras=env_args["camera_names"])
        env.seed(init_state_no)
        return env
    
    if vecenv:
        init_state_no = random.sample(range(10), n_envs)
        if n_envs == 1:
            env = StackDummyVectorEnv([partial(env_func, init_state_no[0])])
        else:
            env = StackSubprocVectorEnv([partial(env_func, init_state_no[i]) for i in range(n_envs)])
    else:
        assert n_envs == 1, "Non-vectorized environment can only have one environment"
        env = env_func()

    return env, task.language, task_emb