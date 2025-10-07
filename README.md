# AMPLIFY: Actionless Motion Priors for Robot Learning from Videos

Jeremy A. Collins, Loránd Cheng, Kunal Aneja, Albert Wilcox, Benjamin Joffe, Animesh Garg

[![Website](https://img.shields.io/badge/Website-amplify--robotics.github.io-0a84ff?logo=google-chrome&logoColor=white&style=flat)](https://amplify-robotics.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2506.14198-b31b1b.svg?logo=arXiv&logoColor=white&style=flat)](https://arxiv.org/abs/2506.14198)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white&style=flat)](https://www.python.org)
![license](https://img.shields.io/github/license/pairlab/AMPLIFY?style=flat&cacheSeconds=1)
[![lucidrains/amplify-pytorch](https://img.shields.io/badge/lucidrains%2Famplify--pytorch-181717?logo=github&logoColor=white&style=flat)](https://github.com/lucidrains/amplify-pytorch)

This repo contains the official implementation of AMPLIFY: Actionless Motion Priors for Robot Learning from Videos.


![AMPLIFY Architecture](assets/architecture.png)


## Setup

#### 1. Create a new Conda environment and clone repo

``` bash
conda create -n amplify python=3.10 -y
conda activate amplify
git clone https://github.com/pairlab/AMPLIFY.git
cd AMPLIFY
```

#### 2. Clone and install third party packages
Note that LIBERO will require [CMake](https://cmake.org/download/) if you do not already have it installed.
See [Notes](#notes) if you run into issues related to `egl_probe`, or are seeing `ModuleNotFoundError`.

``` bash
# LIBERO benchmark
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt
pip install -e .
python benchmark_scripts/download_libero_datasets.py
cd ../AMPLIFY
mkdir -p preprocessed_data && ln -s "$(realpath LIBERO/libero/datasets)" preprocessed_data/libero_demos
```

```bash
# CoTracker
git clone https://github.com/facebookresearch/co-tracker.git
cd co-tracker
pip install -e .
cd ..
```

#### 3. Install AMPLIFY

``` bash
pip install -e .
```

#### 4. Download or preprocess point tracks
The motion tokenizer needs point tracks as input. You can download point tracks we've preprocessed from the LIBERO dataset [here](https://drive.google.com/drive/folders/15-zj-sSl16xuBcQvZM1vhNvT_rL2phGB?usp=sharing) and place them in `/preprocessed_data/<libero_10, libero_90, libero_object, libero_spatial, or libero_goal>`. Alternatively, you can preprocess a dataset yourself using:
```bash
python -m preprocessing.preprocess_libero mode=tracks suite='<libero_10, libero_90, libero_object, libero_spatial, or libero_goal>'
```

#### [optional] Preprocess dataset for text embeddings
To reduce redundant computation during training, you can preprocess the text embeddings for the dataset you've chosen to train on by setting `use_preprocessed_embs=True` in the `train_forward_dynamics.yaml`
and `train_inverse_dynamics.yaml` files. This should only take a few seconds for LIBERO. You can preprocess the text embeddings using:

``` bash
python -m preprocessing.preprocess_libero mode=text suite='<libero_10, libero_90, libero_object, libero_spatial, or libero_goal>'
```

## Training
### Training the motion tokenizer

``` bash
python train_motion_tokenizer.py run_name=<run name> train_datasets='[libero_10:traj0.9]' val_datasets='[libero_10:traj-0.1]'
```

### Training the forward dynamics model

``` bash
python train_forward_dynamics.py run_name=<run name> forward_dynamics.motion_tokenizer.checkpoint=<motion tokenizer checkpoint path> train_datasets='[libero_10:traj0.9]' val_datasets='[libero_10:traj-0.1]'
```

### Training the inverse dynamics model

``` bash
python train_inverse_dynamics.py run_name=<run name> motion_tokenizer_checkpoint=<motion tokenizer checkpoint path> forward_dynamics_checkpoint=<forward dynamics checkpoint path> train_datasets='[libero_10:action0.9]' val_datasets='[libero_10:action-0.1]'
```

The default inverse dynamics model uses a Gaussian action head. You can switch to a Diffusion or Flow-Matching action head by setting `type` to `diffusion` or `flow` in `train_inverse_dynamics.yaml`.

### Dataset CLI Format

Training scripts accept dataset specifications as a list of strings. Each string encodes the dataset name, the modality, and the fraction of demos per task to use:

- Syntax: `[<dataset>:<modality><fraction>, ...]`
- Modalities: `traj` (tracks) or `action`.
- Fractions: `0.0–1.0` specify the portion of demos per task. Negative values select from the end of the demo list (i.e., start at the last demo and go backward). For example, `traj-0.1` uses the last 10% of demos per task.
- Multiple datasets: pass a comma‑separated list to mix datasets, e.g. `train_datasets='[libero_10:traj0.9, libero_object:traj0.5]'`. You may need to remove spaces between datasets to avoid syntax errors.

### Eval on LIBERO

Download a pretrained AMPLIFY checkpoint from [here](https://drive.google.com/drive/folders/10xkV3nj6qNZG4EQvgAu5eMOtO3EaeJHv?usp=sharing) and evaluate using:
```bash
python eval_libero.py dataset='[libero_10]' run_name=<run name> amplify_checkpoint=<AMPLIFY checkpoint path>
```

If you trained the components yourself, first bundle them into a single AMPLIFY checkpoint:
```bash
python -m amplify.bundle_amplify --mt_ckpt <path/to/motion_tokenizer.pt> --fd_ckpt <path/to/forward_dynamics.pt> --id_ckpt <path/to/inverse_dynamics.pt> --name my_amplify_checkpoint   # writes to checkpoints/AMPLIFY/my_amplify_checkpoint.pt
```

## Replicating Paper Results

We provide scripts under `replicate_results/` that mirror the experiments in the paper. Each script assumes you have followed the setup steps above and that checkpoints are saved to the default `checkpoints/` directory.

Run any of the following scripts to replicate the corresponding results from the paper:

```bash
bash replicate_results/replicate_bc.sh
bash replicate_results/replicate_few_shot.sh
bash replicate_results/replicate_zero_shot.sh
```

The scripts produce bundled AMPLIFY checkpoints under `checkpoints/AMPLIFY/` and log evaluation summaries via `eval_libero.py`. Adjust Hydra overrides or `n_envs` flags as needed for your hardware.

## Custom Datasets

### Preprocessing your own dataset
You can preprocess arbitrary modalities (point tracks, images, videos, actions, text, depth, segmentation, etc.) using our pipeline. The easiest way to start is with the template at `preprocessing/preprocess_custom_dataset.py`. That file contains a scaffold for implementing your own preprocessor.

Once you’ve filled the TODOs, run your preprocessor via a small main or your own runner, similar to how we call the LIBERO preprocessor in `preprocessing/preprocess_libero.py`.

### Creating your own dataloader
We provide a small, modular interface for building dataset loaders. Start from `amplify/loaders/custom_dataset.py` and implement the TODOs. Please see `amplify/loaders/libero_dataset.py` for a complete reference implementation.

## Downloads
- [Ground Truth Point Tracks](https://drive.google.com/drive/folders/15-zj-sSl16xuBcQvZM1vhNvT_rL2phGB?usp=sharing) (place in `preprocessed_data/<dataset_name>`)
- [LIBERO Checkpoints (Gaussian, Diffusion, Flow-Matching heads)](https://drive.google.com/drive/folders/10xkV3nj6qNZG4EQvgAu5eMOtO3EaeJHv?usp=sharing) (place in `checkpoints/`)

## Checklist
- [x] Upload preprocessed LIBERO tracks
- [in progress] Upload LIBERO-10, 90, Object, Spatial, and Goal checkpoints
- [ ] Add Something-Something-V2 preprocessing and dataloading
- [ ] Add BridgeData V2 preprocessing and dataloading

## Notes

#### wandb
The code is setup to init a wandb run automatically with the relevant details in so far as you are already logged in. To login, use one of the following:
```bash
wandb login # Then enter API key when prompted
# OR
export WANDB_API_KEY=your_api_key_here
```
Then, set `use_wandb=True` to enable logging to wandb. You can also change the `wandb_group`, `wandb_project`, and `wandb_entity`.

#### Issues with `egl_probe`
During your LIBERO installation you may have ran into an issue where LIBERO depends on `egl_probe`, which requires `CMake` to be installed for compilation. You can fix this trivially by [installing CMake](https://cmake.org/download/), but if you don't want to do that, simply run the following command before installing the `requirements.txt` from LIBERO:
```bash
pip download egl-probe --no-binary :all: && tar -xzf egl_probe-*.tar.gz && cp -r egl_probe-*/egl_probe $(python -c "import site; print(site.getsitepackages()[0])") && echo "egl-probe" > $(python -c "import site; print(site.getsitepackages()[0])")/egl_probe-1.0.2.egg-info && rm -rf egl_probe-*
```

#### `ModuleNotFoundError`: No module named 'libero'
To solve this, simply run `pip install -e . --config-settings editable_mode=compat` instead of `pip install -e .` in the LIBERO directory.

#### Issues with LIBERO dependencies
If you run into issues with LIBERO dependencies, try the following command:
`conda install -c conda-forge "transformers=4.21.1" "tokenizers=0.11.4" -y`

#### MacOS
The codebase is setup to support MacOS and use `mps` as is, but running the training loop for the forward dynamics and inverse dynamics models will require you to set
`PYTORCH_ENABLE_MPS_FALLBACK=1` because a few torch operations aren't yet supported on `mps`.

## Acknowledgements
We thank Namra Patel, Ayush Agarwal, and Shitij Govil for helping with the development of the open-source version of AMPLIFY.

## Citation
If you find this work useful, please use the following citation:
```
@misc{collins2025amplify,
    title={AMPLIFY: Actionless Motion Priors for Robot Learning from Videos},
    author={Jeremy A. Collins and Loránd Cheng and Kunal Aneja and Albert Wilcox and Benjamin Joffe and Animesh Garg},
    year={2025},
    eprint={2506.14198},
    archivePrefix={arXiv},
    primaryClass={cs.RO},
    url={https://arxiv.org/abs/2506.14198}}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.