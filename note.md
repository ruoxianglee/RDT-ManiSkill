## Installation

1. Clone this repo and install prerequisites:

    ```bash
    # Clone this repo
    git clone git@github.com:ruoxianglee/RDT-ManiSkill.git
    cd RDT-ManiSkill
    
    # Create a Conda environment
    conda create -n rdt python=3.10.0
    conda activate rdt
    
    # Install pytorch
    # Look up https://pytorch.org/get-started/previous-versions/ with your cuda version for a correct command
    pip install torch==2.1.0 torchvision==0.16.0  --index-url https://download.pytorch.org/whl/cu121
    
    # Install packaging
    pip install packaging==24.0
    
    # Install flash-attn
    pip install flash-attn --no-build-isolation
    
    # Install other prequisites
    pip install -r requirements.txt
    ```

2. Download off-the-shelf multi-modal encoders:

   You can download the encoders from the following links:

   - `t5-v1_1-xxl`: [link](https://huggingface.co/google/t5-v1_1-xxl/tree/main)🤗
   - `siglip`: [link](https://huggingface.co/google/siglip-so400m-patch14-384)🤗

   And link the encoders to the repo directory:

   ```bash
   # Under the root directory of this repo
   mkdir -p google
   
   # Link the downloaded encoders to this repo
   ln -s /path/to/t5-v1_1-xxl google/t5-v1_1-xxl
   ln -s /path/to/siglip-so400m-patch14-384 google/siglip-so400m-patch14-384
   ```

3. Install ManiSkill

```bash
conda activate rdt
cd ManiSkill
pip install -e .
```

4. Configure Vulkan
Follow the [ManiSkill documentation](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#vulkan) to properly set up Vulkan.

5. Obtain Model Weights
Download the fine-tuned model weights from [Hugging Face repository](https://huggingface.co/robotics-diffusion-transformer/maniskill-model/tree/main/rdt) to the RDT-ManiSkill directory.


# Demonstration
Generate 1,000 trajectories for GraspCup-v1 task through motion planning in ManiSkill.

```bash
cd ManiSkill
python mani_skill/examples/motionplanning/panda/run.py -e "GraspCup-v1" --record-dir "/root/autodl-tmp/demos/" --traj-name="trajectory_cpu" -n 1000 --sim-backend "cpu" --only-count-success

python -m mani_skill.trajectory.replay_trajectory --traj-path "/root/autodl-tmp/demos/GraspCup-v1/motionplanning/trajectory_cpu.h5" --use-first-env-state --sim-backend cpu -c pd_joint_pos -o rgb --save-traj --num-procs 16

```
Note: modify the `record-dir` and `traj-path` accordingly.

# Model
Vision: siglip-so400m-patch14-384
Language: t5-v1_1-xxl

# Controller
The initial action mode of these trajectories is absolute joint position control and we subsequently converted them into delta end-effector pose control to align with the pre-training action space of OpenVLA and Octo.

Consequently, we finetuned OpenVLA and Octo using the delta end-effector pose data

For RDT and Diffusion-Policy we leverage joint position control data for training which is aligned with our pre-training stage as well.

- arm_pd_joint_pos

# Finetune RDT with ManiSkill Data
download https://github.com/NVIDIA/cutlass.git
change
`export CUTLASS_PATH="/root/cutlass"`

Change 
`self.data_dir = "/root/autodl-tmp/demos/"`
`self.tasks = ['GraspCup-v1']`

To fine-tune RDT with Maniskill data, first download the Maniskill data from [here](https://huggingface.co/robotics-diffusion-transformer/maniskill-model) and extract it to `data/datasets/rdt-ft-data`. Then copy the code in `data/hdf5_maniskill_dataset.py` to `data/hdf5_vla_dataset.py` and run the following script:

add 
`"GraspCup-v1": "Grasp a brown cup from the horizontal direction and move it to a target goal position."` 
to self.task2lang dictionary

## Modify finetune_maniskill.sh
<!-- - --pretrained_model_name_or_path="./mp_rank_00_model_states.pt" -->
- add `--precomp_lang_embed` after `--image_aug`

```
bash finetune_maniskill.sh
```

# Evaluate RDT with ManiSkill
```bash
conda activate rdt 
cd eval_sim
python -m eval_sim.eval_rdt_maniskill --env-id GraspCup-v1 \
--pretrained_path PATH_TO_PRETRAINED_MODEL
```

# Questions
2. How many steps are recommended for fine-tuning RDT?
Regardless of the batch size you select, it is recommended to train for at least 150K steps to achieve optimal results.

3. What to do if t5-xxL is too large to store in GPU memory?
Do not load T5-XXL in your GPU memory when training. Pre-compute language embeddings in advance.
Set OFFLOAD_DIR to enable CPU offloading in scripts/encode_lang_batch.py and scripts/encode_lang.py.
Use smaller versions of t5 like t5-base instead of t5-xxL.

# Issues Solved
## Issue 1
```ImportError: cannot import name 'cached_download' from 'huggingface_hub' (/root/miniconda3/envs/rdt/lib/python3.10/site-packages/huggingface_hub/__init__.py)```

Solution
```
pip uninstall -y huggingface_hub
pip install huggingface_hub==0.23.5
```

## Issue 2
`No such file or directory: "Grasp a brown cup from the horizontal direction and move it to a target goal position."`

Solution:
change Line 356 in `train/dataset.py`
```bash
data_dict["lang_embed"] = torch.load(f'text_embed_GraspCup-v1.pt') \
                        if random.random() > self.cond_mask_prob else self.empty_lang_embed
```

## Issue 3
`RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method`s

Solution: set `use_hdf5=True` (`load_from_hdf5` argument in finetune_maniskill.sh seems not work)
```
    train_dataset = VLAConsumerDataset(
        config=config["dataset"],
        tokenizer=tokenizer,
        image_processor=image_processor,
        num_cameras=config["common"]["num_cameras"],
        img_history_size=config["common"]["img_history_size"],
        dataset_type=args.dataset_type,
        image_aug=args.image_aug,
        cond_mask_prob=args.cond_mask_prob,
        cam_ext_mask_prob=args.cam_ext_mask_prob,
        state_noise_snr=args.state_noise_snr,
        use_hdf5=True,
        use_precomp_lang_embed=args.precomp_lang_embed,
    )
    sample_dataset = VLAConsumerDataset(
        config=config["dataset"],
        tokenizer=tokenizer,
        image_processor=image_processor,
        num_cameras=config["common"]["num_cameras"],
        img_history_size=config["common"]["img_history_size"],
        dataset_type=args.dataset_type,
        image_aug=False,
        cond_mask_prob=0,
        cam_ext_mask_prob=-1,
        state_noise_snr=None,
        use_hdf5=True,
        use_precomp_lang_embed=args.precomp_lang_embed,
    )        
```

Solution:
add to Line 275 in `data/producer.py`
```bash
import torch.multiprocessing as mp
mp.set_start_method('spawn')
```


## Issue 4
Keyword error `agilex`
```
        # dataset_names_cfg = 'configs/pretrain_datasets.json' \
            # if dataset_type == 'pretrain' else 'configs/finetune_datasets.json'
        dataset_names_cfg = 'configs/finetune_datasets.json'
```
## Issue 5
### Case 1
`pretrained_model_name_or_path="robotics-diffusion-transformer/rdt-1b"`

```bash
01/15/2025 21:46:27 - INFO - __main__ - Constructing model from pretrained checkpoint.
Traceback (most recent call last):
  File "/root/RoboticsDiffusionTransformer/main.py", line 300, in <module>
    train(args, logger)
  File "/root/RoboticsDiffusionTransformer/train/train.py", line 150, in train
    rdt = RDTRunner.from_pretrained(args.pretrained_model_name_or_path)
  File "/root/miniconda3/envs/rdt/lib/python3.10/site-packages/huggingface_hub/utils/_validators.py", line 114, in _inner_fn
    return fn(*args, **kwargs)
  File "/root/miniconda3/envs/rdt/lib/python3.10/site-packages/huggingface_hub/hub_mixin.py", line 569, in from_pretrained
    instance = cls._from_pretrained(
  File "/root/RoboticsDiffusionTransformer/models/hub_mixin.py", line 41, in _from_pretrained
    model = cls(**model_kwargs)
TypeError: RDTRunner.__init__() missing 8 required keyword-only arguments: 'action_dim', 'pred_horizon', 'config', 'lang_token_dim', 'img_token_dim', 'state_token_dim', 'max_lang_cond_len', and 'img_cond_len'
```

### Case 2
`pretrained_model_name_or_path="./mp_rank_00_model_states.pt"`

```bash
01/15/2025 23:03:10 - INFO - __main__ - Loading from a pretrained checkpoint.
Traceback (most recent call last):
  File "/root/RoboticsDiffusionTransformer/main.py", line 300, in <module>
    train(args, logger)
  File "/root/RoboticsDiffusionTransformer/train/train.py", line 354, in train
    rdt.module.load_state_dict(checkpoint["module"])
  File "/root/miniconda3/envs/rdt/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1931, in __getattr__
    raise AttributeError(
AttributeError: 'RDTRunner' object has no attribute 'module'
```

Solution: 
```
rdt.load_state_dict(checkpoint["module"])
```