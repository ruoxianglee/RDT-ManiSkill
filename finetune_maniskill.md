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
   
Note: Here RDT and ManiSkill are installed into the same conda environment.

```bash
conda activate rdt
cd ManiSkill
pip install -e .
```

4. Configure Vulkan
   
Follow the [ManiSkill documentation](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#vulkan) to properly set up Vulkan. If libvulkan1 and vulkan-utils cannot be located, you can refer to following commands:
```bash
sudo apt-get install libvulkan-dev
sudo apt-get install vulkan-tools
vulkaninfo
```

5. Obtain Model Weights

Download the fine-tuned model weights from [Hugging Face repository](https://huggingface.co/robotics-diffusion-transformer/maniskill-model/tree/main/rdt) to the RDT-ManiSkill directory.


6. Prepare Nvidia cutlass
  
Download https://github.com/NVIDIA/cutlass.git and modify Line 12 in `finetune_maniskill.sh` to `export CUTLASS_PATH="/path/to/cutlass"`


## Prepare Demonstration
Generate 1,000 trajectories for GraspCup-v1 task through motion planning in ManiSkill.

Note: 
 - modify the arguments `record-dir` and `traj-path` accordingly.
 - After replaying trajectories, remember to remove the original trajectory_cpu.h5 and trajectory_cpu.json from directory `"/xxxxx/demos/GraspCup-v1/motionplanning/"`. Just keep the newly generated hdf5 and json files in this directory.

```bash
conda activate rdt

cd ManiSkill

python mani_skill/examples/motionplanning/panda/run.py -e "GraspCup-v1" --record-dir "/xxxxx/demos/" --traj-name="trajectory_cpu" -n 1000 --sim-backend "cpu" --only-count-success

python -m mani_skill.trajectory.replay_trajectory --traj-path "/xxxxx/demos/GraspCup-v1/motionplanning/trajectory_cpu.h5" --use-first-env-state --sim-backend cpu -c pd_joint_pos -o rgb --save-traj --num-procs 16

```


## Finetune RDT with ManiSkill Data
Configure demo path in Line 43 in `data/hdf5_vla_dataset.py`: `self.data_dir = "/xxxxx/demos/"`

```
conda activate rdt

bash finetune_maniskill.sh
```

## Evaluate RDT with ManiSkill
```bash
conda activate rdt 
cd eval_sim
python -m eval_sim.eval_rdt_maniskill --env-id GraspCup-v1 --pretrained_path PATH_TO_PRETRAINED_MODEL
```

## Issues Solved
### Issue 1
```ImportError: cannot import name 'cached_download' from 'huggingface_hub' (/root/miniconda3/envs/rdt/lib/python3.10/site-packages/huggingface_hub/__init__.py)```

Solution:
```
pip uninstall -y huggingface_hub
pip install huggingface_hub==0.23.5
```

### Issue 2
If one single GPU is used, you may encounter this issue:
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
In Line 354 of `train/train.py`, change `rdt.module.load_state_dict(checkpoint["module"])` to `rdt.load_state_dict(checkpoint["module"])`
