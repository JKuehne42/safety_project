#!/bin/bash
#SBATCH --job-name=cdt_training
#SBATCH --mem=40G
#SBATCH --output=cdt_train_out.txt           # Standard output file
#SBATCH --error=cdt_train_error.txt             # Standard error file
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Number of tasks per node
#SBATCH --cpus-per-task=40             # Number of CPU cores per task
#SBATCH --gpus=2
#SBATCH --time=0-16:00:00                # Maximum runtime (D-HH:MM:SS)
#SBATCH --mail-type=END               # Send email at job completion
#SBATCH --mail-user=jkuehne@wpi.edu    # Email address for notifications
#SBATCH -A rbe577 # for RBE577 P3
#SBATCH -p academic # for RBE577 P3

module load apptainer

PROJECT_DIR=${SLURM_SUBMIT_DIR}

echo "Working from: ${PROJECT_DIR}"

# Clean any leftover from a previous run
rm -rf ${PROJECT_DIR}/mujoco_py_writable

# Copy mujoco_py into the project space
apptainer exec --userns \
  --bind ${PROJECT_DIR}:/work \
  ${PROJECT_DIR}/turing_cluster/slurm/box.sif \
  bash -c "cp -r /usr/local/lib/python3.8/dist-packages/mujoco_py /work/mujoco_py_writable"

export WANDB_API_KEY="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"

# Train the model inside the box, using config and demonstration data from outside the box
echo "Starting training..."
apptainer exec --userns --nv \
  --bind ${PROJECT_DIR}:/work \
  --bind ${PROJECT_DIR}/mujoco_py_writable:/usr/local/lib/python3.8/dist-packages/mujoco_py \
  ${PROJECT_DIR}/turing_cluster/slurm/box.sif \
  bash -c "yes | python3 /root/workspace/SDT_RL/stlrl/examples/train/train_cdt.py \
  --task 'OfflineAntJump-v0' \
  --use_cost_prefix True \
  --use_cost_suffix True \
  --reward_scale 0.001 \
  --logdir '/work/CDT_Logs' \
  --render false \
  --update_steps 50_000" 