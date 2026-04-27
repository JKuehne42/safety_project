#!/bin/bash
#SBATCH --job-name=JKuehne-SDT_RL        # Job name
#SBATCH --output=output.txt           # Standard output file
#SBATCH --error=error.txt             # Standard error file
#SBATCH --partition=partition_name    # Partition or queue name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Number of tasks per node
#SBATCH --cpus-per-task=1             # Number of CPU cores per task
#SBATCH --time=0:02:00                # Maximum runtime (D-HH:MM:SS)
#SBATCH --mail-type=END               # Send email at job completion
#SBATCH --mail-user=JKuehne@wpi.edu    # Email address for notifications
#SBATCH -A rbe577 # for RBE577 P3
#SBATCH -p academic # for RBE577 P3

#Load necessary modules (if needed)
#module load module_name

#Your job commands go here
#For example:
python train_cdt.py --reward_scale 0.001 --task 'OfflineCarCircle1Gymnasium-v0' --batch_size 1000 --update_steps 50_000

#Optionally, you can include cleanup commands here (e.g., after the job finishes)
#For example:
#rm some_temp_file.txt