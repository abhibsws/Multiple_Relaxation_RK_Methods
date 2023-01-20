#!/bin/bash 


#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH -J MyJob_3_sol_2_inv
#SBATCH -o MyJob_3_sol_2_inv.%J.out
#SBATCH -e MyJob_3_sol_2_inv.%J.err
#SBATCH --time=100:00:00


module load anaconda3

start=`date +%s.%N`

python KdV_3_soliton_2_inv.py


end=`date +%s.%N`

runtime=$( echo "$end - $start" | bc -l )
echo $runtime




