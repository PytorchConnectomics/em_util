

def get_slurm(cmd, partition='weidf', num_cpu=1, num_gpu=0, memory=10000, time='1-00:00'):
    pref =f"""#!/bin/bash    
    #SBATCH -p {partition}
    #SBATCH -N 1 # number of nodes
    #SBATCH -n {num_cpu} # number of cores
    #SBATCH --mem {memory} # memory pool for all cores
    #SBATCH -t {time} # time (D-HH:MM)
    #SBATCH -o slurm.%N.%j.out # STDOUT
    #SBATCH -e slurm.%N.%j.err # STDERR
    """
    if num_gpu > 0:        
        pref += f'#SBATCH --gres=gpu:{num_gpu} # memory pool for all cores\n'
        
    pref += cmd
    return pref