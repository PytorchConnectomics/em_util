from ..io import write_txt

def write_slurm(cmd, filename=None, job_id=0, job_num=1, partition='shared', num_cpu=1, num_gpu=0, memory=10000, time='1-00:00'):
    out = f"""#!/bin/bash    
#SBATCH -p {partition}
#SBATCH -N 1 # number of nodes
#SBATCH -n {num_cpu} # number of cores
#SBATCH --mem {memory} # memory pool for all cores
#SBATCH -t {time} # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
"""
    if num_gpu > 0:        
        out += f'#SBATCH --gres=gpu:{num_gpu} # memory pool for all cores\n\n'
        
    out += cmd % (job_id, job_num)
    if filename is None:
        return out
    else:
        write_txt(filename, out)

def write_slurm_all(cmd, filename, job_num=1, partition='shared', num_cpu=1, num_gpu=0, memory=10000, time='1-00:00'):
    for job_id in range(job_num):
        write_slurm(cmd, f'{filename}_{job_id}.sh', job_id, job_num, partition, num_cpu, num_gpu, memory, time)
    # print bash command to run submit all jobs
    print('for i in {0..%d};do sbatch %s_${i}.sh && sleep 1;done' % (job_num - 1, filename))
