# Examples

1. Generate and submit multiple slurm jobs
```
from em_util.cluster import write_slurm_all

cmd = 'python example.py %d %d'
write_slurm_all(cmd, 'test')
```
