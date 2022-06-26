```
import logging
import subprocess

from setuptools import setup

import torch

cuda_v = f"cu{torch.version.cuda.replace('.', '')}"
torch_v = torch.__version__.split('.')
torch_v = '.'.join(torch_v[:-1] + ['0'])


def system(command: str):
    output = subprocess.check_output(command, shell=True)
    logging.info(output)


system(f'pip install scipy')
system(f'pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-{torch_v}+{cuda_v}.html')
system(f'pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-{torch_v}+{cuda_v}.html')
system(f'pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-{torch_v}+{cuda_v}.html')
system(f'pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-{torch_v}+{cuda_v}.html')
system(f'pip install torch-geometric')
```