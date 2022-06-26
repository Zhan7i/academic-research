```python
#pytorch镜像源
https://download.pytorch.org/whl/torch_stable.html
#创建新环境
conda create -n pytorch1.4 python=3.6

#安装cuda对应版本的CUDNN  

#安装pytorch镜像
#cuda版
pip install  torch==1.4.0 torchvision==0.5.0 -f https://download.pytorch.org/whl/torch_stable.html
#cpu版
pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

#安装 pytest-runner
pip install pytest-runner

pip install pillow==7.0.0 -i https://pypi.douban.com/simple
    
pip install torch-scatter -i https://pytorch-geometric.com/whl/torch-1.8.0%2Bcu102.html
#安装PYG：
进入pyg镜像网站 https://pytorch-geometric.com/whl/ 选择torch版本下载以下镜像
torch-scatter
torch-sparse
torch-cluster
torch-spline-conv
    
pip install torch-geometric -i https://pypi.douban.com/simple

#安装DGL
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple dgl
    
pip install dgl -f https://data.dgl.ai/wheels/repo.html for CPU.
pip install dgl-cuXX -f https://data.dgl.ai/wheels/repo.html for CUDA.
pip install --pre dgl -f https://data.dgl.ai/wheels-test/repo.html for CPU nightly builds.
pip install --pre dgl-cuXX -f https://data.dgl.ai/wheels-test/repo.html for CUDA nightly builds.
#pip更新安装包没有权限
pip install --user --upgrade  numpy   --user添加权限


```