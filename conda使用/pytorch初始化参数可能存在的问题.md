Pytroch1.8以下版本  不存在torch.nn.parameter的UninitializedParameter属性

由于不需要用到UninitializedParameter，将torch的代码修改一下即可

将 pytorch\Lib\site-packages\torch_geometric\nn\dense\linear.py文件下的 

根据PyG官方文件https://github.com/pyg-team/pytorch_geometric/commit/973d17d888b6b3139dd516baa31d9f3ccac2898a

先将

```python
    if isinstance(self.weight, nn.parameter.UninitializedParameter):
        pass
    #改为
    if self.in_channels <= 0:
    pass
```

再将

```python
    if isinstance(self.weight, nn.parameter.UninitializedParameter):
        pass
    elif self.bias is None:
        pass
    
    #改为
    if self.bias is None or self.in_channels <= 0:
        pass
```



