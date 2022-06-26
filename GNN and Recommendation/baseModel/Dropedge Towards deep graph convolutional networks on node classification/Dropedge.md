> 论文标题：DropEdge: Towards Deep Graph Convolutional Networks on Node Classification
>
> 发表于：2020 ICLR
>
> 作者：Yu Rong, Wenbing Huang, Tingyang Xu
>
> 代码：https://github.com/DropEdge/DropEdge
>
> 论文地址：https://arxiv.org/pdf/1907.10903v4.pdf

## 摘要

- 过度拟合和过度平滑是开发用于节点分类的深度图卷积网络 (GCN) 的两个主要障碍，
  - 过拟合削弱了对小数据集的泛化能力
  - 随着网络深度的增加，过度平滑通过将输出表示与输入特征隔离开来阻碍模型训练
- 提出了 DropEdge，缓解这两个问题。 核心是在每个训练时期从输入图中随机删除一定数量的边，
  - 就像一个数据增强器和一个消息传递缩减器。
  - 要么降低了过度平滑的收敛速度，要么减轻了由它引起的信息丢失
  - 通过 DropEdge，我们实际上是在生成原始图的不同随机变形副本；
    - 因此，我们增加了输入数据的随机性和多样性，从而更好地防止过拟合
  - 在 GCN 中，相邻节点之间的消息传递是沿着边缘路径进行的。
    - 移除某些边会使节点连接更加稀疏，因此当 GCN 非常深入时，在一定程度上避免了过度平滑
- 可以配备许多其他主干模型（例如 GCN、ResGCN、GraphSAGE 和 JKNet）以增强性能

## 结论

- 提出了 DropEdge，这是一种促进深度图卷积网络 (GCN) 发展的新颖而有效的技术。
- 通过随机丢弃一定比率的边，DropEdge在输入数据中包含更多的多样性以防止过拟合，并减少图卷积中的消息传递以缓解过度平滑

## 未来工作

- 更深入地探索深度 GCN 以实现更广泛的潜在应用，

## 介绍

- 节点分类的典型 GCN（Kipf & Welling，2017）通常很浅
- 受深度 CNN 在图像分类上的成功启发，已经提出了一些尝试来探索如何为节点分类构建深度 GCN
  - 它们都没有提供足够富有表现力的架构。
- 本文的动机是分析阻碍更深层次的 GCN 表现良好的因素，并开发解决这些问题的方法。

## 模型架构

## 实验

- ### 数据集

  - Cora、Citeseer、Pubmed  、Reddit 

- ### baseline

  - GCN、
  - FastGCN、
  - AS-GCN 
  - GraphSAGE

- ### 超参数设置

  - 基于五个主干网络：GCN (Kipf & Welling, 2017)、ResGCN (He et al., 2016; Li et al.,  2019)、JKNet (Xu et al., 2018a)、IncepGCN5 和 GraphSAGE (Hamilton et al., 2017）进行实验

- ### 评估指标