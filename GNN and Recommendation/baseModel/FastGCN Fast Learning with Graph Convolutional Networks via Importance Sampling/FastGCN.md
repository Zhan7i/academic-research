> 论文标题：FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling
>
> 发表于：2018 ICLR
>
> 作者：Jie Chen, Tengfei Ma, Cao Xiao
>
> 代码：TensorFlow：https://github.com/matenure/FastGCN
> 			pytorch：https://github.com/gkunnan97/fastgcn_pytorch
>
> 论文地址：https://arxiv.org/pdf/1801.10247v1.pdf

## 摘要

- 图卷积网络（GCN）是一种有效的半监督学习图模型。这个模型最初被设计为通过训练和测试数据的存在来学习。
- 跨层的递归邻域扩展对使用大型密集图进行训练带来了时间和内存挑战。
- 为了放宽测试数据同时可用的要求，我们将图卷积解释为概率测量下嵌入函数的积分变换。
  - 这种解释允许使用Monte Carlo( 蒙特卡罗)方法来一致地估计积分，
- 因此文章中提出的批量训练方案——FastGCN
  - 通过重要性采样增强，FastGCN 不仅对训练有效，而且对推理也有很好的泛化性

## 结论

- FastGCN，GCN  模型的快速改进。它将转导训练推广到归纳方式，并解决了由邻域递归扩展引起的 GCN 的内存瓶颈问题
- 关键因素是重新制定损失和梯度的采样方案，通过嵌入函数积分变换形式的图卷积的另一种观点得到充分证明
- GraphSAGE 提出使用采样来限制邻域大小，但FastGCN比GCN和GraphSAGE 快几个数量级
- GCN 架构的简单性允许根据积分变换对图卷积进行自然解释。观点可以推广到许多基于一阶邻域的图模型，其中的例子包括适用于（网格）流形的  MoNet，以及许多消息传递神经网络
- 工作阐明了用于一致估计积分的基本蒙特卡洛成分

## 未来工作

- 当FastGCN推广到上述其他网络(MoNet、许多消息传递神经网络)时，研究方差是否减少以及如何改进估计器是一个可研究方向

## 介绍

## 模型架构

## 实验

- ### 数据集

  - Cora 、Pubmed、 Reddit

- ### baseline

  - FastGCN
  - GraphSAGE-GCN 
  - GCN (batched) 
  - GCN (original) 

- ### 超参数设置

- ### 评估指标

  - ACC、F1 