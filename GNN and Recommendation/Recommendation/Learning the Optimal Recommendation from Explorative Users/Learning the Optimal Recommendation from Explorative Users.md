> 论文标题：Learning the Optimal Recommendation from Explorative Users
>
> 发表于：2022 AAAI
>
> 作者：Fan Yao, Chuanhao Li, Denis Nekipelov
>
> 代码：
>
> 论文地址：https://arxiv.org/pdf/2110.03068.pdf

## 摘要

- 提出了一个新的问题设置来研究推荐系统和用户之间的顺序交互。
- 描绘了一个更现实的用户行为模型在该模型下用户：
  - 如果推荐明显比其他推荐更差，则拒绝推荐；
  - 根据用户接受的推荐的奖励更新用户的效用估计； 
  - 从系统中扣留已实现的奖励。
- 在 K-armed  bandit 框架中制定了系统与这种探索性用户之间的交互，并研究了在系统端学习最优推荐的问题
- 该系统可以在 O(1/δ) 相互作用中以至少 1 - δ  的概率识别最佳臂，文章证明这是严谨的。
- 文章发现对比了具有固定置信度的最佳手臂识别问题的结果，其中最佳手臂可以在 O(log(1/δ)) 交互中以 1 - δ  的概率被识别。
  - 这一差距说明了当系统从探索性用户对其推荐的揭示偏好而不是从已实现的奖励中学习时，系统必须支付不可避免的成本。

## 结论

- 提出了一个新的学习问题，即从探索性用户的揭示偏好中识别最佳手臂
- 通过对用户的学习行为进行建模来放松用户无所不知的强假设，并研究系统端的学习问题，仅根据揭示的用户反馈来推断用户的真实偏好。
- 开发了具有完整分析的最佳手臂识别算法证明了在这种具有挑战性的设置下仍然可以进行有效的系统学习，并且还揭示了新问题设置引入的内在难度

## 未来工作

- N1的最优选择
- 针对系统和单个用户的问题制定和解决方案也有助于从大量用户中学习

## 介绍

## 模型架构

## 实验

- ### 数据集

- ### baseline

- ### 超参数设置

- ### 评估指标