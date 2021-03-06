---
typora-root-url: Graph Learning Approaches to Recommender Systems A Review
---

## Graph Learning Approaches to Recommender Systems: A Review  

> #### 论文阅读笔记

## 0.摘要

与传统的RS(包括基于内容的过滤和协作过滤)不同，GLRS建立在简单或复杂的图上，其中各种对象(例如，用户、项目和属性)被显式或隐式地连接。随着图学习的快速发展，探索和利用图中的同质或异质关系是构建高级RS的一个很有前途的方向

## 1.介绍

动机：为什么要为RS进行图学习？

- RS中的大部分数据本质上是一个图结构
- 图学习有助于构建可辩解的RS  -> 推荐的可解释性备受关注 
  - 得益于GL对关系的因果推理能力，GLRS可以很容易地根据RS中涉及的不同对象之间的推断关系来支持对推荐结果的解释

形式化:图形学习如何帮助RS?

- 给定一个数据集,考虑图G = {V， E}的对象
  - 例如,用户,项目,视为节点设置成V
  - 例如,购买历史,社会关系,它们之间的关系,视为边设置成E  .
  - 那么,GLRS使用图G作为输入来生成相应的推荐结果R,R = argmaxf(G) 
  - 根据具体的数据和推荐场景，图G可以是同质的或异构的，静态的或动态的，而推荐可以是多种形式，例如，预测评级或对项目的排名。
  - 具体的优化目标也不同:根据图的拓扑结构，优化目标可以是最大的选择效用，也可以是节点之间形成链接的最大概率。

## 2.数据特征和挑战

RS可以考虑许多不同的对象，如用户、物品、属性、上下文等，几乎所有的对象都是通过某种类型的链接相互连接的

| GLRS案例         | 图实例                                  | 推荐任务                          | 方法实现               |
| ---------------- | --------------------------------------- | --------------------------------- | :--------------------- |
| 树图上的RS       | 项目层次图                              | 评级预测                          | 知识图谱               |
| 单部图上的RS     | 用户的社交图、项目的会话图              | 朋友推荐、下一项推荐              | 随机游走、图神经网络   |
| 二部图上的RS     | 基于评级、购买、电子评级的用户项交互图  | Top-n项目推荐                     | 随机游走               |
| 属性图上的RS     | 用户属性图、项目属性图                  | 朋友推荐、社交推荐                | 图表示学习、图神经网络 |
| 复杂异构图上的RS | 结合社会关系或项目特征的用户-项目交互图 | 社交推荐、评级预测、Top-n项目推荐 | 知识图谱、图神经网络   |
| 多源异构图上的RS | 有属性的多重异构图                      | 评级预测、Top-n项目推荐           | 图表示学习             |

1. 树图上的RS

   - 亚马逊网站上出售的商品首先被分成不同的类别(例如电子产品和体育产品)，然后每个类别又分成几个子类别(例如电子产品中的无线产品和电脑)，而每个子类别又包括多个项目(例如iPhone  XR，华为手表GT无线产品)。
     - 这种层次图本质上揭示了物品背后丰富的关系
     - 来自不同但密切相关类别的两件物品(例如iPhone  XR和配件)很可能在功能上存在互补关系。
     - 这种项目之间的层次关系可以极大地提高推荐的性能，例如避免重复向用户推荐同一子类别中类似的项目，从而使推荐列表多样化。

2. 单部图上的RS

   - 在RS中，至少可以定义两个齐次单部图，一个是用户图，另一个是项目图。
     - 具体而言，一方面，用户之间的线上或线下社交关系构成了用户的同质社交图谱
     - 另一方面，同一购物篮或会话中的商品的共现将交易数据中的所有商品连接在一起，从而导致商品的同质会话图
   - 挑战：
     - 社交图谱中的用户通常会通过商品偏好和购物行为相互影响。在提出建议时，有必要考虑社会影响。这种社会影响通常会在社交图谱上传播，因此应该会对推荐结果产生级联影响。因此，如何了解用户间的社会影响及其在用户图上的传播成为一个具体的挑战。
     - 商品间的共现关系通常不仅反映了商品之间的某种潜在关系，如商品功能的互补关系或竞争关系，而且还揭示了用户的某些购物模式。因此，在会话图中合并项目之间的共现关系有助于生成更准确的推荐。因此如何从本质上捕获项目图上的项目间关系，并适当地利用它们来提高推荐的准确性 成为了另一个具体的挑战。

3. 关于二部图的RS

   - 连接用户和商品的交互(如点击、购买)是一个RS的核心信息，所有这些都自然地形成了一个用户-商品二部图。

     - 在二部图中建模的交互可以是同构的

     - 向特定用户推荐商品的任务可以看作是用户-商品二部图上的链接预测，即给定图中已知边，预测可能存在的未知边
     - 挑战: 
       - 如何学习具有同质交互的图上复杂的用户-项目交互，以及这些交互之间的全面关系，以便进行推荐。
       - 如何在一个具有异构交互的图表上捕捉不同类型的交互之间的影响，如点击对购买的影响，以提供更丰富的信息，以生成更准确的推荐

4. 属性图上的RS

   - 异构用户图、产品图或用户交互图
     - 至少有两种不同类型的边分别表示不同的关系:一种表示用户之间的社会关系，另一种表示用户具有一定的属性值(如男性)，在图中共享相同属性值的用户是间接连接的
     - 用户的属性值和建立在其上的间接连接对于提高好友和社交推荐(结合社交关系推荐项目)都具有重要意义。通过提供额外信息来更好地捕捉个性化用户偏好和用户间的影响。
     - 挑战：
       - 如何在异构用户图上建模不同类型的关系以及它们之间的相互影响，然后如何适当地将它们集成到推荐任务中
       - 项和项-属性值关系之间的共现关系形成一个异构项图。这两种关系对于理解项目的分布、出现模式和内在性质都很重要，如何在异构项图上有效地建模这种异构关系，从而提高推荐性能成为该分支面临的另一个挑战

5. 复杂异构图上的RS

   - 为了解决用户-商品交互数据的稀疏性问题，更好地理解用户偏好和商品特征，通常会将社会关系或商品特征等辅助信息与用户-商品交互信息结合，以更好地进行推荐。

     - 为了考虑用户间对物品偏好的影响，通常将用户间的社会关系与用户-物品交互结合起来，构建所谓的社交RS 

     - 为了深入地描述商品，商品特征往往与用户-商品交互结合在一起，为冷启动商品提供建议

   - 将两种异构的推荐信息进行组合，得到两种异构图:一种是基于用户-项目交互的二部图，另一种是用户之间的社交图或项目-特征图。

   - 两个图中的共享用户或项都充当连接它们的桥接器。社会关系或项目特征对于通过考虑偏好传播来深入了解用户，或者通过考虑项目的自然属性来更好地描述项目是非常重要的

   - 挑战：使来自两个图的异构信息能够适当地相互通信，并在本质上进行组合以有利于推荐任务

6. 多源异构图上的RS

   - 为了有效地解决无处不在的数据稀疏和冷启动问题，
   - 除了用户-项目交互，许多相关信息可以有效地利用和集成到RS中，这些信息对来自多个来源的建议有很大的显性或隐性影响。
     - 用户信息表中的用户概况，唯一或离线社交网络中的社会关系，网站信息表中的商品特征，交易表中的商品共现关系等
     - 可以同时用于帮助更好地理解用户偏好和项目特征，以改进推荐
   - 因此，共同构建多个异构图进行推荐:基于用户-物品交互的二部图提供了建模用户选择的关键信息，用户属性图和社交图提供了用户的辅助信息，而基于物品属性图和物品共现图提供了物品的辅助信息。
   - 挑战：
     - 由于这种异质性，不同图上的信息是相对分离的，不能立即使用，因此如何利用不同的图来相互补充和受益是第一个挑战
     - ORE异质图意味着不同图之间可能存在噪声甚至矛盾的风险更高，如何从多源异构图中提取相干信息，减少噪声和非相干信息，从而改进下游推荐是另一大挑战

## 3.面向RS的图学习方法(GLRS)

GLRS方法的分类如图所示。首先将GLR分为四类，然后将某些类别(如图神经网络方法)进一步划分为多个子类别。

![](/../GNN.jpg)

1. 随机游走 Random Walk 
   - 广泛应用于各种图(如用户之间的社交图、项目之间的共现图，捕捉节点之间的复杂关系
   - 基于随机游走的RS首先让随机游走者在构建的以用户和/或项目为节点的图上行走，每一步都有一个预先定义的转移概率来模拟用户和项目之间的隐含偏好或交互传播，然后根据随机游走者在一定步骤前到达节点的概率来对这些候选节点进行推荐排序。
   - 独特的工作机制，善于捕捉各种点之间复杂的、高阶的、间接的关系，从而可以解决同质或异构图中生成推荐的问题
   - 基于随机游走的RS有不同的变体，除了基本的基于随机游走的RS，基于重启的随机游走的RS 是一个基于随机游走的RS的另一个代表类型。
     - 它设置一个常数概率跳回到起始节点，并且通常用于包含许多节点的图形中，以避免移出起始节点的特定上下文。
     - 转移概率是决定推荐结果的关键因素之一
     - 为了提供更特性化的推荐服务，一些基于随机游走的RS计算每个步骤的特定于用户的转移概率
   - 基于随机游走的RS的其他典型应用包括：
     - 对项目进行排序，根据项目在项目-项目共视图图上的重要性进行排序
     - 通过同时在用户-项目二部图上的用户-项目交互进行建模，同时使用项目-项目邻近关系来引导转移，向用户推荐排名靠前的n个项目
   - 缺点：
     - 他们需要在每个用户的每一步生成所有候选项的排序分数，因此效率较低，难以应用于大规模图
     - 与大多数学习范式不同，基于随机游走的RS是基于启发式的，缺乏模型参数来优化目标，这大大降低了推荐性能
2. 图表示学习 Graph Representation Learning 
   - 分析嵌入在图上的复杂关系的一种有效方法
   - 它将每个节点映射到一个潜在的低维表示，从而将图结构信息编码到其中
   - 图表示学习的RS 可分为三类：
     1. 基于因子分解机的RS(GFMRS)；
        - 采用分解机(例如,矩阵因子分解)根据图上的元路径对节点间交换矩阵进行因式分解，获取每个节点的潜在表示，这些表示将用作后续推荐任务的输入。通过这种方式，嵌入在图中的节点之间的复杂关系被编码到潜在表示中。
        - GFMRS能够处理节点的异构性,已被广泛应用于捕获不同类型节点(如用户和项目)之间的关系
        - 缺点：容易受到观测数据稀疏性的影响，因此很难实现理想的建议
     2. 基于图分布式表示的RS(GDRRS)；
        - 遵循Skip-gram模型来学习图中每个用户或项目的分布式表示，以将用户或项目的自身信息及其相邻关系编码为低维向量
        - 通常先使用随机游走来生成一个在一条元路径中共同出现的节点序列，然后使用Skipgram或相似模型来生成推荐的节点表示。
        - GDRRS利用其强大的功能对图上的节点间连接进行编码，广泛应用于同质或异构图形，以捕获RS中各种对象之间的关系
        - GDRRS的简单、高效和有效，在没有深层或复杂的网络结构的情况下，GDRRS显示出巨大的潜力
     3. 基于图神经嵌入的RS(GNERS)。
        - GNER通常利用神经网络，如多层感知器（MLP），来学习用户项在图形中的嵌入，然后使用所学习的嵌入进行推荐。
        - 神经嵌入模型易于与其他下游神经推荐模型（例如基于RNN的模型）集成，以构建端到端的RS，这可以联合训练两个模型，以实现更好的优化
        - 广泛应用在属性图、异构图、多源异构图
3. 图神经网络
   - 基于GNN的RS主要可分为三类：
     - 基于图注意网络的RS(GATRS)
       - 将注意机制引入GNN，以区别性地学习其他用户或项目的不同相关性和影响程度
         - 通过学习关注权重来认真集成邻居中的信息到用户和项目的表示上
         - 强调那些更重要的用户或项目对特定用户或项目的影响，这更符合实际情况
       - 广泛应用于社会图、项目会话图和知识图
     - 基于门控图神经网络的RS(GGNNRS)
       - 将GAT递归单元（GRU）引入GNN，通过迭代吸收图中其他节点的影响来综合捕获节间关系，从而学习优化的节点表示
       - 通过综合考虑嵌入在相应用户或项目图上的复杂的用户间或项目间关系，学习用户或项目嵌入以供推荐。
       - 具有在图上捕捉复杂关系的能力，被广泛用于:
         - 为基于会话的推荐建立会话图中项目之间的复杂转换模型
         - 为时尚推荐建立不同类别时尚产品之间的复杂交互模型
     - 基于图卷积网络的RS(GCNRS)
       - 通常学习如何利用图结构和节点特征信息，使用神经网络从局部图邻域迭代聚合特征信息。
       - 利用卷积和池化操作，通过 在图中有效地聚合 来自 用户和项的邻域的信息 来学习用户和项的信息嵌入
       - 具有强大的特征提取和学习能力，特别是在结合图形结构和节点内容形成方面的优势
       - 广泛应用于：
         - 社交推荐中社交图上的影响力扩散
         - 掘用户项交互图中隐藏的用户项连接信息，以缓解协同过滤中的数据稀疏问题
         - 通过在基于项目属性的知识图上挖掘其关联属性来捕获项目间相关性
4. 知识图谱
   - 基于知识图的RS(KGRS)通常基于外部知识构建知识图
   - 例如边信息，探索用户或项目之间的隐式或高阶连接关系，以丰富其表示，从而提高推荐性能
   - 由于利用了额外的知识，KGRS能够更好地理解用户行为和物品特征，从而产生更具解释性的建议
   - KGRS主要集中在RS的早期阶段KG的构建，而各种现有技术（包括因子分解机、图神经网络）被用于从构建的KG中提取信息并将其集成到后续建议中
   - 根据用于构建KG的知识，KGRS通常可以分为三类：
     - 基于本体的KGRS(OKGRS)
       - 建立了一个基于用户或项目本体的层次KG,以表示树状结构中的层次归属关系
       - 层次KG的一个典型示例: 亚马逊网站使用的树形图，其中产品类别用于组织平台上所有的销售商品
       - 在此类图中，根节点表示最粗粒度的类别（如食品），而叶节点表示特定的项目（如面包）
       - 应用实例：使用它从项目本体图中提取多级的用户兴趣
     - 基于边信息的KGRS(SKGRS)
       - 基于用户或项目的侧面信息（例如，项目属性）构建KG，以发现它们之间的隐含联系
         - 如：共享相同属性值的项目之间的隐式连接（例如，饮料）提供了额外的信息来理解RS中的项目间关系
       - 广泛应用于协同过滤，通过加入额外信息来改进推荐，从而丰富项目表示
     - 基于共同知识的KGRS(CKGRS)
       - 基于共同知识构建KG，例如从在线文本、领域知识等中提取的一般语义
       - 整合了从共同知识中提取的推荐产品或服务之间的外部隐含关系来改进推荐
       - 广泛应用于：
         - 新闻推荐中以发现新闻
         - 电子商务之间潜在的知识层次联系，从而推断用户的潜在需求

## 4.开放式研究方向

- 面向RS的动态图学习
  - 在现实世界的RS中，对象（包括用户和项目）以及它们之间的关系随着时间的推移而变化，从而导致是动态图而不是静态图
  - 这种动态可能会对推荐结果产生重大影响，甚至会随着时间的推移而改变推荐
  - 在现有的GLRS中，这种情况往往被忽略或研究得较少
  - 因此，研究RS的动态图是一个重要方向
- 基于因果推理的RS
  - 因果推理是发现对象或动作之间因果关系的主要技术
  - 目前仍然没有完全理解用户选择背后的原因和意图
  - 因此将因果推理引入到GLRS中，构建更高级、更具解释性的RS是一个很有前途的方向
- 面向RS的多源多模态图形学习
  - 推荐的数据点可以来自具有各种模态的多个源，但它们之间相互关联，并且协作地为推荐做出贡献
  - 一些类型的数据点可能会给其他类型的数据点带来一些噪声，这些不同的数据点之间甚至可能存在一些矛盾
  - 因此，如何在多源多模态图形上建立一个高效的GLRS值得进一步探索
- 面向RS的大规模实时图形学习
  - 用于RS的数据集太大，导致RS在时间和空间上的成本都很高
  - 这种问题在GLRS中更为明显，因为图形结构数据通常更大，需要更多的时间和空间来处理，更不用说对其执行复杂的机器学习技术来生成推荐
  - 因此，有必要进一步研究更先进的GLRS，使大规模实时计算能够生成推荐

## 5.结论

- 图形学习在学习RS中涉及的各种对象之间的复杂关系方面显示出巨大的优势。
- 催生了一种全新的RS范式：基于图形学习的推荐系统（GLRS）。

