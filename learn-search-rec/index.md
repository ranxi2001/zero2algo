---
layout: default
title: 搜广推
description: 召回、排序、广告投放、特征、样本与链路约束
eyebrow: Module 02
---

# 搜广推

搜广推是业务算法岗最常见的核心场景之一。这里把搜索、推荐和广告投放放在同一条链路里讲：先理解召回、粗排、精排、重排，再进入广告投放里的预估、出价、预算和 ROI。

## 这部分会解决什么

- 召回、粗排、精排、重排的职责边界
- 多路召回、排序模型、特征工程、样本构造的常见做法
- 冷启动、探索、多样性和链路延迟这类真实约束
- 广告投放中的 CTR / CVR / LTV、Budget Pacing、出价优化和价值建模

## 建议阅读顺序

1. [搜广推三层结构：召回 / 粗排 / 精排](./01-search-recall-ranking/index.html)
2. [多路召回与候选集设计](./02-multi-channel-retrieval/index.html)
3. [排序模型：从 LR 到 DIN / DCN](./03-ranking-models/index.html)
4. [冷启动、探索与多样性](./04-cold-start-and-diversity/index.html)
5. [特征工程与样本构造](./05-feature-and-sample-design/index.html)
6. [广告投放链路：定向 / 竞价 / 预算 / 结算](./06-ads-pipeline/index.html)
7. [CTR、CVR、LTV 指标体系](./07-ctr-cvr-ltv/index.html)
8. [Budget Pacing 与流量控制](./08-budget-pacing/index.html)
9. [出价优化与价值建模](./09-bid-optimization/index.html)

## 当前文章

| 文章 | 作用 |
| --- | --- |
| [搜广推三层结构：召回 / 粗排 / 精排](./01-search-recall-ranking/index.html) | 先搭起搜索、推荐、广告共用的排序链路 |
| [多路召回与候选集设计](./02-multi-channel-retrieval/index.html) | 理解召回为什么通常是多路并行 |
| [排序模型：从 LR 到 DIN / DCN](./03-ranking-models/index.html) | 形成常见排序模型的认知地图 |
| [冷启动、探索与多样性](./04-cold-start-and-diversity/index.html) | 补齐效果之外的体验和生态约束 |
| [特征工程与样本构造](./05-feature-and-sample-design/index.html) | 讲清离线训练和线上表现之间的桥梁 |
| [广告投放链路：定向 / 竞价 / 预算 / 结算](./06-ads-pipeline/index.html) | 把广告作为搜广推商业化分支纳入同一链路 |
| [CTR、CVR、LTV 指标体系](./07-ctr-cvr-ltv/index.html) | 理解广告预估和价值建模的核心指标 |
| [Budget Pacing 与流量控制](./08-budget-pacing/index.html) | 理解预算消耗节奏和流量选择 |
| [出价优化与价值建模](./09-bid-optimization/index.html) | 理解出价、ROI、价值和反馈控制 |
