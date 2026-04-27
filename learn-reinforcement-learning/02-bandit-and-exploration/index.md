---
layout: default
title: Bandit 与探索利用
description: 在线决策里探索利用平衡的基本框架和常见策略
eyebrow: 强化学习 / 02
---

# Bandit 与探索利用

很多工业问题还没到需要完整 RL 的程度，但已经需要解决探索利用冲突。

这时 Bandit 常常是第一站。

## 这篇会覆盖什么

- 多臂 Bandit 和 contextual bandit 的区别
- epsilon-greedy、UCB、Thompson Sampling 的直觉
- 为什么很多冷启动和投放问题本质上是探索问题

## 当前提纲

- 基本问题定义
- 常见探索策略
- 线上约束和监控

下一篇建议继续看：

- [RL 在排序和推荐中的用法](../03-rl-for-ranking/index.html)
