---
layout: default
title: RAG、知识库与检索增强
description: 检索增强生成系统的基本组件、收益点与常见误区
eyebrow: NLP / AIGC / 03
---

# RAG、知识库与检索增强

RAG 现在很常见，但很多项目其实只是“把检索和生成硬拼起来”。

这篇会从系统视角看 RAG，而不是只看向量库和 prompt。

## 这篇会覆盖什么

- chunk、embedding、retrieval、rerank、generation 的分工
- 为什么很多 RAG 问题本质上是检索问题而不是生成问题
- 如何评估一个知识库系统到底有没有帮上忙

## 当前提纲

- 组件分层
- 典型失效模式
- 评测与迭代方法

下一篇建议继续看：

- [AIGC 评测与效果对齐](../04-aigc-evaluation/index.html)
