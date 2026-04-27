---
layout: default
title: site-architecture
description: zero2algo 静态站与内容组织方式
eyebrow: Notes
---

# Site Architecture

`zero2algo` 沿用 `zero2Agent` 的 Jekyll 静态站结构，重点是把课程首页、文档页、模块导航和内容生产方式统一下来。

## 当前结构

| 层级 | 作用 |
| --- | --- |
| `index.html` | 首页展示与学习路线入口 |
| `_layouts/default.html` | Markdown 页面统一布局 |
| `_data/nav.yml` | 模块和文章侧边栏导航 |
| `assets/css/style.css` | 首页视觉与模块卡片样式 |
| `assets/css/docs.css` | 文档页、代码块、目录样式 |
| `learn-*` / `final-project` | 算法主题模块入口目录 |

## 模块分层

1. 基础认知：先讲岗位、能力栈、指标、实验和求职路径
2. 方向专题：搜广推、广告投放、强化学习、NLP / AIGC、运筹调度、业务算法、内容风控
3. 面试与项目：把项目表达、系统设计和真实案例补齐

## 设计原则

- 内容优先，样式服务内容
- 首页和正文页分离，避免互相污染
- 每篇文章独占一个目录，统一用 `index.md`
- 新增模块或文章时，只需要补目录、正文和导航数据
