---
name: new-article
description: 在 zero2algo 项目中创建新的学习文章。当用户说“写一篇新文章”“创建文章”“新建文章”“在某模块下添加一篇关于 X 的文章”“帮我起草一篇讲 XX 的内容”“整理面经”时触发。适用于所有算法模块下的新内容。
---

# new-article：创建新的算法学习文章

本技能用于在 `zero2algo` 项目中创建符合项目风格的 Markdown 学习文章。

## 项目约定

目录结构：

```text
{module-dir}/
└── {NN}-{slug}/
    └── index.md
```

- `{NN}` 是两位数字编号，例如 `01`、`09`、`10`
- `{slug}` 使用英文小写和连字符，例如 `ranking-system-design`
- 每篇文章独占一个子目录，正文文件统一命名为 `index.md`

已有模块：

| 模块 | 目录 |
| --- | --- |
| 算法岗基础 | `learn-algo-basic/` |
| 搜广推 | `learn-search-rec/` |
| 广告投放 | `learn-ad-delivery/` |
| 强化学习 | `learn-reinforcement-learning/` |
| NLP / AIGC | `learn-nlp-aigc/` |
| 运筹调度 | `learn-optimization-scheduling/` |
| 业务算法 | `learn-business-algorithm/` |
| 内容风控 | `learn-content-safety/` |
| 算法面试 | `learn-algo-interview/` |
| Final Project | `final-project/` |

Frontmatter 格式：

```yaml
---
layout: default
title: {中文标题}
description: {一句话描述}
eyebrow: {模块名} / {NN}
---
```

## 写作风格

`zero2algo` 的读者是准备算法岗位的开发者，可能会刷题、会写模型，但需要补齐业务目标、指标、系统约束和面试表达。

写作时遵守：

1. 先讲问题为什么重要，再讲方法。
2. 用业务目标、指标、数据、模型、系统约束串起内容。
3. 避免只堆模型名词，要说明方法解决什么问题、有什么代价。
4. 主动暴露真实复杂度，比如采样偏差、线上延迟、收益归因、冷启动、成本控制。
5. 语言精炼，段落短，少修饰，多给判断框架。
6. 涉及链路、分层、对比时优先使用 `text` 或 `mermaid` 代码块。
7. 文末用“下一篇建议继续看：”收尾，并给出相对路径链接。

## 文章模板

```markdown
---
layout: default
title: {标题}
description: {一句话描述}
eyebrow: {模块名} / {NN}
---

# {标题}

{开篇：说明这个问题在算法岗位、真实项目或面试里为什么重要。}

## 这篇会覆盖什么

- {要点 1}
- {要点 2}
- {要点 3}

## {核心问题或机制}

{正文}

## {实践建议或常见坑}

{正文}

## 小结

- {核心判断 1}
- {核心判断 2}
- {核心判断 3}

下一篇建议继续看：

- [{下一篇标题}]({相对路径}/index.html)
```

## 执行步骤

1. 确认目标模块、文章编号、标题和 slug。
2. 创建 `{module-dir}/{NN}-{slug}/index.md`。
3. 更新对应模块的 `index.md` 阅读顺序和文章表。
4. 更新 `_data/nav.yml`，让侧边栏能显示新文章。
5. 汇报新增文件和文章位置。
