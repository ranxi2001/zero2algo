---
name: new-module
description: 在 zero2algo 项目中创建新的学习模块。当用户说“新建模块”“添加模块”“创建一个新的学习章节”“我想增加一个关于 X 的模块”时触发。负责创建模块目录、index.md，并同步更新主页和导航数据。
---

# new-module：创建新的算法学习模块

本技能用于在 `zero2algo` 项目中新建学习模块，并保持站点导航一致。

## 当前模块结构

```text
zero2algo/
├── learn-algo-basic/
├── learn-search-rec/
├── learn-ad-delivery/
├── learn-reinforcement-learning/
├── learn-nlp-aigc/
├── learn-optimization-scheduling/
├── learn-business-algorithm/
├── learn-content-safety/
├── learn-algo-interview/
└── final-project/
```

## 模块 index.md 结构

```markdown
---
layout: default
title: {模块显示名}
description: {一句话描述}
eyebrow: Module {NN}
---

# {模块显示名}

{2-3 句话说明这个模块解决什么问题，适合什么阶段的读者。}

## 这部分会解决什么

- {主线 1}
- {主线 2}
- {主线 3}

## 建议阅读顺序

1. [{文章 1}](./01-{slug}/index.html)

## 当前文章

| 文章 | 作用 |
| --- | --- |
| [{文章 1}](./01-{slug}/index.html) | {作用} |
```

## 需要同步更新的位置

- `_data/nav.yml`：添加模块和文章导航。
- `index.html`：首页学习路线区域添加模块卡片。
- `_config.yml`：`include` 中添加模块目录。

## 执行步骤

1. 确认模块编号、目录名、显示名、描述和初始文章规划。
2. 创建模块目录和 `index.md`。
3. 更新 `_data/nav.yml`、`index.html` 和 `_config.yml`。
4. 汇报创建和修改的文件。
