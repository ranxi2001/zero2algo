# zero2algo

面向业务算法岗位的系统教程站，覆盖搜广推（含广告投放）、强化学习、NLP / AIGC、运筹调度、业务算法、内容风控和算法面试。

在线地址：

- https://onefly.top/zero2algo

## 模块规划

| 模块 | 目录 | 内容 |
| --- | --- | --- |
| 算法岗基础 | `learn-algo-basic/` | 岗位认知、能力栈、指标、实验、求职路线 |
| 搜广推 | `learn-search-rec/` | 召回、粗排、精排、特征样本、广告投放、预算、出价 |
| 强化学习 | `learn-reinforcement-learning/` | MDP、Bandit、排序 RL、离线 RL、安全评估 |
| NLP / AIGC | `learn-nlp-aigc/` | 经典 NLP、RAG、AIGC 评测与落地 |
| 运筹调度 | `learn-optimization-scheduling/` | 建模、LP/ILP、匹配、启发式、调度分配 |
| 业务算法 | `learn-business-algorithm/` | 问题抽象、目标函数、策略实验、跨团队协作 |
| 内容风控 | `learn-content-safety/` | 规则模型协同、多模态审核、人审回流 |
| 算法面试 | `learn-algo-interview/` | 项目表达、指标归因、系统设计、Case、JD 面经 |
| Final Project | `final-project/` | 业务算法作品项目模板 |

## 站点结构

```text
zero2algo/
├── _layouts/                       # Jekyll 页面模板
├── _data/nav.yml                   # 模块与文章导航
├── assets/                         # CSS / JS / 图片
├── docs/architecture/              # 站点说明
├── learn-algo-basic/               # 模块 01
├── learn-search-rec/               # 模块 02，包含广告投放章节
├── learn-reinforcement-learning/   # 模块 03
├── learn-nlp-aigc/                 # 模块 04
├── learn-optimization-scheduling/  # 模块 05
├── learn-business-algorithm/       # 模块 06
├── learn-content-safety/           # 模块 07
├── learn-algo-interview/           # 模块 08
└── final-project/                  # 作品项目
```

## 本地运行

```bash
gem install bundler jekyll
bundle install
bundle exec jekyll serve
```

本地访问：

- `http://localhost:4000/zero2algo`

## 适合谁

- 想从 `zero2Leetcode` 的程序设计算法刷题过渡到真实业务算法岗位的人。
- 会刷题、会训练模型，但讲不清业务目标、指标和收益归因的人。
- 准备推荐、搜索、广告、风控、运筹、NLP / AIGC、算法平台相关岗位的人。
- 想把模型能力补成算法工程能力的人。
