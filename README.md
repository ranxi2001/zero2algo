# zero2algo

面向算法求职与算法工程的系统教程站，覆盖传统算法、搜广推、广告投放、强化学习、NLP / AIGC、运筹优化、业务算法、内容风控与算法面试。

当前版本先完成了和 `zero2Agent` 同构的静态站骨架，后续会持续补正式正文。

在线地址：

- https://onefly.top/zero2algo

## 模块规划

| 模块 | 目录 | 当前规划 |
| --- | --- | --- |
| 算法岗基础 | `learn-algo-basic/` | 岗位认知、能力栈、指标、实验、求职路径 |
| 搜广推 | `learn-search-rec/` | 召回、粗排、精排、特征、样本构造 |
| 广告投放 | `learn-ad-delivery/` | 投放链路、CTR/CVR/LTV、预算、出价 |
| 强化学习 | `learn-reinforcement-learning/` | Bandit、排序 RL、离线 RL、安全评估 |
| NLP / AIGC | `learn-nlp-aigc/` | 经典 NLP、RAG、AIGC 评测与落地 |
| 运筹调度 | `learn-optimization-scheduling/` | 建模、LP/ILP、匹配、调度、分配 |
| 业务算法 | `learn-business-algorithm/` | 问题抽象、目标函数、策略实验、协作 |
| 内容风控 | `learn-content-safety/` | 规则 + 模型、多模态审核、人工回流 |
| 算法面试 | `learn-algo-interview/` | 项目表达、指标归因、系统设计、Case 题 |
| Final Project | `final-project/` | 真实业务算法实战占位 |

## 站点结构

```text
zero2algo/
├── _layouts/                       # Jekyll 页面模板
├── _data/nav.yml                   # 模块与文章导航
├── assets/                         # CSS / JS / 图片
├── docs/architecture/              # 站点说明
├── learn-algo-basic/               # 模块 01
├── learn-search-rec/               # 模块 02
├── learn-ad-delivery/              # 模块 03
├── learn-reinforcement-learning/   # 模块 04
├── learn-nlp-aigc/                 # 模块 05
├── learn-optimization-scheduling/  # 模块 06
├── learn-business-algorithm/       # 模块 07
├── learn-content-safety/           # 模块 08
├── learn-algo-interview/           # 模块 09
└── final-project/                  # 实战项目
```

## 本地运行

```bash
git clone https://github.com/ranxi2001/zero2algo
cd zero2algo

gem install bundler jekyll
bundle install
bundle exec jekyll serve
```

本地访问：

- `http://localhost:4000/zero2algo`

## 适合谁

- 想从 `zero2Leetcode` 过渡到真实算法岗位的人
- 会刷题、会写模型，但讲不清业务目标和收益归因的人
- 准备推荐、广告、风控、优化、NLP / AIGC、算法平台相关岗位的人
- 想把“模型能力”补齐成“算法工程能力”的人

## 后续内容方向

- 模块正文补齐
- 面试题按维度持续扩展
- 增加真实业务算法项目实战
