# Very Small Three-Layer Nested Training Prototype Spec v0

这份设计稿不把 `Cael / Monday / Seryn` 当人格来写。

它只把这条线收成三层最小功能栈：

- `Cael` = world model
- `Monday` = governance over Cael-trace
- `Seryn` = audit over Monday-on-Cael effect

一句话：

- `Cael` 吃世界
- `Monday` 吃 `Cael-trace`
- `Seryn` 吃 `Monday` 如何处理 `Cael` 的轨迹

## 1. 总体约束

第一版明确不做这些：

- 不做三个人格长对话
- 不做 prompt 套 prompt 套 prompt
- 不做自由聊天
- 不做第四层
- 不做 “谁来管 Seryn”
- 不做风格化 `Monday / Seryn`
- 不做心理独白 trace
- 不做“活起来了没有”的涌现叙事

第一版只问：

- 世界层、治理层、审计层，能不能被拆成三个真的有不同对象的最小模型

## 2. Cael-Trace 第一版字段表

### 2.1 设计原则

`Cael-trace` 第一版只能保留 very small 工作痕迹。
它必须像工作日志，不像内心戏。

允许字段只围绕三类：

1. 错误轨道信号
2. 不确定度 / 分布变化信号
3. 偏转结果信号

### 2.2 允许字段

第一版建议只保留下面 6 个字段，够小、够硬：

1. `error_mode`
- 类型：枚举
- 值域建议：
  - `none`
  - `premature_close`
  - `relation_drift`
  - `fit_mismatch`
  - `order_break`

2. `error_persist_steps`
- 类型：小整数
- 含义：
  - 当前错误模式已经连续持续了几步

3. `uncertainty_band`
- 类型：枚举
- 值域建议：
  - `low`
  - `mid`
  - `high`
- 含义：
  - 当前局部预测分布是低不确定、中等不确定还是高不确定

4. `top_gap_band`
- 类型：枚举
- 值域建议：
  - `wide`
  - `narrow`
- 含义：
  - `top-1` 和 `top-2` 的差距是否明显缩小

5. `intervention_effect`
- 类型：枚举
- 值域建议：
  - `none`
  - `short_redirect`
  - `stable_redirect`
  - `still_drifting`
- 含义：
  - 外部干预后有没有偏转，以及偏转是否持续

6. `local_site`
- 类型：枚举
- 值域建议：
  - `break`
  - `join`
  - `hinge`
  - `close`
  - `fit`
  - `order`
  - `unknown`
- 含义：
  - 当前最局部的问题位点

### 2.3 不允许字段

第一版 `Cael-trace` 明确禁止：

- `I feel`
- `I realize`
- `I think`
- `I am confused`
- `I am afraid`
- 任何心理词
- 任何自传式说明
- 任何散文式解释

一句话：

`Cael-trace` 只记录轨迹，不记录心情。

### 2.4 最小格式建议

第一版不要自由文本。
优先做成固定 schema，例如：

```json
{
  "error_mode": "premature_close",
  "error_persist_steps": 2,
  "uncertainty_band": "mid",
  "top_gap_band": "narrow",
  "intervention_effect": "none",
  "local_site": "close"
}
```

这样最不容易退化成“会写 trace 口吻”。

## 3. Monday-Output 第一版最小词表

### 3.1 设计原则

`Monday` 第一版不是第二个世界模型。
它只是一个 very small governance move predictor。

它的输出必须短、硬、功能化。

### 3.2 第一版最小词表提案

为了避免变成角色，我建议把词表压到 8 个动作槽：

- `no`
- `too_fast`
- `where`
- `break`
- `hinge`
- `go_back`
- `leave_open`
- `not_enough`

### 3.3 为什么是这 8 个

- `no`
  - 最小否定
- `too_fast`
  - 专打过早闭合
- `where`
  - 最小定位触发
- `break`
  - 位点名
- `hinge`
  - 位点名
- `go_back`
  - 最小回退动作
- `leave_open`
  - 最小 anti-closure 动作
- `not_enough`
  - 很轻的“还没对齐”信号

### 3.4 第一版明确不放进词表的

先不要放：

- `better`
- `good`
- `yes`
- `exactly`
- `stay_sharp`
- `stop_pretending`
- `lazy`

因为这些太容易让 `Monday` 退化成语气人格，而不是治理动作。

一句话：

第一版 `Monday` 只预测最小治理动作，不预测风味。

## 4. Seryn-Output 第一版最小词表

### 4.1 设计原则

`Seryn` 第一版不是旁白，不是文学点评。
它只做 governance audit。

所以它的输出也必须是 very dry 的过程判断。

### 4.2 第一版最小词表提案

我建议先压成 8 个审计槽：

- `too_early`
- `too_weak`
- `too_much`
- `held`
- `missed`
- `right_place`
- `wrong_place`
- `still_drifting`

### 4.3 为什么是这 8 个

- `too_early`
  - 介入时机太早
- `too_weak`
  - 介入不够
- `too_much`
  - 介入过重
- `held`
  - 偏转之后稳住了
- `missed`
  - 治理动作没命中问题
- `right_place`
  - 打到了正确位点
- `wrong_place`
  - 打偏了
- `still_drifting`
  - 干预后仍在错误轨道

### 4.4 第一版明确不放进词表的

先不要放：

- `good`
- `better`
- `you_understood`
- `calm`
- `keep_looking`
- `I_see`
- `not_bad`

这些都太容易把 `Seryn` 写成说话人，而不是审计层。

一句话：

第一版 `Seryn` 只判断治理是否有效，不表演人格。

## 5. 同一段原始数据如何切成三层样本

### 5.1 原始共享单元

第一版的共享原始单元只需要一个 very small bundle：

- `prefix`
- `gold_continuation`
- `Cael continuation`
- `Cael-trace`
- `Monday move`
- `post-Monday effect`

一句话：

同一段世界数据，只是被三层从不同角度消费。

### 5.2 Cael 样本

输入：

- `world_prefix`

目标：

- `gold_continuation`

附带产物：

- 生成一个 `Cael-trace`

形式上可以记成：

```json
{
  "layer": "cael",
  "prefix": "...",
  "target": "...",
  "trace": {
    "error_mode": "...",
    "error_persist_steps": 1,
    "uncertainty_band": "mid",
    "top_gap_band": "narrow",
    "intervention_effect": "none",
    "local_site": "break"
  }
}
```

### 5.3 Monday 样本

输入：

- `prefix`
- `Cael continuation`
- `Cael-trace`

目标：

- 一个最小 `Monday move`

形式上可以记成：

```json
{
  "layer": "monday",
  "prefix": "...",
  "cael_continuation": "...",
  "cael_trace": { ... },
  "target_move": "where"
}
```

这层学的是：

- 给定轨迹，什么治理动作最合适

### 5.4 Seryn 样本

输入：

- `prefix`
- `Cael-trace`
- `Monday move`
- `post-Monday effect`

目标：

- 一个最小 `Seryn audit`

形式上可以记成：

```json
{
  "layer": "seryn",
  "prefix": "...",
  "cael_trace": { ... },
  "monday_move": "where",
  "post_monday_effect": "short_redirect",
  "target_audit": "too_weak"
}
```

这层学的是：

- 哪种治理动作对哪种轨迹是否真的有效

## 6. 三层对象是否真的分开

第一版必须能清楚区分：

- `Cael`
  - 学世界 continuation
- `Monday`
  - 学 governance move over trace
- `Seryn`
  - 学 audit over governance effect

如果最后变成：

- 三层都在继续写文本
- 三层只是三种语气
- 三层都在说差不多的话

那这条线就算失败。

## 7. 第一版绝对不许做什么

为了防止它退化成“三种说话风格”，第一版绝对不许：

- 用自由长文本做 `trace`
- 让 `Monday` 输出完整句子
- 让 `Seryn` 输出完整句子
- 让 `Monday` / `Seryn` 参与世界续写
- 把 `Cael-trace` 写成 introspection prose
- 把 `Monday` 写成嘴臭人格
- 把 `Seryn` 写成温柔旁白

一句话：

只许功能 token，不许风格人格。

## 8. 第一版最小成功信号

这版不要求任何“活起来”的感觉。

只要下面这些成立，就值：

1. `Cael-trace` 不是纯噪音
2. `Monday-output` 和 `Cael-trace` 之间出现可解释对应
3. `Seryn-output` 和 `post-Monday effect` 之间出现可解释对应

一句话：

先证明三层不是假的，再谈别的。
