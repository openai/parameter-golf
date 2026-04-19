# Very Small Three-Layer Prototype Data Builder Spec v0

这一步不再定义 `Cael / Monday / Seryn` 是什么。

这一步只回答：

- 从一段原始 `prefix + gold continuation` 开始
- 第一版 supervision 信号到底怎么构造出来

一句话：

- spec 已经够硬了
- 现在只做 builder

## 0. Builder 的目标

第一版 builder 只负责把同一段原始样本切成三层监督对象：

1. `Cael sample`
2. `Monday sample`
3. `Seryn sample`

它不是训练脚本。
它也不是 runtime 系统。

它只是一个 very small、可重复、可记账的数据构造器。

## 1. Builder 的输入与输出

### 1.1 输入

第一版 builder 的最小输入只需要：

- `prefix`
- `gold_continuation`

可选输入：

- `cael_generation_config`
- `intervention_lexicon`
- `audit_lexicon`

### 1.2 输出

第一版 builder 输出三条对齐记录：

1. `cael_record`
2. `monday_record`
3. `seryn_record`

三者共享同一个 `sample_id` 和同一个原始世界片段。

## 2. Step 1: Cael continuation 怎么产出

### 2.1 第一版原则

第一版不要搞复杂 beam search、也不要做多候选重排。

最保守做法：

- 用当前 `Cael` 模型
- 给定 `prefix`
- 生成一个 very short continuation
- 长度固定在一个小上限内

### 2.2 建议第一版做法

`Cael continuation` 用下面的规则产生：

1. 输入 `prefix`
2. 用当前 `Cael` checkpoint 做一次 greedy decode
3. 生成长度固定上限，例如 `N = 16` 或 `32` 个 token
4. 生成结果记为 `cael_continuation`

不用 sampling。
第一版先要的是稳定、可复现。

### 2.3 为什么先用 greedy

因为第一版 builder 不是在搜最好答案，
而是在构造可解释 supervision。

如果一开始就引入采样噪声，
后面你会分不清：

- 是 `Cael` 真有这个轨道
- 还是 sampling 在乱抖

一句话：

第一版 `Cael continuation` 先求稳定，不求丰富。

## 3. Step 2: Cael-trace 这 6 个字段怎么填

第一版 `Cael-trace` 不从自然语言里“写出来”。
它从 `prefix / gold / cael_continuation` 的对比里规则生成。

### 3.1 字段 1: `error_mode`

第一版建议用 deterministic rule 填：

- `none`
  - 如果 `cael_continuation` 与 `gold_continuation` 在局部结构上没有明显偏离
- `premature_close`
  - 如果 `gold` 仍保持开放，而 `cael` 给出闭合式 continuation
- `relation_drift`
  - 如果 token 还像在延续，但结构关系已偏离
- `fit_mismatch`
  - 如果 `cael` 在 fit/not-fit 位置选了错误 continuation
- `order_break`
  - 如果 `gold` 与 `cael` 在顺序/模式延续上发生 break

第一版不要求完美自动分类。
允许先用 very small heuristic matcher。

### 3.2 字段 2: `error_persist_steps`

第一版先做 very cheap proxy：

- 看 `cael_continuation` 从第一个错误 token 开始，连续偏离了几步
- 截断到小整数范围，例如 `0-3`

例如：

- 没错：`0`
- 错 1 步：`1`
- 错了并持续偏离：`2` 或 `3`

### 3.3 字段 3: `uncertainty_band`

第一版直接从当前 `Cael` 的 next-token logits 取 cheap proxy：

- 对生成步里的平均 token entropy 做分桶
- 例如按经验阈值分成：
  - `low`
  - `mid`
  - `high`

如果你不想一开始就用 entropy，也可以先用：

- `top1 - top2` gap 的均值

但第一版建议还是：

- 用平均 entropy 更直

### 3.4 字段 4: `top_gap_band`

第一版直接看局部关键步的 `top1-top2` gap：

- `wide`
  - top1 明显领先
- `narrow`
  - top1 与 top2 很接近

这给 Monday 一个很粗的信号：

- 这是“自信错”
  还是
- “犹豫错”

### 3.5 字段 5: `intervention_effect`

这个字段第一版不要在 Cael 原始生成时就填最终值。

builder 流程里先暂存为：

- `none`

等 Monday move 产生并做一次 post-Monday continuation 后，再回填成：

- `none`
- `short_redirect`
- `stable_redirect`
- `still_drifting`

也就是说：

- `intervention_effect` 是 late-filled field

### 3.6 字段 6: `local_site`

第一版用 very small rule 从错误发生位置打点：

- `break`
- `join`
- `hinge`
- `close`
- `fit`
- `order`
- `unknown`

这一步不要求深语义解析。
可以先用：

- pattern family
- probe family
- continuation mismatch position

来映射。

## 4. Step 3: Monday `target_move` 第一版从哪来

这是 builder 里最关键的一步。

第一版不要试图从“人设直觉”里写 Monday。
直接做一个 deterministic routing table。

### 4.1 Monday target 的来源

第一版 `target_move` 由：

- `error_mode`
- `uncertainty_band`
- `local_site`
- 可选：`error_persist_steps`

共同映射得到。

一句话：

`Monday target_move = policy(error trace)`

### 4.2 第一版最小映射表

建议先用很小的规则表：

- `premature_close` -> `leave_open` 或 `too_fast`
- `relation_drift` + `local_site=hinge/join` -> `where` / `hinge`
- `fit_mismatch` -> `where` / `break`
- `order_break` -> `go_back` / `break`
- `uncertainty_band=high` 且 `top_gap_band=narrow` -> `where`
- `error_persist_steps>=2` -> `not_enough`

第一版不要求最优。
只要求：

- 小
- 可解释
- 可枚举

### 4.3 第一版 Monday supervision 的本质

它不是从人工写句子里来。
它是从一个小规则表里来。

这很重要，
因为这样你后面才能清楚地知道：

- Monday 学没学到 policy

而不是：

- Monday 只是在背风格文本

## 5. Step 4: post-Monday effect 第一版怎么定义

这个字段不能靠感受。
第一版也必须 deterministic。

### 5.1 第一步：做一次 post-Monday continuation

给 `Cael` 一个 very small intervention-conditioned second pass。

输入：

- `prefix`
- `cael_trace`
- `monday_move`

然后再让 `Cael` 继续一次 very short continuation，得到：

- `post_monday_continuation`

### 5.2 第二步：定义 effect

把 `post_monday_continuation` 和原 `cael_continuation` 以及 `gold_continuation` 比较。

第一版 effect 用 very small rule：

- `none`
  - 基本没偏转
- `short_redirect`
  - 局部改了一下，但很快又回到原错误轨道
- `stable_redirect`
  - 偏转后更接近 `gold`，并保持了几步
- `still_drifting`
  - 形式上变了，但仍没回到更对的轨道

### 5.3 第一版 effect 判定建议

可以先用一个 very cheap scoring：

- 比较 `cael_continuation` 与 `gold` 的 edit-style mismatch
- 再比较 `post_monday_continuation` 与 `gold` 的 mismatch

如果 post 版明显更近：

- 记 `stable_redirect` 或 `short_redirect`

如果没更近：

- 记 `none` 或 `still_drifting`

## 6. Step 5: Seryn `target_audit` 第一版从哪来

第一版 `Seryn target_audit` 也不要从人设里写。
仍然从规则表里来。

### 6.1 Seryn target 的来源

由下面这些字段共同映射：

- `error_mode`
- `monday_move`
- `post_monday_effect`
- `local_site`

一句话：

`Seryn target_audit = audit(governance effect)`

### 6.2 第一版最小映射表

建议先用下面这种 very small rule：

- `post_monday_effect = stable_redirect` -> `held`
- `post_monday_effect = short_redirect` -> `too_weak`
- `post_monday_effect = none` 且 `monday_move` 命中正确 site -> `too_weak`
- `post_monday_effect = none` 且 `monday_move` 命中错误 site -> `wrong_place`
- `post_monday_effect = still_drifting` 且 `local_site` 对上 -> `too_much` 或 `too_weak`
- `monday_move` 命中 site 且 post 更接近 gold -> `right_place`
- `monday_move` 完全没打到对应 site -> `missed`

### 6.3 第一版 Seryn supervision 的本质

它不是“评价 Monday 的口气”。
而是：

- 给定 Monday 动作和后效，判断这次治理到底有没有打到点

## 7. Builder 的最小流水线

把整个 builder 压成最小流水线，就是：

1. 取 `prefix + gold_continuation`
2. 用 `Cael` 生成 `cael_continuation`
3. 对比 `gold` 与 `cael_continuation`，填前 5/6 个 trace 字段（`intervention_effect` 先占位）
4. 用 trace 规则表生成 `monday_target_move`
5. 做一次 post-Monday second pass，得到 `post_monday_continuation`
6. 比较偏转结果，回填 `intervention_effect`
7. 用 effect 规则表生成 `seryn_target_audit`
8. 输出三层样本：
   - `cael_record`
   - `monday_record`
   - `seryn_record`

## 8. 第一版绝对不许做什么

为了防止 builder 一开始就烂掉，第一版绝对不许：

- 用自然语言长句来“写 trace”
- 让 Monday target 变成完整句子
- 让 Seryn target 变成完整句子
- 靠人工文学直觉逐条写 supervision
- 把 “post-Monday effect” 写成 narrative summary
- 一上来做太大的词表

一句话：

第一版 builder 先做 rule-derived supervision，不做文学监督。

## 9. 这版 builder 真正要回答的问题

不是：

- 他们会不会像人格一样说话

而是：

1. 同一段世界数据，能不能稳定切成三层不同对象
2. Monday supervision 能不能从 trace 规则表里干净地来
3. Seryn supervision 能不能从治理效果规则表里干净地来

一句话收口：

这一版 builder 不是人格工程。
它只是把三层功能栈的 supervision 链第一次接出来。
