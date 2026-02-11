# log10 测例覆盖清单与加速比表模板

## 1. 基本信息

|字段|内容|
|---|---|
|算子|`log10` / `log10_` / `log10.out`|
|分支|`[FlagGems Operator Development Competition]` PR 分支名|
|设备|示例：`Ascend xxx` / `NVIDIA xxx`|
|编译器|`FlagTree 0.3.0rc1+ascend3.2`|
|PyTorch|示例：`2.x.x`|
|FlagGems Commit|`<commit sha>`|

## 2. 赛题精度口径

- `torch.allclose` 判定：
  - bit 级一致场景：`rtol=0`，`atol=0`
  - 非 bit 级一致场景：`rtol=1e-4`，`atol` 取 dtype 对应值
- `log10` 建议按“非 bit 级一致场景”验收。

|dtype|atol|
|---|---|
|`torch.float16`|`1.0e-3`|
|`torch.float32`|`1.3e-6`|
|`torch.bfloat16`|`0.016`|
|`torch.float64`|`1.0e-7`|

## 3. 测例覆盖清单

> 说明：每行对应一个可复现实验点，需覆盖小/常规/大规模、不同维数、默认+特殊值+异常分支。

|Case ID|接口分支|规模分类|shape|维数|dtype|参数模式|输入场景|预期行为|结果|
|---|---|---|---|---:|---|---|---|---|---|
|L10-001|`log10`|小|`(1, 1)`|2|`float16`|默认|正数随机|数值对齐|PASS/FAIL|
|L10-002|`log10`|小|`(8, 8)`|2|`float32`|默认|正数随机|数值对齐|PASS/FAIL|
|L10-003|`log10`|常规|`(64, 64)`|2|`float16`|默认|正数随机|数值对齐|PASS/FAIL|
|L10-004|`log10`|常规|`(256, 256)`|2|`float32`|默认|正数随机|数值对齐|PASS/FAIL|
|L10-005|`log10`|大|`(1024, 1024)`|2|`float16`|默认|正数随机|数值对齐|PASS/FAIL|
|L10-006|`log10`|大|`(4096, 4096)`|2|`float32`|默认|正数随机|数值对齐|PASS/FAIL|
|L10-007|`log10_`|常规|`(20, 320, 15)`|3|`float16`|原位|正数随机|原位结果对齐|PASS/FAIL|
|L10-008|`log10.out`|常规|`(20, 320, 15)`|3|`float32`|out|正数随机|写回结果对齐|PASS/FAIL|
|L10-009|`log10`|边界|`(8,)`|1|`float32`|特殊值|`[-100,-1,-0,0,1,10,inf,nan]`|`nan/inf` 行为对齐|PASS/FAIL|
|L10-010|`log10`|边界|`(0, 16)`|2|`float16`|空张量|空输入|返回空张量|PASS/FAIL|
|L10-011|`log10`|边界|`base[:, ::2]`|2|`float32`|非连续|非 contiguous 输入|数值对齐|PASS/FAIL|
|L10-012|`log10_`|边界|`base[:, ::2]`|2|`float16`|原位+非连续|非 contiguous 输入|原位结果对齐|PASS/FAIL|
|L10-013|`log10.out`|边界|`inp/out: base[:, ::2]`|2|`float32`|out+非连续|非 contiguous 输入输出|写回结果对齐|PASS/FAIL|
|L10-014|`log10.out`|异常|`inp:(4,4), out:(4,5)`|2|`float32`|shape 异常|out shape 不匹配|抛出正确异常|PASS/FAIL|
|L10-015|`log10.out`|异常|`inp float32, out int32`|2|`mixed`|dtype 异常|out dtype 非法|抛出正确异常|PASS/FAIL|
|L10-016|`log10_`|异常|`(4,)`|1|`int32`|原位 dtype 异常|整型输入|抛出正确异常|PASS/FAIL|

## 4. 加速比对比表（逐例）

> 计算方式：`Speedup = PyTorch_latency_ms / FlagGems_latency_ms`，赛题要求建议标注 `>= 0.9` 是否达标。

|Case ID|接口分支|shape|dtype|PyTorch latency (ms)|FlagGems latency (ms)|Speedup|是否达标(`>=0.9`)|备注|
|---|---|---|---|---:|---:|---:|---|---|
|L10-001|`log10`|`(1,1)`|`float16`||||Y/N||
|L10-002|`log10`|`(8,8)`|`float32`||||Y/N||
|L10-003|`log10`|`(64,64)`|`float16`||||Y/N||
|L10-004|`log10`|`(256,256)`|`float32`||||Y/N||
|L10-005|`log10`|`(1024,1024)`|`float16`||||Y/N||
|L10-006|`log10`|`(4096,4096)`|`float32`||||Y/N||
|L10-007|`log10_`|`(20,320,15)`|`float16`||||Y/N||
|L10-008|`log10.out`|`(20,320,15)`|`float32`||||Y/N||

## 5. 建议命令

```bash
# log10 准确性（包含普通/原位/out/边界/异常）
pytest -q tests/test_unary_pointwise_ops.py -k log10

# log10 性能（前向+原位）
pytest -q benchmark/test_unary_pointwise_perf.py -k "log10 or log10_"
```

> 提交 PR 时，建议将本文件拷贝到 PR 描述或附加文档中，并填入真实测量数据。
