# 1. 赛题综述

随着大模型的发展，算子性能已经成为影响模型吞吐与延迟的关键因素。Triton 作为一种开源的 GPU 编程语言及编译器，兼具跨硬件适配能力与良好的易用性，正逐渐成为产业界与学术界编写自定义算子的主流选择之一。

FlagGems 作为聚焦 Triton 生态的开源算子库，旨在为大模型训练推理提供高效、通用、可扩展的算子解决方案，构建活跃且有竞争力的开源生态。本次算子开发赛道以 “赋能开源生态，吸纳核心贡献者” 为核心宗旨，面向开发者搭建技术交流与能力展示平台。本赛道围绕实际开源需求设有20 道不同难度算子开发赛题，覆盖基础数学运算、深度学习高频算子、前沿复杂场景算子三大类，既为新手开发者提供低门槛的开源入门路径，也为资深技术专家预留突破技术边界的探索空间。

所有参赛方案将经过社区严格评审，获奖作品将直接纳入 FlagGems 开源库，参赛者同步成为社区正式贡献者，享受贡献者专属权益（如生态 roadmap 参与权、技术资源倾斜等）。本赛道希望通过以赛促建的方式，汇聚全球开发者智慧，丰富算子库的功能覆盖、提升核心性能、强化跨平台适配能力，推动 FlagGems 成为 AI 领域极具影响力的开源算子生态。

# 2. 赛题概览

本赛道共设有20 道赛题，按难度与技术考察维度划分，适配不同技术水平的开发者，力求更多开发者加入比赛。

- 难度分布：8 道初级算子题（基础数学 / 通用基础算子）、8 道中级算子题（经典深度学习算子）、4 道高级算子题（前沿复杂场景算子）；
    
- 设计逻辑：以 “基础实现→性能优化→创新突破” 为核心脉络，从基础数学算子的功能实现，到深度学习高频算子的性能调优与反向传播适配，再到前沿复杂算子的算法创新与场景落地，形成完整的能力考察体系；
    
- 核心导向：所有赛题均源于 FlagGems 开源生态的实际需求与技术缺口，聚焦通用数学计算、计算机视觉、语音识别、深度学习框架适配等核心场景，确保参赛成果具备直接的开源应用价值；
    
- 考察维度：兼顾 “功能正确性、性能竞争力、开源适配性、跨平台兼容性、测例完整度、代码可读性多个维度，既重视工程落地能力，也鼓励技术创新探索。
    

# 3. **时间安排**

- 报名阶段：1月9日-5月20日
    
- 开发阶段：2月24日-5月20日
    
- 评审阶段：5月21日-6月4日
    
- 结果发布：6月6日-6月10日
    

# 4. 赛题详情 

20道赛题列表如下：

|算子编号|算子名称|难度|schema|算子分类|Torch API|
|---|---|---|---|---|---|
|1|log10|初级|log10(Tensor self) -> Tensor  <br>log10_(Tensor(a!) self) -> Tensor(a!)  <br>log10.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)|pointwise|[https://docs.pytorch.org/docs/stable/generated/torch.log10.html](https://docs.pytorch.org/docs/stable/generated/torch.log10.html)|
|2|logaddexp|初级|logaddexp(Tensor self, Tensor other) -> Tensor  <br>logaddexp.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)|pointwise|[https://docs.pytorch.org/docs/stable/generated/torch.logaddexp.html](https://docs.pytorch.org/docs/stable/generated/torch.logaddexp.html)|
|3|cosh|初级|cosh(Tensor self) -> Tensor|pointwise|[https://docs.pytorch.org/docs/stable/generated/torch.cosh.html](https://docs.pytorch.org/docs/stable/generated/torch.cosh.html)|
|4|gcd|初级|gcd(Tensor self, Tensor other) -> Tensor|pointwise|[https://docs.pytorch.org/docs/stable/generated/torch.gcd.html](https://docs.pytorch.org/docs/stable/generated/torch.gcd.html)|
|5|tril|初级|tril(input, diagonal=0, *, out=None) -> Tensor|pointwise|[https://docs.pytorch.org/docs/stable/generated/torch.tril.html](https://docs.pytorch.org/docs/stable/generated/torch.tril.html)|
|6|roll|初级|torch.roll(input, shifts, dims=None)|layout|[https://docs.pytorch.org/docs/stable/generated/torch.roll.html#torch.roll](https://docs.pytorch.org/docs/stable/generated/torch.roll.html#torch.roll)|
|7|leaky ReLU|初级|torch.nn.LeakyReLU(negative_slope=0.01, inplace=False) -> Tensor|pointwise|[https://docs.pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html#torch.nn.LeakyReLU](https://docs.pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html#torch.nn.LeakyReLU)|
|8|asinh|初级|asinh(input: Tensor, *, out: Optional[Tensor]) -> Tensor|pointwise|[https://docs.pytorch.org/docs/stable/generated/torch.asinh.html](https://docs.pytorch.org/docs/stable/generated/torch.asinh.html)|
|9|upsample_nearest2d|中级|upsample_nearest2d(Tensor self, SymInt[2] output_size, float? scales_h=None, float? scales_w=None) -> Tensor  <br>upsample_nearest2d_backward(Tensor grad_output, SymInt[2] output_size, SymInt[4] input_size, float? scales_h=None, float? scales_w=None) -> Tensor|upsampling|[https://docs.pytorch.org/docs/stable/generated/torch.nn.UpsamplingNearest2d.html](https://docs.pytorch.org/docs/stable/generated/torch.nn.UpsamplingNearest2d.html)|
|10|scatter_reduce|中级|scatter_reduce.two(Tensor self, int dim, Tensor index, Tensor src, str reduce, *, bool include_self=True) -> Tensor  <br>scatter_reduce_.two(Tensor(a!) self, int dim, Tensor index, Tensor src, str reduce, *, bool include_self=True) -> Tensor(a!)  <br>scatter_reduce.two_out(Tensor self, int dim, Tensor index, Tensor src, str reduce, *, bool include_self=True, Tensor(a!) out) -> Tensor(a!)|index|[https://docs.pytorch.org/docs/stable/generated/torch.scatter_reduce.html](https://docs.pytorch.org/docs/stable/generated/torch.scatter_reduce.html)|
|11|median|中级|median.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)|reduction|[https://docs.pytorch.org/docs/stable/generated/torch.median.html](https://docs.pytorch.org/docs/stable/generated/torch.median.html)|
|12|smooth_l1_loss|中级|smooth_l1_loss(Tensor self, Tensor target, int reduction=Mean, float beta=1.0) -> Tensor|loss|[https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.smooth_l1_loss.html](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.smooth_l1_loss.html)|
|13|pixel_shuffle|中级|torch.nn.functional.pixel_shuffle(input, upscale_factor) -> Tensor|layout|[https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.pixel_shuffle.html#torch.nn.functional.pixel_shuffle](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.pixel_shuffle.html#torch.nn.functional.pixel_shuffle)|
|14|conv_transpose2d|中级|torch.nn.functional.conv_transpose2d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -> Tensor|conv|[https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.conv_transpose2d.html#torch-nn-functional-conv-transpose2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.conv_transpose2d.html#torch-nn-functional-conv-transpose2d)|
|15|avg_pool3d|中级|avg_pool3d(input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override) -> Tensor|reduction|[https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.avg_pool3d.html](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.avg_pool3d.html)|
|16|max_pool3d|中级|max_pool3d(input, kernel_size, stride, padding, dilation, ceil_mode, return_indices) -> Tensor|reduction|[https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.max_pool3d.html](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.max_pool3d.html)|
|17|chunk_gated_delta_rule|高级|无|fused|无单独算子 API；该结构应用于 Qwen3-Next 模型，可尝试提取可融合结构并实现。  <br>• Qwen3-Next 官方模型结构：[https://huggingface.co/Qwen](https://huggingface.co/Qwen)  <br>• 参考实现：[https://github.com/NVlabs/GatedDeltaNet/blob/main/lit_gpt/gated_delta_rule_ops/chunk.py#L616](https://github.com/NVlabs/GatedDeltaNet/blob/main/lit_gpt/gated_delta_rule_ops/chunk.py#L616)|
|18|svd|高级|svd(Tensor self, bool some=True, bool compute_uv=True) -> (Tensor U, Tensor S, Tensor V)|linalg|[https://docs.pytorch.org/docs/stable/generated/torch.svd.html](https://docs.pytorch.org/docs/stable/generated/torch.svd.html)|
|19|ctc_loss|高级|torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0, reduction='mean', zero_infinity=False)|loss|[https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.ctc_loss.html#torch.nn.functional.ctc_loss](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.ctc_loss.html#torch.nn.functional.ctc_loss)|
|20|grid_sample|高级|grid_sample(input, grid, mode, padding_mode, align_corners) -> Tensor|special|[https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html)|

## 4.1 赛题要求

本赛道的所有赛题均围绕 FlagGems 的实际应用需求设计，核心要求是实现算子功能的正确性、性能的竞争力及开源生态的适配性，所有获奖作品需满足 FlagGems 开源贡献标准，可直接集成至社区开源库。

### 4.1.1 功能正确性

- 所有算子输出结果需与官方基准对齐：
    
    - 需覆盖边界用例（如极值、零值、负数、空张量、动态形状），异常输入（如非法 dtype、维度不匹配）需有明确的错误处理逻辑。
        
    - 整数 / 逻辑类算子（如 gcd、tril）：输出结果需与基准完全一致，无偏差；
        
    - 浮点类算子（如 log10、cosh、svd）：与 PyTorch 官方实现误差必须在可接受范围。
        
    
    具体精度评价标准如下：
    
    计算类型的算子：精度验收标准采用torch.allclose接口，具体计算公式如下：
    

**torch.allclose(_input: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)_, _other: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)_, _rtol: [float](https://docs.python.org/3/library/functions.html#float) = 1e-05_, _atol: [float](https://docs.python.org/3/library/functions.html#float) = 1e-08_, _equal_nan: [bool](https://docs.python.org/3/library/functions.html#bool) = False_) → [bool](https://docs.python.org/3/library/functions.html#bool)**

This function checks if **`input`** and **`other`** satisfy the condition:

∣inputi−otheri∣≤atol+rtol×∣otheri∣∣inputi​−otheri​∣≤atol+rtol×∣otheri​∣

若算子(或数据类型)可以做到相对于参考bit级一致的精度，则rtol与atol均定为0，否则取1e-4，atol的数值与计算数据类型相关，具体如下：

|计算数据类型|atol|
|---|---|
|torch.bool|0|
|torch.uint8|0|
|torch.int8|0|
|torch.int16|0|
|torch.int32|0|
|torch.int64|0|
|torch.float8_e4m3fn|1.00E-03|
|torch.float8_e5m2|1.00E-03|
|torch.float8_e4m3fnuz|1.00E-03|
|torch.float8_e5m2fnuz|1.00E-03|
|torch.float16|1.00E-03|
|torch.float32|1.30E-06|
|torch.bfloat16|0.016|
|torch.float64|1.00E-07|
|torch.complex32|1.00E-03|
|torch.complex64|1.30E-06|

### 4.1.2开源适配要求

- 代码需遵循 Flaggems 社区编码规范（变量命名、函数注释、代码分层等），无冗余代码、硬编码；
    
- 算子接口需与 PyTorch 同名算子接口对齐（参数名、输入输出格式、默认值），降低社区用户使用成本；
    
- 需附带完整的单元测试用例（基于 pytest），代码覆盖率必须达到FlagGems仓库要求。
    

### 4.1.3兼容性要求

- 数据类型：至少支持 float32/float16；
    
- 硬件适配：参赛者可自备算力并搭建开发环境，若能适配至少 1 类主办方提供的加速卡，可额外获得加分。
    

> **说明：主办方会提供官方算力支持。有算力和开发环境需求的，请到 FlagOS 官方平台下“赛道一”申请使用 [https://flagos.io/RaceDetail?id=296flq8k&lang=cn。](https://flagos.io/RaceDetail?id=296flq8k&lang=cn%E3%80%82)**

### 4.1.4测例完整度要求

测试用例的完整度和算子功能的完整性为核心考察目标之一，所有赛题的测试用例需全面覆盖以下维度，无遗漏场景：

- 输入规模：覆盖小尺寸（如 1×1、8×8）、常规尺寸（如 64×64、256×256）、大尺寸（如 1024×1024、4096×4096）三类输入规模；
    
- 输入维数：覆盖算子定义的全部合法维数范围（如 tril 需覆盖 2 维矩阵、4 维批量矩阵，roll 需覆盖 1-5 维张量）；
    
- 参数模式：覆盖所有可选参数的默认值、边界值、特殊值模式
    
- 功能完整性：测试用例需验证算子所有功能分支（如 log10 需覆盖原位 / 非原位运算，conv_transpose2d 需覆盖不同 padding/stride/dilation 参数组合）；
    
- 提交要求：需附带测例覆盖清单，明确标注每个测例对应的规模、shape、维数、参数模式，同时提供测例执行结果加速比对比表（与 PyTorch 原生实现逐例比对）。
    

## 4.2 提交规范要求

- 代码结构需严格遵循Flaggems目录规范。具体可参考[https://github.com/flagos-ai/FlagGems/blob/master/CONTRIBUTING.md](https://github.com/flagos-ai/FlagGems/blob/master/CONTRIBUTING.md)
    
- 代码风格必须严格满足 FlagGems 算子库的编码风格规范，c++ 遵循Google C++ Style Guide，Python 遵循 PEP 8，具体可以参考：[https://github.com/flagos-ai/FlagGems/blob/master/docs/code_countribution.md](https://github.com/flagos-ai/FlagGems/blob/master/docs/code_countribution.md)
    
- 代码需为原创
    
- 参赛选手需承诺获奖作品可按 Apache 2.0 协议纳入 FlagGems 开源库，本人作为核心贡献者署名
    

### 4.2.1 提交成果

主办方提供3次成果提交窗口，参赛者须**在窗口期截止前尽早提交PR，一旦收到符合要求的PR，该算子任务将被关闭。**

**提交成果窗口期**

- 截止UTC+8 3月20日 23:59前，在GitHub上的FlagGems仓库（[https://github.com/flagos-ai/FlagGems/pulls](https://github.com/flagos-ai/FlagGems/pulls)）发起PR，**PR title 须标明 [FlagGems Operator Development Competition]**
    
- 截止UTC+8 4月20日 23:59前，在GitHub上的FlagGems仓库（[https://github.com/flagos-ai/FlagGems/pulls](https://github.com/flagos-ai/FlagGems/pulls)）发起PR，**PR title 须标明 [FlagGems Operator Development Competition]**
    
- 截止UTC+8 5月20日 23:59前，在GitHub上的FlagGems仓库（[https://github.com/flagos-ai/FlagGems/pulls](https://github.com/flagos-ai/FlagGems/pulls)）发起PR，**PR title 须标明 [FlagGems Operator Development Competition]**
    

**官方算力的使用时间要求**

如需使用主办方提供的官方统一算力环境，参赛者需前往 **「FlagOS官网_赛道一_赏金猎人」**（[https://flagos.io/RaceDetail?id=296flq8k&lang=cn](https://flagos.io/RaceDetail?id=296flq8k&lang=cn)）认领算子任务并获得算力开发环境。

- 初级算子：自认领成功后，获得5天**官方提供算力**开发环境
    
- 中级算子：自认领成功后，获得10天**官方提供算力**开发环境
    
- 高级算子：自认领成功后，获得15天**官方提供算力**开发环境
    

### 4.2.2 提交地址

参赛者需提交相应代码到以下FlagGems算子库GitHub地址并发起pull request，**PR title 须标明 [FlagGems Operator Development Competition]声明该PR为比赛来源。**

**pull request地址:** [https://github.com/flagos-ai/FlagGems/pulls](https://github.com/flagos-ai/FlagGems/pulls)

## 4.3 评审方案

本次评审方案围绕 “公平、公正、公开” 核心原则制定，旨在筛选出技术达标、适配开源生态、具备实际应用价值的优质算子方案，吸纳符合 Flaggems 社区要求的核心贡献者。评审全程以赛题要求、提交规范为核心依据，结合算子开源落地价值进行综合评议。

评审原则：

1. **技术导向**：优先考量算子的功能正确性、性能竞争力及工程落地性，贴合 FlagGems 开源库的实际应用需求；
    
2. **开源适配优先**：重点关注代码规范、CI 兼容性、社区易用性，确保方案可直接纳入开源库；
    

### **4.3.1 评审流程与依据**：

### （一）评审流程

|评审主体|核心动作|
|---|---|
|Flaggems 核心开发者 + 算法 / 工程专家|1. 形式审查：校验提交代码完整性、代码结构合规性；  <br>2. 自动化校验：执行 Flaggems 官方 CI 流水线，验证代码风格、单元测试、兼容性；  <br>3. 逐项验证算子功能、性能、测例是否达标；  <br>4. 确定获奖并合入仓库。|

### （二）评审依据

1. **核心依据**：《FlagGems 社区算子开发赛题要求》（含赛题要求、提交规范）；
    
2. **参考依据**：
    
    - FlagGems 开源库编码规范、CI 流水线测试标准；
        
    - PyTorch 官方算子接口定义及性能基准；
        
    - 工业级算子开发通用标准（如数值稳定性、跨硬件适配要求）；
        
    - Flaggems 社区开源贡献者准入标准。
        

### 4.3.2 评审标准：

评审总分 100 分，按 “功能正确性、性能竞争力、开源适配性、跨平台兼容性、测例完整度、代码可读性” 多个维度量化评分，各维度权重及评分细则如下：

|评审维度|权重占比|核心评分细则|
|---|---|---|
|功能正确性|30%|1. 结果精度：浮点类算子与基准的绝对误差参考功能正确性说明；整数 / 逻辑类算子结果需与基准完全一致；  <br>2. 边界用例处理：覆盖极值、零值、NaN、Inf 等边界场景；  <br>3. 异常处理：非法输入需给出明确错误提示。|
|性能竞争力|20%|1. 算子必须使用 Flaggems 仓库内置的 benchmark 框架进行 device 端性能测试；  <br>2. 算子相较于 PyTorch 及其他框架原生实现，device 端加速比须 ≥ 0.9，加速比越高得分越高。|
|开源适配性|10%|1. 代码风格：符合 Flaggems 编码规范，并通过 pre-commit 校验（必要）；  <br>2. CI 测试：通过 Flaggems 全部必要的 CI 流水线测试（必要）；  <br>3. 接口适配：与 PyTorch 或 vLLM 及 Transformer Engine 框架接口完全对齐（必要）。|
|跨平台兼容性|10%|1. 能兼容至少一款官方提供硬件加速卡   <br>2. 必须使用FlagTree编译器，版本号为0.3.0rc1+ascend3.2   <br>3. 需要提供算子实现对于硬件算力、带宽利用率数据 据|
|测例完整度|20%|1. 测例在维度、规模、数据类型及接口参数等方面覆盖齐全；  <br>2. 测例质量高，能够高效命中代码覆盖率；  <br>3. 测例开发需参考：[https://github.com/flagos-ai/FlagGems/blob/master/tests](https://github.com/flagos-ai/FlagGems/blob/master/tests)|
|代码可读性|10%|1. 注释齐全；  <br>2. 变量及函数命名规范。|

## 4.4 获奖方案

本赛道围绕 “分层激励、聚焦开源” 核心，根据算子难度，设置**不同的奖励**，所有获奖作品均直接纳入 FlagGems 开源库，参赛者同步成为社区贡献者。每道赛题设有一个获奖名额，若某道赛题无开发者满足获奖标准，则该赛题对应奖项空缺，不降级递补。奖金与算子难度相关，具体如下：

|对应赛题难度|核心权益|
|---|---|
|高级|1. 奖金3000元；   <br>2. 优先获得社区技术资源倾斜（如加速卡资源）；  <br>3. 方案纳入开源库，作者署名；  <br>4.优先参与 FlagGems 生态 roadmap 规划；|
|中级|1. 奖金 2000 元；   <br>2. 优先获得社区技术资源倾斜（如加速卡资源）；   <br>3. 方案纳入开源库，作者署名；|
|初级|1. 奖金 1000 元；  <br>2. 方案纳入开源库，作者署名；|

### **结果公示与通知**

主办方会在FlagOS官网（[https://flagos.io/events?lang=cn&tab=competition](https://flagos.io/events?lang=cn&tab=competition)），每月公布一次阶段性成绩。

- 第一次成绩公布：3月30日
    
- 第二次成绩公布：4月27日
    
- 第三次成绩公布：5月29日