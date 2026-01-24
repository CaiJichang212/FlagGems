# FlagGems 中的 Pytest

FlagGems 使用 `pytest` 进行算子精度和性能测试，并进一步利用 Triton 的 `triton.testing.do_bench` 进行内核级性能评估。

### 1. 测试算子精度

- 在特定后端（如 CUDA）上运行参考实现

```bash
cd tests
pytest test_${op_name}_ops.py
```

- 在 CPU 上运行参考实现

```bash
cd tests
pytest test_${op_name}_ops.py --ref cpu
```

### 2. 测试模型精度

```bash
cd examples
pytest model_${model_name}_test.py
```

### 3. 测试算子性能

- 测试 CUDA 性能
  ```bash
  cd benchmark
  pytest test_xx_perf.py -s
  ```
- 测试端到端性能
  ```bash
  cd benchmark
  pytest test_xx_perf.py -s --mode cpu
  ```

### 4. 运行时显示日志信息

```bash
pytest program.py --log-cli-level debug
```

注：在性能测试中**不建议**使用此选项。
