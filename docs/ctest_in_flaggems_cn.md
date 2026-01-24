# FlagGems 中的 C++ 测试

如果你在构建 FlagGems 时将 C 扩展的 CMake 选项 `FLAGGEMS_BUILD_CTESTS` 设置为 `ON`，你可以在 `FlagGems/build/cpython-3xx` 目录下使用以下命令运行 ctest：

```bash
ctest .
```

这将运行 `FlagGems/ctests` 下的所有测试文件。

使用 `ctest -V -R xxx_test` 来运行特定测试并查看日志信息，其中：

- `-R <regex>`：仅运行名称匹配给定正则表达式的测试。
- `-V`：启用详细模式，打印每个测试的详细输出，包括发送到 stdout/stderr 的任何消息。

例如：

```bash
TORCH_CPP_LOG_LEVEL=INFO ctest -V -R test_triton_reduction
```

我们也使用了 PyTorch ATen 日志，因此你需要设置环境变量 `TORCH_CPP_LOG_LEVEL=INFO` 以在 `libtorch_example` 中获取更多日志。
