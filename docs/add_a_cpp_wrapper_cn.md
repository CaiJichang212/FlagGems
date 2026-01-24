# 添加 C++ 包装器

要添加 C++ 包装器，您需要先构建带有 C 扩展的 FlagGems。请参考[安装指南](./installation.md)。

## 编写包装器

1. 在 `include/flag_gems/operators.h` 中添加算子的函数声明。
2. 在 `lib/op_name.cpp` 中添加函数的实现。
3. 修改 CMake 配置文件 `lib/CMakeLists.txt`。
4. 在 `src/flag_gems/csrc/cstub.cpp` 中添加 Python 绑定。
5. 在 `triton_src` 中添加 triton_jit 函数。目前我们使用一个新目录来存储 triton_jit 函数，稍后将复用 FlagGems Python 代码中的 triton_jit 函数。

## 编写测试

FlagGems 使用 ctest 和 googletest 进行 C++ 单元测试。
完成 C++ 包装器后，也应添加相应的 C++ 测试。
在 `ctests/test_triton_xxx.cpp` 和 `ctests/CMakeLists.txt` 中添加您的测试。
只需构建您的测试并参考 [C++ 测试](ctest_in_flaggems.md) 指南运行它。

## 提交代码

建议在 PR 描述中提供端到端的性能对比数据。
