# 逐点动态 (Pointwise Dynamic)

## 逐点操作 (Pointwise operations)

逐点运算符（Pointwise operators）非常容易并行化。大多数并行编程指南都从两个连续向量之间的逐点加法开始。对于 [Triton 中的 vector_add](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html#sphx-glr-getting-started-tutorials-01-vector-add-py)，实现任务划分方案很简单，即每个 CTA 从每个输入向量读取一个连续范围，并写入输出向量的一个连续范围。

然而，逐点运算符的实际用例可能更为复杂：

- 输入张量可能是非连续的：它们在内存中可能是连续的，但不是行优先顺序；或者它们可能不密集；或者它们可能有内部重叠；
- 输入张量可能具有任意且/或不同数量的维度。并且并不总是能够将它们视为相同形状的连续向量；
- 输入张量可能具有不同但可广播（broadcastable）的形状；
- 输入可能混合了张量与非张量；
- 不同的逐点运算符共享计算索引的通用逻辑，而为每个运算符重复编写这些逻辑非常繁琐；

我们提出了一种基于代码生成的方法来解决这些问题。我们的设计原则是：

- 逐点操作通常是内存受限的（memory-bound），因此应避免通过复制张量来使其成为连续向量；
- 逐点运算符应支持任意且/或不同秩（ranks）、大小、步长（strides）的输入，支持广播输入，以及混合张量和非张量；
- 不同的逐点运算符应共享通用的内部设施，可以作为库，也可以作为基于模板的代码生成机制，以减少模板代码；
- 通用的内部设施应是可配置的，以适应不同的后端。

结果是一个装饰器 `@pointwise_dynamic`。它为逐点运算符提供了一个通用包装器，并提供了一种根据操作和输入配置生成 Triton 内核及相应包装器的机制。

## 代码生成 (Code generation)

`pointwise_dynamic` 的基本用法是装饰一个具有返回值的 `triton.jit` 函数，该函数用于将输入映射到输出。这个 jit 函数类似于带有 `__device__` 声明说明符的函数，即可以从设备调用的函数。我们生成一个 Triton jit 函数来调用它，该函数的作用类似于 CUDA 内核（带有 `__global__` 声明说明符的函数），负责在全局内存中加载和存储数据。

为了支持不同秩、形状、步长的输入张量，我们传递输出张量的形状（这也是逐点操作的任务空间）以及每个张量在每个维度上的步长。形状和步长被解包并作为整数传递给内核。由于 Triton 内核不支持将元组作为参数，我们必须为不同数量的形状和步长整数生成不同的内核。尽管 Triton 从 3.3 版本开始支持元组作为参数，但它并不支持元组上的所有操作（索引、迭代等）。

在 Triton 内核中，我们根据任务空间的形状将任务空间中的索引映射到张量的多维索引（multi-index）。然后，我们根据每个张量在每个维度上的步长，将它们从张量多维索引映射到各张量上的内存偏移量。例如，对于形状为 `(2, 3)` 和 `(2, 3)` 的张量进行二进制加法操作，任务空间是 `(2, 3)`，那么任务 ID 4 将被映射到任务空间中的 `(1, 1)`。假设左操作数（lhs）的步长为 `(3, 1)`，则该张量的内存偏移量为 4；右操作数（rhs）的步长为 `(1, 2)`，则其内存偏移量为 3。

对于具有可广播但不同形状的张量，我们首先广播这些形状以获得任务空间的形状，并将每个张量视为任务形状，这会返回共享相同存储但相对于新形状具有新步长的新张量。

在大多数情况下，你可以将装饰后的 Triton jit 函数视为一个表示该操作的标量函数。但请记住，生成的内核会使用 `tl.tensor` 作为输入来调用该装饰函数。因此，请避免在控制流（`if` 或 `while`）中使用 `tl.tensor` 作为条件，因为 Triton 不支持非标量张量作为条件。

在上述描述中，我们将任务索引（整数）映射到每个张量的内存偏移量，因为我们将逐点操作中的任务视为一维张量，并为每个 CTA 进行划分。我们也有其他的任务空间和划分方案，但为了简洁起见，此处将其省略。

除了内核之外，我们还为相应的内核生成包装器。该包装器期望输出具有正确的形状、步长、数据类型（dtype）和设备元数据，并已准备好进行计算。

## 元数据计算 (MetaData Computation)

由于逐点运算符在元数据计算方面共享相似的逻辑，这已作为一个通用函数实现，供所有 `PointwiseDynamicFunction` 使用。它涉及：

- 形状推导（shape inference）：通过广播输入张量形状来推导输出形状；
- 输出布局推导（output layout inference）：如有必要，为输出张量推导合适的布局（步长顺序）；
- 类型提升（type promotion）：根据规定的规则推导输出数据类型；
- 设备推导（device inference）：推导输出设备以及启动内核的设备；
- 输出分配；
- 推导任务空间的秩。这是一个与代码生成相关的因素，取决于参数。它还涉及在所有预分配张量都是密集且非重叠，并且每个维度具有相同大小和步长的情况下，尝试将任务空间的维度减少到 1。

预分配的输出张量也可以传递给 `PointwiseDynamicFunctions`。在输出张量中存在预分配张量的情况下，这些预分配张量的形状、布局、数据类型和设备将受到尊重并进行检查。

元数据计算也可以跳过，但在这种情况下，你应该确保输出具有正确的元数据并已预分配。并且你应该提供任务空间的秩。

## 缓存与分发 (Caching and dispatching)

装饰器 `pointwise_dynamic` 返回一个 `PointwiseDynamicFunction` 对象，它作为所有装饰函数的代理。它缓存所有生成的 Python 模块并分发给它们。

分发结果仅取决于任务空间的秩，而不取决于任务空间的形状。

## 使用 pointwise_dynamic 装饰器 (Use pointwise_dynamic decorator)

### 基础 (Basic)

使用 `pointwise_dynamic` 装饰逐点运算符函数可以节省手动处理张量寻址、张量读/写、并行分块（tiling）、张量广播、动态维度、非连续存储、类型提升等工作。

例如，在以下代码中，你只需要提供一个描述计算逻辑（Payload）的 Triton jit 函数，装饰后的函数就可以接受 Torch 张量作为输入和输出，并支持广播、类型提升等。

```python
@pointwise_dynamic(promotion_methods=[(0, "COMPLEX_TO_FLOAT")])
@triton.jit
def abs_func(x):
    return tl.abs(x)
```

由于装饰后的函数没有为代码生成提供足够的信息，我们通过向 `pointwise_dynamic` 传递参数来提供其他必要信息。

### 张量/非张量 (Tensor/Non-Tensor)

默认情况下，`pointwise_dynamic` 将每个参数都视为张量，并生成加载/存储它们的代码。但可以通过向 `is_tensor` 参数传递布尔值列表来配置，以指示相应的参数是否为张量。

对于非张量参数，可以通过向装饰器传递 `dtypes` 来指定其类型，尽管这不是必需的。对于张量参数，`dtypes` 中相应的值会被忽略，因为其数据类型是动态的，Triton 可以根据它进行分发。

例如，在以下代码中，`alpha` 参数被定义为非张量浮点数，而 `x` 和 `y` 参数被定义为张量。

```python
@pointwise_dynamic(
    is_tensor=[True, True, False],
    dtypes=[None, None, float],
    promotion_methods=[(0,"DEFAULT")]
)
@triton.jit
def add_func(x, y, alpha):
    return x + y * alpha

a = torch.randn(128, 256, device="cuda")
b = torch.randn(256, device="cuda")
add_func(a, b, 0.2)
```

### 输出数据类型 (Output dtypes)

为了让逐点运算符分配具有正确数据类型的输出，`promotion_methods` 是必需的。由于输出数据类型可能根据某些规则依赖于输入数据类型，因此指定规则比直接提供输出数据类型更具表现力。

`promotion_methods` 是一个元组列表（每个输出一个），每个元组由若干个参数索引和一个提升方法组成。参数索引（整数）用于指示参数的位置，由提升方法依赖。提升方法（枚举或字符串）表示类型提升的方法。

- `DEFAULT` 是类型提升的默认规则，适用于大多数数值运算；
- `NO_OPMATH` 表示按原样复制数据类型，适用于非数值运算，如数据复制。

```python
class ELEMENTWISE_TYPE_PROMOTION_KIND(Enum):
    DEFAULT = (0,)
    NO_OPMATH = (1,)
    INT_TO_FLOAT = (2,)
    ALWAYS_BOOL = (3,)
    COMPLEX_TO_FLOAT = (4,)
    BOOL_TO_LONG = (5,)
```

示例：

- `DEFAULT` ：add
- `NO_OPMATH` ： where, nextafter, cat
- `INT_TO_FLOAT` ：sin
- `ALWAYS_BOOL` ：eq
- `COMPLEX_TO_FLOAT` ：abs
- `BOOL_TO_LONG` ：pow

### 输出数量 (Number of outputs)

对于具有多个输出张量的逐点操作，我们需要告知 `pointwise_dynamic` 输出的数量，以便它可以生成存储输出张量的代码。对于输入的数量，可以从 `is_tensor` 或 `dtypes` 的长度推断出来。

```python
@pointwise_dynamic(
    promotion_methods=[
        ((0, 1), "DEFAULT"),
        ((0, 1), "DEFAULT"),
    ],
    num_outputs=2,
)
@triton.jit
def polar_kernel(abs, angle):
    real = abs * tl.cos(angle)
    imag = abs * tl.sin(angle)
    return real, imag
```

## 使用 PointwiseDynamicFunction (Use PointwiseDynamicFunction)

### 基础 (Basic)

`PointwiseDynamicFunction` 可以使用与装饰函数相同的函数签名进行调用，如前面的示例所示。

### 原地操作与输出参数 (Inplace Operation & Output arguments)

由于 `pointwise_dynamic` 生成的包装器将输出作为参数，我们可以使用它来实现原地操作。对于所有的 `PointwiseDynamicFunction`，你可以通过关键字传递输出参数。为了区分输入参数和输出参数，我们现在遵循一个简单的规则：所有输入参数必须按位置传递，而所有输出参数必须通过关键字传递。

输出参数被命名为 `out{output_index}`。由于装饰函数没有返回值的名称，我们简单地采用在 `out` 后加输出索引的命名规则。

我们可以用它来实现原地操作。示例：

```python
@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def add_func(x, y, alpha):
    return x + y * alpha


def add_(A, B, *, alpha=1):
    return add_func(A, B, alpha, out0=A)
```

我们也可以传递预分配的输出张量，该张量不在输入张量中。示例：

```python
@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def add_func(x, y, alpha):
    return x + y * alpha


def add_(A, B, *, alpha=1, out=None):
    return add_func(A, B, alpha, out=out)
```

请注意，在这些情况下，你必须确保输出具有正确的元数据。

### 手动实例化 (Manual Instantiation)

对于某些操作，你可能希望跳过元数据计算，特别是减少任务空间秩的过程，并手动准备所有输入和输出。然后你可以调用 `PointwiseDynamicFunction` 的 `instantiate` 方法，并传入特定的任务秩来获取特定的缓存函数并直接调用它。

例如，`flip` 运算符并不是严格意义上的逐点运算符，因为输出中的每个元素仅取决于输入中对应位置的元素。但如果我们能创建一个具有负步长和偏移数据指针的输入张量视图，它就可以被框定为逐点复制。这就是我们如何使用 `pointwise_dynamic` 实现它的。

```python
@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def copy_func(x):
    return x

def flip(A: torch.Tensor, dims) -> torch.Tensor:
    strides = list(A.stride())
    flip_dims_b = [False for _ in A.stride()]
    for dim in dims:
        assert (
            dim >= -A.dim() and dim < A.dim()
        ), "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(
            -A.dim(), A.dim() - 1, dim
        )
        assert not flip_dims_b[
            dim
        ], "dim {} appears multiple times in the list of dims".format(dim)
        flip_dims_b[dim] = True
    n = 0
    offset = 0
    for i in range(len(flip_dims_b)):
        if flip_dims_b[i] and A.size(i) > 1 and A.stride(i) != 0:
            offset += strides[i] * (A.shape[i] - 1)
            strides[i] = -strides[i]
            n += 1
    if n == 0 or A.numel() <= 1:
        return A.clone()
    out = torch.empty_like(A)
    # a flipped view of A
    flipped_A = StridedBuffer(A, strides=strides, offset=offset)

    overload = copy_func.instantiate(A.ndim)
    overload(flipped_A, out0=out)
    return out
```
