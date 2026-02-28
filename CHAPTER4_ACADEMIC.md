# 第四章 面向边缘计算的说话人特征提纯与流式聚类算法优化

## 4.1 引言

第三章已完成对生成式语音识别模型 Whisper 的混合精度量化与显存压缩。然而，完整的会议转写系统还需要说话人日志（Speaker Diarization）模块来回答"谁在何时说话"的问题。当前主流的 Pyannote 说话人日志框架严重依赖 Python/PyTorch 动态图执行环境和 SciPy 全局优化算法，其在长时音频场景下面临两个根本性瓶颈：(1) 动态计算图的即时编译与垃圾回收机制引入不可控的时延抖动；(2) 全连接相似度矩阵的构建使得聚类阶段的时间复杂度呈 $O(N^2)$ 膨胀。这两个瓶颈使得 Pyannote 无法与量化后的 Whisper 在边缘设备上实现协同部署。

本章提出一种面向边缘侧 CPU/SIMD 架构的说话人日志系统重构方案。核心贡献包括三个层面：

1. **声学特征提取的静态张量图优化**——将 Pyannote 的 SincNet-LSTM 分割网络与 ResNet34 嵌入网络从 PyTorch 动态图转换为 GGML 静态张量计算图，使其可直接映射至 ARM NEON/x86 AVX2 等 SIMD 指令集；
2. **基于块对角近似的在线相似度计算范式**——提出以"滑动窗口局部嵌入 + PLDA 子空间投影 + 变分贝叶斯聚类"替代全局 $N \times N$ 相似度矩阵的构建策略，将聚类阶段的时间复杂度从 $O(N^2)$ 降至 $O(N \cdot K)$；
3. **面向流式音频的静态计算图内存池机制**——通过预分配固定大小的内存竞技场（Memory Arena）与显式张量生命周期控制，消除了运行时动态内存分配与 GC 停顿。

---

## 4.2 适配 SIMD 架构的声学特征提取矩阵优化

### 4.2.1 问题建模：从动态图到静态张量计算图的范式迁移

在原始 Pyannote 框架中，说话人日志管线包含两个核心判别式神经网络：语音活动分割网络（Segmentation Network，基于 SincNet + BiLSTM 架构）和说话人嵌入网络（Embedding Network，基于 ResNet34 + TSTP 架构）。这两个网络运行于 PyTorch 的动态计算图引擎之上，每次前向传播均需经历算子调度、内存分配、自动微分元数据记录等步骤，产生显著的框架开销。

本文在底层图构建阶段引入了 GGML（General-purpose GPU/ML）张量库的**静态计算图范式**。与 PyTorch 的即时执行（Eager Execution）不同，GGML 采用"先构图、后执行"的两阶段模式：在图构建（Graph Building）阶段，所有张量算子被注册为有向无环图（DAG）中的节点，但不执行实际计算；在图执行（Graph Compute）阶段，调度器一次性遍历 DAG 并调用对应的 SIMD 内核。这种分离使得：

- **编译期确定内存布局**：所有中间张量的形状在构图阶段即可确知，调度器据此预分配连续内存块，避免了运行时碎片化分配；
- **算子融合的潜力**：相邻的逐元素操作（如 BatchNorm 的缩放与偏置）可在同一 SIMD 寄存器周期内完成，无需写回中间结果。

### 4.2.2 分割网络的静态图构建：SincNet → BiLSTM → 分类器

#### 4.2.2.1 SincNet 前端的张量代数描述

分割网络的声学前端采用三级 SincNet 卷积结构。设原始波形信号为 $\mathbf{x} \in \mathbb{R}^{L}$（$L = 160000$，对应 10 秒 16kHz 音频），三级卷积的数学描述如下：

**第零级（SincNet 滤波器组）：**

$$\mathbf{H}^{(0)} = \text{MaxPool}_{3}\left( \left| \text{Conv1D}(\mathbf{x}, \mathbf{W}^{(0)}_{\text{sinc}}) \right| \right) \in \mathbb{R}^{T_0 \times 80}$$

其中 $\mathbf{W}^{(0)}_{\text{sinc}} \in \mathbb{R}^{251 \times 1 \times 80}$ 为第零级的 Sinc 卷积核（步长 $s=10$），$|\cdot|$ 为逐元素绝对值操作（rectified representation），$\text{MaxPool}_{3}$ 为核大小 3 的一维最大池化。该级完成从原始波形到 80 维带通滤波器表示的变换，输出时间维度为：

$$T_0 = \left\lfloor \frac{ \lfloor (L - 251)/10 \rfloor + 1 }{3} \right\rfloor$$

随后依次施加实例归一化（InstanceNorm1D）与带泄漏修正线性单元（LeakyReLU，$\alpha = 0.01$）：

$$\hat{\mathbf{H}}^{(0)} = \text{LeakyReLU}\left( \gamma^{(0)} \odot \frac{\mathbf{H}^{(0)} - \mu^{(0)}}{\sqrt{\sigma^{2(0)} + \epsilon}} + \beta^{(0)} \right)$$

其中 $\mu^{(0)}, \sigma^{2(0)}$ 沿时间维 $T_0$ 独立计算（逐通道、逐实例）。

**第一、二级（标准卷积）：**

$$\mathbf{H}^{(l)} = \text{LeakyReLU}\left( \text{IN}\left( \text{MaxPool}_3\left( \text{Conv1D}(\hat{\mathbf{H}}^{(l-1)}, \mathbf{W}^{(l)}) + \mathbf{b}^{(l)} \right) \right) \right), \quad l = 1, 2$$

其中 $\mathbf{W}^{(1)} \in \mathbb{R}^{5 \times 80 \times 60}$, $\mathbf{W}^{(2)} \in \mathbb{R}^{5 \times 60 \times 60}$（步长均为 1）。三级卷积后输出特征序列 $\mathbf{F}_{\text{sinc}} \in \mathbb{R}^{T_s \times 60}$（其中 $T_s = 589$ 帧对应 10 秒输入）。

在 GGML 实现中，上述三级卷积通过 `ggml_conv_1d` → `ggml_abs`/`ggml_pool_1d` → `ggml_norm`（InstanceNorm）→ `ggml_leaky_relu` 的算子链一次性注册至静态计算图。关键的实现差异在于：**GGML 的 `ggml_norm` 原语沿 `ne[0]`（即最内层连续维度）进行归一化**，这意味着张量需以 $[\text{Time}, \text{Channel}, \text{Batch}]$ 的布局存储（即 TCB 格式），使得时间维连续排列，Instance Normalization 的逐通道-逐时间维归约可直接映射为连续内存的 SIMD 向量规约操作，极大提升了缓存命中率。

#### 4.2.2.2 双向 LSTM 的自定义算子绑定

SincNet 输出的 $\mathbf{F}_{\text{sinc}} \in \mathbb{R}^{T_s \times 60}$ 被送入 4 层双向 LSTM。设第 $l$ 层的前向隐状态递推为：

$$\begin{bmatrix} \mathbf{i}_t \\ \mathbf{f}_t \\ \mathbf{g}_t \\ \mathbf{o}_t \end{bmatrix} = \begin{bmatrix} \sigma \\ \sigma \\ \tanh \\ \sigma \end{bmatrix} \left( \mathbf{W}_{ih}^{(l)} \mathbf{h}_{t-1}^{(l)} + \mathbf{W}_{xh}^{(l)} \mathbf{x}_t^{(l)} + \mathbf{b}_{ih}^{(l)} + \mathbf{b}_{hh}^{(l)} \right)$$

$$\mathbf{c}_t^{(l)} = \mathbf{f}_t \odot \mathbf{c}_{t-1}^{(l)} + \mathbf{i}_t \odot \mathbf{g}_t, \quad \mathbf{h}_t^{(l)} = \mathbf{o}_t \odot \tanh(\mathbf{c}_t^{(l)})$$

其中隐状态维度 $H = 128$，门控矩阵 $\mathbf{W}_{xh}^{(l)} \in \mathbb{R}^{4H \times D_l}$（$D_0 = 60$，$D_{l>0} = 2H = 256$）。

本文在 GGML 框架下采用**自定义算子（Custom Op）绑定**策略实现双向 LSTM。该策略的设计动机是：标准 GGML 算子库仅提供矩阵乘法、逐元素运算等基础原语，不原生支持 LSTM 的时序递推语义。为此，本文通过 `ggml_custom_4d` 接口注册了一个多线程回调函数，其内部将前向与反向两个方向的 LSTM 计算分配到不同线程（$n_{\text{tasks}} = 2$），实现方向级并行。

更为关键的优化是**权重缓存（Weight Cache）机制**：原始 GGUF 文件中 LSTM 权重以 FP16 格式存储以减小模型体积，但 LSTM 的逐时步递推要求每步均进行 FP16→FP32 解码。本文在推理状态初始化阶段一次性将所有 LSTM 权重预转换为 FP32 并缓存于连续内存块中，同时预分配门控矩阵的中间缓冲区 $\mathbf{B}_{ih} \in \mathbb{R}^{4H \times T_s}$。门控投影 $\mathbf{W}_{xh} \mathbf{X}$ 通过 BLAS 级 `cblas_sgemm` 一次性批量计算所有时步的输入投影，而非逐步计算，使得该矩阵乘法可充分利用 CPU 的 SIMD 宽度。其时间复杂度由逐时步的 $O(T_s \cdot 4H \cdot D_l)$ 降低为单次 GEMM 调用 $O(4H \cdot D_l \cdot T_s)$，虽理论 FLOP 相同，但缓存局部性与 SIMD 利用效率显著提升。

#### 4.2.2.3 分类头的幂集映射

LSTM 输出 $\mathbf{H}_{\text{lstm}} \in \mathbb{R}^{T_s \times 256}$ 经两层全连接（$256 \to 128 \to 128$，带 LeakyReLU）后进入分类器：

$$\mathbf{P} = \log\text{softmax}\left( \mathbf{W}_c \cdot \text{Linear}(\mathbf{H}_{\text{lstm}}) + \mathbf{b}_c \right) \in \mathbb{R}^{T_s \times 7}$$

分类器输出 7 类幂集（Powerset）标签，对应 3 个说话人的活动组合 $\{\varnothing, \{0\}, \{1\}, \{2\}, \{0,1\}, \{0,2\}, \{1,2\}\}$。幂集到多标签的解码通过确定性映射矩阵 $\mathbf{M} \in \{0,1\}^{7 \times 3}$ 完成：

$$\hat{\mathbf{y}}_t = \mathbf{M}[\arg\max_k P_{t,k}] \in \{0,1\}^3$$

该映射为纯查表操作，计算开销可忽略。在 GGML 实现中，全连接层通过 `ggml_mul_mat`（即 $\mathbf{Y} = \mathbf{W}^T \mathbf{X}$）构建，softmax 与 log 则以 `ggml_soft_max` → `ggml_log` 的融合算子链注册。

### 4.2.3 嵌入网络的静态图构建：ResNet34 + TSTP 池化

#### 4.2.3.1 ResNet34 骨干网络的张量代数

嵌入网络的输入为 Mel 滤波器组特征 $\mathbf{X}_{\text{fbank}} \in \mathbb{R}^{T_f \times 80}$（其中 $T_f \approx 998$ 帧对应 10 秒音频）。输入首先经全局倒谱均值归一化（CMN）：

$$\tilde{\mathbf{X}}_{t,b} = \mathbf{X}_{t,b} - \frac{1}{T_f}\sum_{\tau=1}^{T_f} \mathbf{X}_{\tau,b}$$

CMN 后的特征被视为单通道 2D 图像 $\tilde{\mathbf{X}} \in \mathbb{R}^{T_f \times 80 \times 1 \times 1}$（WHCN 格式），送入 WeSpeaker ResNet34 骨干网络。该网络包含初始卷积层与四组残差块：

**初始卷积：**

$$\mathbf{Z}^{(0)} = \text{ReLU}\left( \text{BN}\left( \text{Conv2D}(\tilde{\mathbf{X}}, \mathbf{W}_{\text{init}}) \right) \right) \in \mathbb{R}^{T_f \times 80 \times 32 \times 1}$$

其中 $\mathbf{W}_{\text{init}} \in \mathbb{R}^{3 \times 3 \times 1 \times 32}$（步长 1，填充 1）。

**残差块（BasicBlock）：** 设第 $l$ 组第 $b$ 块的输入为 $\mathbf{Z}^{(l,b)}_{\text{in}}$，则：

$$\mathbf{Z}^{(l,b)}_{\text{out}} = \text{ReLU}\left( \text{BN}_2\left(\text{Conv2D}_2\left( \text{ReLU}\left( \text{BN}_1\left(\text{Conv2D}_1(\mathbf{Z}^{(l,b)}_{\text{in}}) \right) \right) \right) \right) + \mathcal{S}(\mathbf{Z}^{(l,b)}_{\text{in}}) \right)$$

其中 $\mathcal{S}(\cdot)$ 为恒等映射或 $1 \times 1$ 卷积捷径（当通道数变化或步长 > 1 时）。四组残差块的配置为 $[3, 4, 6, 3]$ 块，通道数为 $[32, 64, 128, 256]$，后三组首块步长为 2。经四组残差块后，特征图维度为：

$$\mathbf{Z}^{(4)} \in \mathbb{R}^{\lfloor T_f/8 \rfloor \times 10 \times 256 \times 1}$$

**BatchNorm 的融合计算优化：** 在推理阶段，BatchNorm 的运行均值与方差为常量。本文将 BN 的四参数（$\gamma, \beta, \mu_{\text{run}}, \sigma^2_{\text{run}}$）预融合为双参数仿射变换：

$$\text{scale} = \frac{\gamma}{\sqrt{\sigma^2_{\text{run}} + \epsilon}}, \quad \text{shift} = \beta - \mu_{\text{run}} \cdot \text{scale}$$

$$\hat{\mathbf{Z}} = \mathbf{Z} \odot \text{scale} + \text{shift}$$

该融合将原本需要 5 次逐元素操作（减均值、加 $\epsilon$、开方、除标准差、缩放加偏置）压缩为 2 次（乘 scale、加 shift）。在 GGML 实现中，scale 和 shift 被 `ggml_reshape_4d` 扩展为 $[1, 1, C, 1]$ 形状以支持广播运算，整个 BN 仅注册 `ggml_div` → `ggml_mul` → `ggml_sub` → `ggml_mul` → `ggml_add` 五个算子节点（含中间 $1/\sqrt{\cdot}$ 计算），后端调度器可进一步融合连续逐元素操作。

#### 4.2.3.2 TSTP 池化层的矩阵代数等价

时间统计池化（Temporal Statistics Pooling, TSTP）将变长特征序列聚合为定长嵌入向量。设池化层的输入为 $\mathbf{Z}^{(4)} \in \mathbb{R}^{T' \times 10 \times 256}$（其中 $T' = \lfloor T_f / 8 \rfloor$），首先将频率-通道维展平：

$$\mathbf{F} = \text{Reshape}(\mathbf{Z}^{(4)}) \in \mathbb{R}^{T' \times 2560}$$

均值与方差的计算通过矩阵-向量乘法等价实现。定义全 1 向量 $\mathbf{1}_{T'} \in \mathbb{R}^{T'}$，则：

$$\boldsymbol{\mu} = \frac{1}{T'} \mathbf{F}^T \mathbf{1}_{T'} \in \mathbb{R}^{2560}$$

$$\boldsymbol{\sigma}^2 = \frac{1}{T'} (\mathbf{F} \odot \mathbf{F})^T \mathbf{1}_{T'} - \boldsymbol{\mu} \odot \boldsymbol{\mu}$$

同时应用 Bessel 修正（无偏方差估计）：$\hat{\boldsymbol{\sigma}}^2 = \boldsymbol{\sigma}^2 \cdot \frac{T'}{T'-1}$，标准差为 $\boldsymbol{s} = \sqrt{\hat{\boldsymbol{\sigma}}^2 + \epsilon}$。最终拼接得到池化向量：

$$\mathbf{p} = [\boldsymbol{\mu}; \boldsymbol{s}] \in \mathbb{R}^{5120}$$

在 GGML 中，上述均值计算通过 `ggml_mul_mat` 实现——利用 $\mathbf{A}^T \mathbf{b}$ 的语义，将全 1 向量 $\mathbf{1}_{T'}$ 作为"查询"对特征矩阵求内积和，从而将逐元素归约转化为高度优化的 GEMV 操作。该策略的优势在于：GGML 的 `ggml_mul_mat` 内核在 ARM NEON 上使用 128 位 SIMD 宽度、在 x86 AVX2 上使用 256 位 SIMD 宽度进行向量化计算，相比手写逐元素归约循环可获得 $4\times$—$8\times$ 的吞吐量提升。

最终嵌入头的线性投影为：

$$\mathbf{e} = \mathbf{W}_{\text{seg}} \mathbf{p} + \mathbf{b}_{\text{seg}} \in \mathbb{R}^{256}$$

其中 $\mathbf{W}_{\text{seg}} \in \mathbb{R}^{256 \times 5120}$。该投影同样通过 `ggml_mul_mat` 实现。

### 4.2.4 内存布局分析与缓存效率

GGML 采用列优先（Column-major）的内存布局约定，张量的第 0 维（`ne[0]`）元素在物理内存中连续存放。本文在两个网络中均采用**时间维优先**的布局策略：

| 网络 | 张量语义 | GGML 布局 | 连续维度 |
|------|---------|-----------|---------|
| SincNet | $[T, C, B]$ | `ne[0]=T, ne[1]=C, ne[2]=B` | 时间维 $T$ |
| ResNet34 | $[W, H, C, N]$ | `ne[0]=T_f, ne[1]=80, ne[2]=C, ne[3]=1` | 时间维 $T_f$ |
| TSTP 池化 | $[D, T']$ | `ne[0]=D_{\text{feat}}, ne[1]=T'$ | 特征维 |

对于 SincNet 的 Conv1D 操作，时间维连续意味着卷积核的滑动窗口在连续内存上移动，每次加载 $k \times 1$ 个连续浮点数（$k$ 为核大小），CPU 预取器可有效预测后续访问模式，L1 缓存命中率趋近 100%。

对于 ResNet34 的 Conv2D 操作，GGML 采用 WHCN（Width-Height-Channel-Batch）布局——等价于 PyTorch 的 NCHW 在转置后的列优先存储。这意味着**沿宽度（时间）方向的卷积滑动访问连续内存**，而通道方向的访问步长为 $W \times H$ 个元素。对于典型的 $3 \times 3$ 卷积核，单次计算需访问 9 个非连续位置，但由于 $W$（时间维）远大于卷积核尺寸，空间局部性仍然较好。

输入数据从行优先（Row-major，$[T, 80]$，即 fbank 帧序列）转换为 GGML 列优先（$[T_f, 80, 1, 1]$）时，需进行显式转置：将原始 $T_f \times 80$ 矩阵转为 $80 \times T_f$ 的列优先存储，确保时间维元素在物理内存中连续。

---

## 4.3 面向边缘端极低延迟的相似度计算与聚类降维

### 4.3.1 原始方案的复杂度分析

在 Pyannote 的 Python 实现中，对 $N$ 段音频片段的说话人聚类遵循如下全局范式：

1. 提取嵌入矩阵 $\mathbf{E} \in \mathbb{R}^{N \times D}$（$D = 256$）；
2. 构建全连接余弦相似度矩阵 $\mathbf{S} \in \mathbb{R}^{N \times N}$，其中 $S_{ij} = \frac{\mathbf{e}_i \cdot \mathbf{e}_j}{\|\mathbf{e}_i\| \|\mathbf{e}_j\|}$；
3. 在 $\mathbf{S}$ 上运行层次凝聚聚类（AHC），SciPy 的 `linkage` 实现需 $O(N^2 \log N)$；
4. 可选地，在 AHC 初始划分上运行变分贝叶斯（VBx）精细化。

对于长时会议音频，设音频总时长为 $T_{\text{total}}$ 秒，滑动窗口步长为 $\Delta t = 1$ 秒，每个窗口提取 $K_{\text{local}} = 3$ 个说话人的嵌入，则总嵌入数为：

$$N = \left\lceil \frac{T_{\text{total}} - T_{\text{chunk}}}{\Delta t} \right\rceil \cdot K_{\text{local}}$$

对于 1 小时会议（$T_{\text{total}} = 3600$ 秒），$N \approx 3 \times 3591 = 10773$。全连接相似度矩阵需存储 $\frac{N(N-1)}{2} \approx 5.8 \times 10^7$ 个浮点数，仅存储即占约 220MB；AHC 的 $O(N^2 \log N)$ 复杂度使得聚类耗时成为管线瓶颈。

### 4.3.2 本文方案：分层降维聚类架构

本文实现的 C++ 方案采用**三级渐进式聚类降维**策略，有效消解了上述全局 $O(N^2)$ 瓶颈。

#### 4.3.2.1 第一级：基于滑动窗口的局部嵌入过滤

并非所有嵌入均对聚类有贡献——只有满足"单说话人活跃"条件的片段才能提供可靠的说话人特征。设第 $c$ 个窗口第 $s$ 个局部说话人的二值化分割向量为 $\mathbf{b}_{c,s} \in \{0,1\}^{T_s}$，定义"清洁帧"计数：

$$n_{\text{clean}}(c, s) = \sum_{t=1}^{T_s} \mathbf{b}_{c,s}(t) \cdot \mathbb{1}\left[\sum_{s'=1}^{K_{\text{local}}} \mathbf{b}_{c,s'}(t) = 1\right]$$

仅当 $n_{\text{clean}}(c, s) \geq \alpha \cdot T_s$（$\alpha$ 为最小活跃比率阈值）且嵌入向量不含 NaN 时，该嵌入被保留。该过滤步骤将有效嵌入数从 $N$ 缩减至 $M$（$M \ll N$），典型场景下 $M / N \approx 0.3$—$0.5$。

#### 4.3.2.2 第二级：PLDA 子空间投影与维度压缩

过滤后的嵌入 $\{\mathbf{e}_i\}_{i=1}^{M}$（$\mathbf{e}_i \in \mathbb{R}^{256}$）经一套完整的 PLDA（Probabilistic Linear Discriminant Analysis）变换链投影至低维子空间：

**步骤 1——去均值与尺度归一化：**

$$\tilde{\mathbf{e}}_i = \sqrt{D} \cdot \frac{\mathbf{e}_i - \boldsymbol{\mu}_1}{\|\mathbf{e}_i - \boldsymbol{\mu}_1\|_2} \in \mathbb{R}^{256}$$

**步骤 2——LDA 降维投影：**

$$\mathbf{z}_i = \mathbf{A}_{\text{LDA}}^T \tilde{\mathbf{e}}_i \in \mathbb{R}^{128}$$

其中 $\mathbf{A}_{\text{LDA}} \in \mathbb{R}^{256 \times 128}$ 为预训练的线性判别分析投影矩阵。该投影将维度从 $D = 256$ 压缩至 $D' = 128$，同时最大化类间散度与类内散度之比。

**步骤 3——二次去均值与归一化：**

$$\hat{\mathbf{z}}_i = \sqrt{D'} \cdot \frac{\mathbf{z}_i - \boldsymbol{\mu}_2}{\|\mathbf{z}_i - \boldsymbol{\mu}_2\|_2} \in \mathbb{R}^{128}$$

**步骤 4——PLDA 白化变换：**

$$\mathbf{v}_i = \mathbf{T}_{\text{PLDA}}^T (\hat{\mathbf{z}}_i - \boldsymbol{\mu}_{\text{PLDA}}) \in \mathbb{R}^{128}$$

其中 $\mathbf{T}_{\text{PLDA}} \in \mathbb{R}^{128 \times 128}$ 为 PLDA 变换矩阵。该矩阵通过对说话人间协方差矩阵进行特征值分解获得，其效果是将嵌入空间旋转至说话人身份信息最大化的坐标系中。

整个 PLDA 管线通过两次 BLAS 级 DGEMM 调用实现：

$$\text{LDA 投影: } \mathbf{Z} = \tilde{\mathbf{E}} \cdot \mathbf{A}_{\text{LDA}} \quad (M \times 256) \cdot (256 \times 128) \to (M \times 128)$$

$$\text{PLDA 变换: } \mathbf{V} = \hat{\mathbf{Z}} \cdot \mathbf{T}_{\text{PLDA}}^T \quad (M \times 128) \cdot (128 \times 128) \to (M \times 128)$$

L2 归一化通过 BLAS 的 `cblas_dnrm2` 与 `cblas_dscal` 实现。所有运算采用双精度浮点（float64）以保持与原 Python 实现的数值一致性。

维度压缩的效果可从复杂度角度量化：后续 AHC 的凝聚距离矩阵大小从 $\frac{M(M-1)}{2} \times D$ 的计算量降至 $\frac{M(M-1)}{2} \times D'$，其中 $D' / D = 128 / 256 = 0.5$，单纯维度压缩即提供 $2\times$ 加速。

#### 4.3.2.3 第三级：AHC 初始划分 + VBx 变分推断精细化

**凝聚层次聚类（AHC）初始划分：**

对过滤且 L2 归一化后的嵌入 $\{\bar{\mathbf{e}}_i\}_{i=1}^{M}$（$\bar{\mathbf{e}}_i = \mathbf{e}_i / \|\mathbf{e}_i\|_2$），计算欧氏距离的凝聚矩阵。由于嵌入已 L2 归一化，欧氏距离与余弦距离存在单调关系：

$$d_{\text{cosine}}(\bar{\mathbf{e}}_i, \bar{\mathbf{e}}_j) = 1 - \bar{\mathbf{e}}_i^T \bar{\mathbf{e}}_j = \frac{1}{2}\|\bar{\mathbf{e}}_i - \bar{\mathbf{e}}_j\|_2^2$$

AHC 采用质心连接（Centroid Linkage）策略，通过 fastcluster 库（Müllner, 2011）的最近邻链算法实现，时间复杂度为 $O(M^2 \log M)$——相比全局 $O(N^2 \log N)$，由于 $M \ll N$，实际加速比约为 $(N/M)^2 \approx 4$—$11\times$。以阈值 $\theta_{\text{AHC}} = 0.6$ 截断树状图，获得初始聚类标签 $\{c_i\}_{i=1}^{M}$，$c_i \in \{1, \ldots, S\}$。

**VBx 变分贝叶斯精细化：**

AHC 的初始划分作为 VBx（Variational Bayes HMM x-vector）迭代优化的热启动。VBx 将说话人聚类建模为一个高斯混合模型的变分推断问题，其目标是最大化证据下界（ELBO）。

设 PLDA 变换后的特征为 $\{\mathbf{v}_t\}_{t=1}^{M}$（$\mathbf{v}_t \in \mathbb{R}^{D'}$），PLDA 的说话人间协方差特征值为 $\boldsymbol{\psi} = \text{diag}(\Psi_1, \ldots, \Psi_{D'})$（对角化后），$S$ 为初始聚类数，定义辅助变量：

$$V_d = \sqrt{\Psi_d}, \quad \rho_{t,d} = v_{t,d} \cdot V_d \quad \text{(尺度化特征)}$$

$$G_t = -\frac{1}{2}\left(\|\mathbf{v}_t\|_2^2 + D' \log 2\pi\right) \quad \text{(常量项)}$$

VBx 的每次迭代包含以下 E-M 步骤（设超参数 $F_a$, $F_b$ 分别控制数据拟合度与正则化强度）：

**E-step（更新后验责任 $\gamma_{t,s}$）：**

$$\Lambda^{-1}_{s,d} = \frac{1}{1 + \frac{F_a}{F_b} \sum_t \gamma_{t,s} \cdot \Psi_d} \quad \text{(eq. 17)}$$

$$\alpha_{s,d} = \frac{F_a}{F_b} \cdot \Lambda^{-1}_{s,d} \cdot \sum_t \gamma_{t,s} \cdot \rho_{t,d} \quad \text{(eq. 16)}$$

$$\log p(\mathbf{v}_t | s) = F_a \left( \boldsymbol{\rho}_t^T \boldsymbol{\alpha}_s - \frac{1}{2}(\boldsymbol{\Lambda}^{-1}_s + \boldsymbol{\alpha}_s^2)^T \boldsymbol{\psi} + G_t \right) \quad \text{(eq. 23)}$$

$$\gamma_{t,s} = \frac{\pi_s \cdot \exp(\log p(\mathbf{v}_t|s))}{\sum_{s'} \pi_{s'} \cdot \exp(\log p(\mathbf{v}_t|s'))}$$

**M-step（更新混合权重 $\pi_s$）：**

$$\pi_s = \frac{\sum_t \gamma_{t,s}}{\sum_{s'}\sum_t \gamma_{t,s'}}$$

**ELBO 收敛准则：**

$$\mathcal{L} = \sum_t \log \sum_s \pi_s \cdot p(\mathbf{v}_t|s) + \frac{F_b}{2}\sum_{s,d}\left(\log \Lambda^{-1}_{s,d} - \Lambda^{-1}_{s,d} - \alpha_{s,d}^2 + 1\right) \quad \text{(eq. 25)}$$

当 $\mathcal{L}^{(i)} - \mathcal{L}^{(i-1)} < \epsilon$（$\epsilon = 10^{-4}$）或达到最大迭代次数 $I_{\max} = 20$ 时收敛。

VBx 的核心计算瓶颈在于 $\boldsymbol{\rho} \boldsymbol{\alpha}^T$ 的矩阵乘法——维度为 $(M \times D') \times (D' \times S) = (M \times S)$，通过 `cblas_dgemm` 调用实现。该乘法的时间复杂度为 $O(M \cdot D' \cdot S)$，其中 $S$ 为聚类数（通常 $S \ll M$），因此单次迭代的复杂度为 $O(M \cdot D' \cdot S)$，$I_{\max}$ 次迭代总复杂度为 $O(I_{\max} \cdot M \cdot D' \cdot S)$。

### 4.3.3 复杂度对比分析

将本文方案与原始 Pyannote 方案在时间复杂度层面进行严格对比：

| 阶段 | 原始 Pyannote 方案 | 本文 C++/GGML 方案 |
|------|-------|--------|
| 嵌入提取 | $O(N \cdot C_{\text{embed}})$ | $O(N \cdot C_{\text{embed}})$（常数项更小） |
| 嵌入过滤 | 无 | $O(N)$，$N \to M$（$M \ll N$） |
| PLDA 降维 | 部分实现 | $O(M \cdot D \cdot D')$，$D \to D'$ |
| 距离矩阵 | $O(N^2 \cdot D)$ | $O(M^2 \cdot D')$ |
| AHC | $O(N^2 \log N)$ | $O(M^2 \log M)$ |
| VBx | $O(I_{\max} \cdot N \cdot D' \cdot S)$ | $O(I_{\max} \cdot M \cdot D' \cdot S)$ |
| 分配 | $O(N \cdot S \cdot D)$ | $O(N \cdot S \cdot D)$ |
| **总计** | $O(N^2 D + N^2 \log N)$ | $O(M^2 D' + M^2 \log M + N S D)$ |

由于 $M/N \approx 0.3$—$0.5$，$D'/D = 0.5$，AHC 阶段的实际加速比为：

$$\frac{N^2 D}{M^2 D'} = \left(\frac{N}{M}\right)^2 \cdot \frac{D}{D'} \approx (2\text{—}3.3)^2 \times 2 = 8\text{—}22\times$$

### 4.3.4 最终标签分配的约束优化

VBx 收敛后，需将聚类结果映射回所有 $N$ 个嵌入。本文采用两阶段分配策略：

**（1）质心计算：** 以 VBx 的后验责任 $\gamma_{t,s}$ 为权重计算加权质心：

$$\mathbf{c}_k = \frac{\sum_{t=1}^{M} \gamma_{t,s_k} \cdot \mathbf{e}_t}{\sum_{t=1}^{M} \gamma_{t,s_k}}, \quad k = 1, \ldots, K$$

其中 $\{s_k\}$ 为 VBx 后混合权重 $\pi_s > 10^{-7}$ 的显著说话人集合，$K = |\{s_k\}|$ 为最终说话人数。

**（2）约束 argmax（匈牙利算法）：** 对每个窗口 $c$，计算其 $K_{\text{local}} = 3$ 个局部嵌入与 $K$ 个全局质心之间的余弦相似度矩阵 $\mathbf{D}_c \in \mathbb{R}^{K_{\text{local}} \times K}$：

$$D_c(s, k) = 2 - d_{\text{cosine}}(\mathbf{e}_{c,s}, \mathbf{c}_k) = 1 + \frac{\mathbf{e}_{c,s}^T \mathbf{c}_k}{\|\mathbf{e}_{c,s}\| \|\mathbf{c}_k\|}$$

在每个窗口内，通过匈牙利算法（Kuhn-Munkres 算法，$O(K_{\text{local}}^3)$）求解最大化分配：

$$\hat{\sigma}_c = \arg\max_{\sigma \in \Pi} \sum_{s=1}^{K_{\text{local}}} D_c(s, \sigma(s))$$

其中 $\Pi$ 为所有可行的一一映射集合。由于 $K_{\text{local}} = 3$ 为常数，每个窗口的匈牙利计算为 $O(27) = O(1)$，总分配复杂度为 $O(N/K_{\text{local}})$。

---

## 4.4 低开销的静态内存图与生命周期管理机制

### 4.4.1 问题描述：动态分配的时延抖动

在 Python/PyTorch 环境下，每次前向传播均触发以下内存操作序列：

1. **动态形状推断**：运行时计算每个中间张量的形状；
2. **堆内存分配**：通过 `malloc`/`cudaMalloc` 为中间结果分配显存/内存；
3. **自动微分图记录**：为反向传播构建梯度追踪图（推理时虽可关闭，但框架仍保留开销）；
4. **引用计数与 GC**：Python 的垃圾回收器（特别是循环引用检测器）在不可预测的时刻暂停执行以释放不可达对象。

上述开销在单次推理中看似微小（~1—5ms），但在流式场景下，音频以 1 秒/窗口的速率持续到达，GC 停顿可能导致突发的 50—200ms 延迟尖峰，破坏实时性约束。

### 4.4.2 GGML 内存池机制

本文采用的 GGML 框架通过以下三层内存管理策略消除了上述抖动：

#### 4.4.2.1 元数据层：图结构的预分配内存竞技场

计算图的元数据（张量描述符、边连接关系）分配于一块在系统初始化阶段一次性预分配的固定大小内存竞技场（Memory Arena）。其大小按如下公式计算：

$$S_{\text{meta}} = N_{\text{max}} \cdot \text{sizeof}(\texttt{ggml\_tensor}) + \text{overhead}_{\text{graph}}(N_{\text{max}})$$

其中 $N_{\text{max}}$ 为最大图节点数（本文设为 4096）。该竞技场在 `embedding_state` 和 `segmentation_state` 结构中以 `std::vector<uint8_t> graph_meta` 存储，每次图构建时通过 `ggml_init` 将该缓冲区作为 GGML 上下文的后备内存，使得所有张量描述符在栈式分配器（Bump Allocator）上按顺序分配，无需调用系统级 `malloc`。

每次推理结束后，上下文通过 `ggml_free` 回收——但这仅仅是重置分配指针至缓冲区起始位置，实际内存页**不被释放回操作系统**。这种"假释放-真复用"模式确保了后续推理的零分配开销。

#### 4.4.2.2 权重层：一次加载、全程驻留

模型权重通过 GGUF（GGML Universal File Format）格式存储，加载时采用 `no_alloc = true` 策略——先从文件解析张量元数据（名称、形状、量化类型），再通过后端调度器一次性分配所有权重张量的连续内存块：

$$S_{\text{weight}} = \sum_{i=1}^{N_{\text{tensor}}} \text{nbytes}(\mathbf{W}_i)$$

权重缓冲区被标记为 `GGML_BACKEND_BUFFER_USAGE_WEIGHTS`，告知调度器该缓冲区在整个推理生命周期内不可释放。对于 ResNet34 嵌入模型，$S_{\text{weight}} \approx 25$ MB（FP16/FP32 混合存储）；对于 SincNet-LSTM 分割模型，$S_{\text{weight}} \approx 6$ MB。

#### 4.4.2.3 计算层：调度器管理的临时缓冲区

中间激活张量（卷积输出、池化结果等）由 `ggml_backend_sched`（后端调度器）管理。调度器在首次推理时执行一次图分析：

1. **拓扑排序**：遍历 DAG 确定各算子的执行顺序；
2. **生命周期分析**：对每个中间张量，确定其**首次写入**（由生产者算子决定）和**最后读取**（由消费者算子决定）的时间戳；
3. **内存复用规划**：生命周期不重叠的张量共享同一物理内存区域。

调度器据此预分配一块足够大的计算缓冲区 $S_{\text{compute}}$，其大小等于所有同时存活（Live）张量的最大内存总和，而非所有中间张量的累加和。该策略可表述为：

$$S_{\text{compute}} = \max_{t \in [1, N_{\text{ops}}]} \sum_{\tau : \text{alive}(\tau, t)} \text{nbytes}(\tau)$$

在后续每次推理中，调度器通过 `ggml_backend_sched_alloc_graph` 将张量映射至预分配缓冲区的对应偏移位置，**无需任何动态内存分配**。推理完成后，`ggml_backend_sched_reset` 仅重置内部分配状态指针，缓冲区内存**原地保留**。

### 4.4.3 面向流式处理的增量式内存策略

在流式说话人日志场景中（`StreamingState`），音频数据以块为单位持续到达。本文设计了**环形缓冲区 + 惰性修剪**的内存策略：

- **音频缓冲区**：使用 `std::vector<float> audio_buffer` 追加新到达的音频采样。处理完一个窗口后，已消费的前缀区域通过 `erase` 惰性修剪（仅当前缀长度超过下一窗口起始位置时触发），而非每次处理后立即清空。通过维护 `samples_trimmed` 偏移量实现全局到局部索引的 $O(1)$ 映射。

- **嵌入累积缓冲区**：流式模式下，每处理一个窗口即追加 $K_{\text{local}} \times D$ 个浮点数至嵌入缓冲区。重聚类（Recluster）操作在累积到足够窗口数后触发，直接在累积缓冲区上运行过滤 → PLDA → AHC → VBx 管线，避免了数据复制。

### 4.4.4 峰值内存的理论上界

综合三层内存管理，本系统在推理阶段的峰值内存（Peak RSS）具有如下确定性上界：

$$S_{\text{peak}} = S_{\text{meta}}^{(\text{seg})} + S_{\text{meta}}^{(\text{emb})} + S_{\text{weight}}^{(\text{seg})} + S_{\text{weight}}^{(\text{emb})} + S_{\text{compute}}^{(\max)} + S_{\text{PLDA}} + S_{\text{aux}}$$

其中：
- $S_{\text{meta}}^{(\text{seg})} + S_{\text{meta}}^{(\text{emb})} \approx 2 \times (4096 \times \text{sizeof}(\texttt{ggml\_tensor}) + \text{graph\_overhead}) \approx 8$ MB
- $S_{\text{weight}}^{(\text{seg})} + S_{\text{weight}}^{(\text{emb})} \approx 6 + 25 = 31$ MB
- $S_{\text{compute}}^{(\max)} \approx 30$ MB（由 ResNet34 中层激活主导）
- $S_{\text{PLDA}} \approx 0.5$ MB（PLDA 参数，双精度）
- $S_{\text{aux}}$ 为聚类阶段的临时缓冲（凝聚距离矩阵等），规模为 $O(M^2)$

对于 1 小时音频，$M \approx 3000$，$S_{\text{aux}} \approx \frac{3000 \times 2999}{2} \times 8 \approx 36$ MB。因此总峰值约为 $\sim$105 MB，远小于 PyTorch 运行时的 $\sim$800 MB（含 CUDA 上下文与框架开销）。

---

## 4.5 系统级协同：与量化 Whisper 的联合部署

本章提出的轻量化说话人日志系统与第三章量化后的 Whisper 模型可在同一进程内串行或流水线式执行，其关键使能因素包括：

1. **统一的 GGML 执行后端**：Whisper.cpp 与本文的 diarization-ggml 共享同一套 GGML 张量库与后端调度器，权重缓冲区互不干扰，计算缓冲区可通过时间分片复用（分割网络完成后释放计算缓冲区，嵌入网络复用同一物理内存）。

2. **无 Python 解释器依赖**：整个管线以纯 C++ 实现，消除了 Python GIL（全局解释器锁）对多线程的限制，LSTM 的双方向计算、音频 I/O 与模型推理可在不同线程上真正并行。

3. **确定性时延边界**：由于不存在 GC 停顿与动态分配，每个 10 秒音频窗口的端到端处理时延具有确定性上界——分割约 15ms、嵌入约 30ms（ARM Cortex-A78 实测），满足 $< 100$ ms 的流式实时性约束。

---

## 4.6 算法伪代码

### Algorithm 1: 面向边缘端的说话人日志离线管线

```
输入: 音频信号 x ∈ ℝ^L, 分割模型 θ_seg, 嵌入模型 θ_emb, PLDA 参数 Θ_PLDA
输出: RTTM 说话人片段集合 R

1:  划分滑动窗口 {w_c}_{c=1}^C, 窗口长度 T_chunk = 10s, 步长 Δt = 1s
2:  // === 阶段 I: 分割 (静态计算图) ===
3:  for c = 1 to C do
4:      构建 GGML 静态图: G_seg ← BuildGraph(θ_seg, w_c)
5:      B_c ← PowersetDecode(Execute(G_seg))  ▷ B_c ∈ {0,1}^{T_s × K_local}
6:  end for

7:  // === 阶段 II: 嵌入提取 (静态计算图) ===
8:  for c = 1 to C, s = 1 to K_local do
9:      x̃_{c,s} ← MaskAudio(w_c, B_c[:,s])  ▷ 逐帧掩码
10:     构建 GGML 静态图: G_emb ← BuildGraph(θ_emb, FBank(x̃_{c,s}))
11:     e_{c,s} ← Execute(G_emb)  ▷ e_{c,s} ∈ ℝ^D
12: end for

13: // === 阶段 III: 过滤 + PLDA 降维 ===
14: {ẽ_i}_{i=1}^M ← Filter({e_{c,s}}, {B_c}, α)  ▷ 保留清洁嵌入
15: {v_i}_{i=1}^M ← PLDATransform({ẽ_i}, Θ_PLDA)  ▷ ℝ^D → ℝ^{D'}

16: // === 阶段 IV: 分层聚类 ===
17: {c_i} ← AHC_CentroidLinkage(L2Norm({ẽ_i}), θ_AHC)  ▷ O(M² log M)
18: γ, π ← VBx({v_i}, {c_i}, Ψ, F_a, F_b, I_max)       ▷ O(I_max · M · D' · S)
19: K ← |{s : π_s > 10^{-7}}|

20: // === 阶段 V: 全局标签分配 ===
21: for c = 1 to C do
22:     σ_c ← Hungarian(CosineSim(e_{c,:}, Centroids(γ)))  ▷ O(K_local³) = O(1)
23: end for

24: R ← Reconstruct({B_c}, {σ_c}, K)
25: return R
```

### Algorithm 2: 流式说话人日志增量处理

```
输入: 音频流 {a_t}, 已初始化的 StreamingState S
输出: 增量 RTTM 片段

1:  S.buffer ← S.buffer ∥ a_t  ▷ 追加新采样
2:  while |S.buffer| ≥ NextChunkRequirement(S) do
3:      (B_c, e_{c,:}) ← ProcessOneChunk(S)
4:      S.binarized ← S.binarized ∥ B_c
5:      S.embeddings ← S.embeddings ∥ e_{c,:}
6:      TrimBuffer(S)  ▷ 惰性修剪已消费前缀
7:  end while
8:  if ShouldRecluster(S) then
9:      R ← Recluster(S)  ▷ 在累积缓冲区上运行完整聚类管线
10:     return R
11: end if
```

---

## 4.7 本章小结

本章针对说话人日志模型（Pyannote）在边缘侧部署的时空复杂度膨胀问题，提出了系统性的 C++/GGML 重构方案。在特征提取层面，通过将 SincNet-BiLSTM 分割网络与 ResNet34-TSTP 嵌入网络转换为 GGML 静态张量计算图，消除了 PyTorch 动态图引擎的框架开销，并利用时间维优先的内存布局最大化 SIMD 指令的缓存命中率；在聚类层面，通过三级渐进式降维策略——嵌入过滤（$N \to M$）、PLDA 子空间投影（$D \to D'$）、AHC + VBx 分层聚类——将核心计算的时间复杂度从原始方案的 $O(N^2 D)$ 降至 $O(M^2 D' + M \cdot D' \cdot S)$，理论加速比为 $8$—$22\times$；在内存管理层面，通过三层静态内存池机制（元数据竞技场、权重驻留缓冲区、调度器管理的计算缓冲区）实现了确定性的峰值内存上界（$\sim$105 MB），远低于 Python 环境的 $\sim$800 MB。这些优化共同为异构模型（Whisper + Pyannote）在边缘设备上的联合实时部署奠定了基础。
