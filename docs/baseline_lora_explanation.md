# 为什么 Baseline 版本也要将 mm_projector 加入 LoRA？

## 🤔 问题

在 `baseline-2` 分支中，我们将 `mm_projector`（现在是 `nn.Linear`）加入了 LoRA 的 `target_modules`：

```python
target_modules=["q_proj", "v_proj", "mm_projector"]
```

**为什么这样做？为什么不直接全参数训练 `mm_projector`？**

---

## 💡 原因分析

### 原因 1: 公平对比

**目标**：让 Baseline 和 T-LoRA 版本的可训练参数量接近，确保对比公平。

| 版本 | mm_projector | 训练方式 | 参数量 |
|------|-------------|---------|--------|
| **Baseline** | `nn.Linear` (1024→4096) | LoRA | ~41K 参数 |
| **T-LoRA** | `TemporalTransformer` | 全参数 | ~2-3M 参数 |

**如果 Baseline 全参数训练**：
- Baseline: `mm_projector` 全参数 = 4.2M 参数
- T-LoRA: `TemporalTransformer` 全参数 = 2-3M 参数
- **问题**：Baseline 反而参数量更大！对比不公平

**使用 LoRA 后**：
- Baseline: `mm_projector` LoRA = ~41K 参数
- T-LoRA: `TemporalTransformer` 全参数 = 2-3M 参数
- **优势**：T-LoRA 参数量更大，但引入了时序建模能力，对比更公平

---

### 原因 2: 保持训练策略一致

**目标**：两个版本都使用相同的训练策略（QLoRA），只改变 `mm_projector` 的结构。

**训练策略**：
```
Baseline:
├── LLM (q_proj, v_proj) → LoRA ✅
└── mm_projector (Linear) → LoRA ✅

T-LoRA:
├── LLM (q_proj, v_proj) → LoRA ✅
└── mm_projector (TemporalTransformer) → 全参数 ✅
```

**优势**：
- 两个版本都使用 QLoRA 训练 LLM
- 区别只在 `mm_projector` 的处理方式
- 对比更清晰：**时序建模能力 vs 简单投影**

---

### 原因 3: 降低 Baseline 的训练成本

**目标**：Baseline 作为对比实验，应该尽可能轻量级。

**参数量对比**：

**Baseline (mm_projector 使用 LoRA)**:
```
q_proj LoRA: 32 × 65K = 2.1M
v_proj LoRA: 32 × 65K = 2.1M
mm_projector LoRA: 41K
总计: ~4.2M 可训练参数
```

**Baseline (mm_projector 全参数)**:
```
q_proj LoRA: 32 × 65K = 2.1M
v_proj LoRA: 32 × 65K = 2.1M
mm_projector 全参数: 4.2M
总计: ~8.4M 可训练参数
```

**使用 LoRA 的优势**：
- 参数量减半（4.2M vs 8.4M）
- 训练更快
- 显存占用更少

---

## 🎯 两种方案对比

### 方案 A: Baseline 使用 LoRA（当前方案）

```python
target_modules=["q_proj", "v_proj", "mm_projector"]
```

**优点**：
- ✅ 参数量小，训练快
- ✅ 与 T-LoRA 的训练策略一致（都使用 QLoRA）
- ✅ 对比更公平（T-LoRA 参数量更大，但引入了时序建模）

**缺点**：
- ⚠️ Baseline 的 `mm_projector` 表达能力可能受限（LoRA rank=8）

---

### 方案 B: Baseline 全参数训练（替代方案）

```python
target_modules=["q_proj", "v_proj"]  # 不包含 mm_projector
# 然后手动设置 mm_projector 为可训练
for param in model.get_model().mm_projector.parameters():
    param.requires_grad = True
```

**优点**：
- ✅ `mm_projector` 表达能力更强（全参数）
- ✅ 更接近原始 Video-ChatGPT 的训练方式

**缺点**：
- ⚠️ 参数量更大（8.4M vs 4.2M）
- ⚠️ 训练成本更高
- ⚠️ 与 T-LoRA 的训练策略不一致

---

## 📊 参数量详细对比

### Baseline (mm_projector 使用 LoRA)

```
mm_projector: Linear(1024, 4096)
LoRA rank = 8:
  - LoRA_A: (8, 1024) = 8K
  - LoRA_B: (4096, 8) = 33K
  总计: ~41K 参数
```

### Baseline (mm_projector 全参数)

```
mm_projector: Linear(1024, 4096)
全参数:
  - weight: (4096, 1024) = 4.2M
  - bias: (4096) = 4K
  总计: ~4.2M 参数
```

### T-LoRA (TemporalTransformer 全参数)

```
TemporalTransformer:
  - temporal_pos_embed: (1, 100, 1024) = 102K
  - transformer (2层): ~1.5M
  - output_proj: Linear(1024, 4096) = 4.2M
  总计: ~2-3M 参数
```

---

## 🎯 推荐方案

**当前方案（Baseline 使用 LoRA）更合理**，原因：

1. **公平对比**：T-LoRA 参数量更大，但引入了时序建模能力
2. **训练效率**：Baseline 训练更快，适合快速验证
3. **策略一致**：两个版本都使用 QLoRA，对比更清晰

**如果担心 LoRA 表达能力不足**：
- 可以增加 LoRA rank（如 r=16 或 r=32）
- 或者使用方案 B（全参数训练）

---

## 💡 总结

**为什么 Baseline 也要用 LoRA？**

1. ✅ **公平对比**：确保两个版本的可训练参数量接近
2. ✅ **训练效率**：降低 Baseline 的训练成本
3. ✅ **策略一致**：两个版本都使用 QLoRA，只改变 `mm_projector` 结构

**核心思想**：
- Baseline: 简单投影（Linear）+ LoRA = 轻量级
- T-LoRA: 时序建模（TemporalTransformer）+ 全参数 = 更强的表达能力

这样的对比更能突出 **T-LoRA 的时序建模优势**！

