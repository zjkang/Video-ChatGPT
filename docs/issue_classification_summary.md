# 重复数字问题：训练 vs 推理问题总结

## 🎯 结论：**主要是推理问题**（约80%），但需要验证训练是否充分（约20%）

---

## ✅ 训练设置检查

### 训练数据格式 ✅ 正确

```python
# train_mem.py 第112行
prompt = f"Human: <video> {video_tokens} {q}\nAssistant: {a}</s>"
```
- ✅ 训练数据**包含** `</s>` 标记
- ✅ `</s>` 在 "Assistant:" 之后，属于答案部分

### Labels Masking ✅ 正确

```python
# train_mem.py 第130-141行
prompt_before_answer = f"Human: <video> {video_tokens} {q}\nAssistant:"
before_ids = before_tokenized.input_ids[0].tolist()
labels[:len(before_ids)] = -100  # 只mask掉"Assistant:"之前的内容
```

**关键点**：
- ✅ `</s>` token 在 "Assistant:" **之后**
- ✅ `</s>` token **不会被mask**（因为它在 `len(before_ids)` 之后）
- ✅ `</s>` token **会包含在 labels 中**
- ✅ 模型**应该**学会预测 `</s>` token

**结论**：如果训练充分，模型应该学会了使用 `</s>` 作为结束标记。

---

## ❌ 推理问题（主要问题）

### 问题1: KeywordsStoppingCriteria 的 Bug ⚠️ 严重

```python
# video_chatgpt/model/utils.py 第22行
outputs = self.tokenizer.batch_decode(..., skip_special_tokens=True)[0]
```

**问题**：
- `skip_special_tokens=True` 会**跳过** `</s>` token
- 当 token ID 检测失败时，字符串检测无法找到 `</s>`
- **停止条件完全失效**

### 问题2: 没有使用 eos_token_id ⚠️ 严重

```python
# 修复前
model.generate(
    ...
    # ❌ 没有 eos_token_id
    stopping_criteria=[stopping_criteria]  # 可能失效
)
```

### 问题3: 没有 repetition_penalty ⚠️ 中等

- 模型可能陷入重复生成循环

---

## 🔬 如何验证是训练还是推理问题？

### 方法1: 检查生成的 token 序列

在推理代码中添加：

```python
output_ids = model.generate(...)
eos_token_id = tokenizer.eos_token_id

# 检查是否生成了 </s> token
if eos_token_id in output_ids[0]:
    eos_positions = (output_ids[0] == eos_token_id).nonzero()
    print(f"✅ 模型生成了 </s> token，位置: {eos_positions}")
    print("   → 这是推理停止条件的问题（已修复）")
else:
    print("❌ 模型没有生成 </s> token")
    print("   → 可能是训练问题：模型没有学会使用 </s>")
```

### 方法2: 检查训练数据

```python
# 检查一个训练样本
sample = dataset[0]
prompt = tokenizer.decode(sample['input_ids'])
print(f"训练样本prompt: {prompt}")
print(f"是否包含 </s>: {'</s>' in prompt}")

# 检查 </s> 是否在 labels 中
eos_token_id = tokenizer.eos_token_id
eos_in_labels = (sample['labels'] == eos_token_id).any()
print(f"</s> 是否在 labels 中: {eos_in_labels}")
```

---

## 📊 问题分类

| 问题类型 | 可能性 | 证据 | 状态 |
|---------|--------|------|------|
| **推理停止条件失效** | 80% | KeywordsStoppingCriteria bug | ✅ 已修复 |
| **缺少 eos_token_id** | 80% | 代码中没有设置 | ✅ 已修复 |
| **缺少 repetition_penalty** | 60% | 代码中没有设置 | ✅ 已修复 |
| **模型训练不充分** | 20% | 需要验证模型是否生成 `</s>` | ⚠️ 待验证 |
| **训练数据问题** | 10% | 代码检查显示数据格式正确 | ✅ 已检查 |

---

## 💡 建议的验证步骤

### 步骤1: 测试修复后的推理代码

```bash
python run_cli.py \
    --projection_path ./checkpoints/Baseline_T16 \
    --video_path ./sample_2.mp4 \
    --question "Describe the video." \
    --num_frames 16
```

**观察**：
- 是否还会生成大量重复数字？
- 输出是否在合理长度停止？
- 调试信息中是否显示生成了 `</s>` token？

### 步骤2: 如果问题仍然存在

添加调试代码检查模型是否生成了 `</s>`：

```python
# 在 run_cli.py 中添加
eos_token_id = tokenizer.eos_token_id
if eos_token_id in output_ids[0]:
    print("✅ 模型生成了 </s> token，但停止条件可能仍有问题")
else:
    print("❌ 模型没有生成 </s> token，可能是训练问题")
    print("   建议：增加训练步数或检查训练数据")
```

### 步骤3: 如果模型没有生成 `</s>`

**可能的原因**：
1. 训练步数不够
2. 学习率设置不当
3. 训练数据质量问题

**解决方案**：
- 增加训练步数
- 检查训练 loss 是否正常下降
- 验证训练数据中 `</s>` 的位置

---

## 🎯 当前状态

### ✅ 已修复的推理问题

1. ✅ 添加了 `eos_token_id` 直接检测（最可靠）
2. ✅ 添加了 `repetition_penalty=1.1`
3. ✅ 添加了重复检测和截断逻辑（防御性措施）
4. ✅ 添加了调试信息

### ⚠️ 需要验证的

1. ⚠️ 模型是否真的生成了 `</s>` token？
2. ⚠️ 如果生成了 `</s>`，停止条件是否正常工作？
3. ⚠️ 如果没生成 `</s>`，是否需要重新训练？

---

## 📝 总结

**主要问题**：推理时的停止条件失效（已修复）

**次要问题**：需要验证模型是否学会了使用 `</s>`（待测试）

**建议**：
1. 先测试修复后的推理代码
2. 如果问题解决 → 确认是推理问题
3. 如果问题仍然存在 → 检查模型是否生成 `</s>` token
4. 如果模型没有生成 `</s>` → 需要重新训练或增加训练步数

