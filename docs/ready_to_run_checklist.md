# 代码运行前检查清单

## ✅ 代码完整性检查

### 1. 代码文件完整性
- ✅ `train_mem.py` - 训练脚本完整
- ✅ `video_chatgpt/model/video_chatgpt.py` - 模型定义完整
- ✅ `video_chatgpt/model/temporal_transformer.py` - TemporalTransformer 实现完整
- ✅ 所有导入正确
- ✅ 所有函数定义完整

### 2. 代码逻辑检查
- ✅ 模型初始化流程完整
- ✅ 数据加载流程完整
- ✅ 训练循环配置正确
- ✅ 保存逻辑正确
- ✅ dtype 和设备处理正确

---

## 📋 运行前必需条件

### 1. 环境依赖

**必需的 Python 包**:
```bash
pip install torch transformers peft bitsandbytes numpy datasets accelerate
```

**检查命令**:
```bash
python check_ready_to_run.py
```

### 2. 模型文件

**需要下载 Llama-2-7B-Chat 模型**:
```bash
# 模型路径: ./checkpoints/Llama-2-7b-chat-hf
# 可以使用 download_all.py 下载
python download_all.py
```

**必需的文件**:
- `config.json`
- `tokenizer.json` 或 `tokenizer.model`
- `*.safetensors` 或 `*.bin` (模型权重)

### 3. 数据文件

**训练数据**:
- 路径: `data/mini_dataset/mini_train.json`
- 格式: JSON 数组，每个元素包含:
  ```json
  {
    "video_id": "xxx",
    "q": "问题文本",
    "a": "答案文本"
  }
  ```

**视频特征**:
- 路径: `data/mini_dataset/features/`
- 格式: `{video_id}.pkl` 文件
- 形状: `(100, 1024)` numpy array (float32)

### 4. 硬件要求

**最低要求**:
- GPU: 至少 8GB 显存（推荐 16GB+）
- 内存: 至少 16GB RAM
- CUDA: 11.8+ (如果使用 GPU)

**检查 GPU**:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

---

## 🚀 运行命令

### 基本运行（使用默认参数）

```bash
python train_mem.py
```

### 自定义参数运行

```bash
python train_mem.py \
  --output_dir ./outputs/baseline \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-4 \
  --max_steps 10 \
  --logging_steps 1 \
  --data_path data/mini_dataset/mini_train.json \
  --features_folder data/mini_dataset/features
```

---

## ⚠️ 可能遇到的问题

### 1. 模型路径不存在

**错误**:
```
FileNotFoundError: ./checkpoints/Llama-2-7b-chat-hf not found
```

**解决**:
```bash
# 下载模型
python download_all.py
# 或手动下载到指定路径
```

### 2. 数据文件不存在

**错误**:
```
FileNotFoundError: data/mini_dataset/mini_train.json
```

**解决**:
- 准备训练数据 JSON 文件
- 准备视频特征 `.pkl` 文件
- 或修改 `DataArguments` 中的默认路径

### 3. CUDA 内存不足

**错误**:
```
RuntimeError: CUDA out of memory
```

**解决**:
- 减小 `per_device_train_batch_size` (如改为 1)
- 增加 `gradient_accumulation_steps` (如改为 8)
- 确保使用 4-bit 量化（代码中已配置）

### 4. 包未安装

**错误**:
```
ModuleNotFoundError: No module named 'peft'
```

**解决**:
```bash
pip install peft bitsandbytes transformers accelerate
```

### 5. TemporalTransformer 导入错误

**错误**:
```
ImportError: cannot import name 'TemporalTransformer'
```

**解决**:
- 确保 `video_chatgpt/model/temporal_transformer.py` 存在
- 检查 `__init__.py` 文件是否正确

---

## ✅ 运行前检查清单

使用检查脚本:
```bash
python check_ready_to_run.py
```

手动检查:
- [ ] Python 包已安装
- [ ] 模型文件已下载
- [ ] 数据文件已准备
- [ ] GPU 可用（推荐）
- [ ] 输出目录可写
- [ ] 代码文件完整

---

## 🎯 快速测试

### 最小测试（1-2 steps）

```bash
python train_mem.py \
  --max_steps 2 \
  --logging_steps 1 \
  --save_strategy "no"
```

**预期输出**:
- ✅ 模型加载成功
- ✅ 数据加载成功
- ✅ 训练开始
- ✅ Loss 正常计算
- ✅ 无错误

---

## 📝 总结

**代码状态**: ✅ **可以直接运行**

**前提条件**:
1. ✅ 环境依赖已安装
2. ✅ 模型文件已下载
3. ✅ 数据文件已准备
4. ✅ GPU 可用（推荐）

**如果满足以上条件，代码应该可以直接运行！**

---

**运行检查脚本**:
```bash
python check_ready_to_run.py
```

