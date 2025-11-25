# STT (Speech-to-Text) 模块

## 模块概述

STT 模块负责将语音片段转换为文本，是 VRChat 社交辅助系统语音理解管道的核心环节。

### 核心功能

- **实时语音识别**：支持中英混合语音的实时转文本
- **双引擎架构**：本地引擎（Faster-Whisper）+ 云端引擎（阿里云/腾讯云）
- **自动降级**：引擎失败时自动切换备用引擎
- **质量控制**：置信度评估、文本长度限制
- **统计监控**：识别成功率、延迟等性能指标

### 技术架构

```
┌─────────────────────────────────────┐
│       STTRecognizer (主接口)         │
└──────────────┬──────────────────────┘
               │
┌──────────────┴──────────────────────┐
│       EngineManager (引擎管理)       │
├──────────────┬──────────────────────┤
│  主引擎      │  降级引擎              │
└──────┬───────┴──────┬───────────────┘
       │              │
┌──────┴────┐  ┌──────┴─────┐
│ 本地引擎   │  │  云端引擎   │
│ Whisper   │  │ Aliyun/TX  │
└───────────┘  └────────────┘
```

## 快速开始

### 1. 安装依赖

```bash
pip install faster-whisper requests
```

### 2. 基础使用

```python
from src.stt import STTRecognizer
import numpy as np

# 初始化识别器
recognizer = STTRecognizer()

# 准备音频数据（16kHz, float32）
audio_data = np.random.randn(16000).astype(np.float32)

# 执行识别
result = recognizer.recognize(audio_data)

if result.success:
    print(f"识别文本: {result.text}")
    print(f"置信度: {result.confidence:.3f}")
    print(f"语言: {result.language}")
else:
    print(f"识别失败: {result.error_message}")
```

### 3. 使用回调函数

```python
def on_recognition_result(result):
    print(f"收到识别结果: {result.text}")

# 设置回调
recognizer.set_callback(on_recognition_result)

# 识别时自动调用回调
recognizer.recognize(audio_data)
```

### 4. 异步识别

```python
def handle_result(result):
    print(f"异步识别完成: {result.text}")

# 提交异步识别任务
request_id = recognizer.recognize_async(
    audio_data,
    callback=handle_result
)

print(f"任务已提交: {request_id}")
```

## 配置说明

配置文件位置：`config/stt_config.yaml`

### 本地引擎配置

```yaml
stt:
  default_engine: local
  local:
    model_size: medium        # tiny/base/small/medium/large
    device: auto              # cpu/cuda/auto
    compute_type: float16     # float32/float16/int8
    language: auto            # zh/en/auto
    beam_size: 5
    temperature: 0.0
```

**模型大小选择：**

| 模型 | 参数量 | 显存需求 | 延迟 | 准确率 | 推荐场景 |
|-----|-------|---------|-----|--------|---------|
| tiny | 39M | ~1GB | ~200ms | 较低 | 测试/低配 |
| base | 74M | ~1GB | ~300ms | 一般 | 快速原型 |
| small | 244M | ~2GB | ~500ms | 良好 | 平衡性能 |
| **medium** | 769M | ~5GB | ~800ms | **优秀** | **推荐配置** |
| large | 1550M | ~10GB | ~1200ms | 最优 | 高配环境 |

### 云端引擎配置

```yaml
stt:
  cloud:
    provider: aliyun          # aliyun/tencent
    timeout: 5
    retry_count: 3
    
    aliyun:
      app_key: "your_app_key"
      access_key_id: "your_access_key_id"
      access_key_secret: "your_access_key_secret"
```

### 降级策略配置

```yaml
stt:
  fallback_enabled: true      # 启用降级
  default_engine: local       # 主引擎
  # 主引擎失败时自动切换到备用引擎
```

## API 参考

### STTRecognizer 类

#### 初始化

```python
STTRecognizer(config_path: Optional[str] = None)
```

**参数：**
- `config_path`: 配置文件路径（可选，默认 `config/stt_config.yaml`）

#### 同步识别

```python
recognize(
    audio_data: np.ndarray,
    sample_rate: int = 16000,
    **kwargs
) -> RecognitionResult
```

**参数：**
- `audio_data`: 音频数据（numpy array, float32, 一维）
- `sample_rate`: 采样率（默认 16000）
- `language`: 语言提示（可选，'zh'/'en'/'auto'）
- `speaker_id`: 说话人ID（可选）

**返回：** `RecognitionResult` 对象

#### 异步识别

```python
recognize_async(
    audio_data: np.ndarray,
    callback: Callable[[RecognitionResult], None],
    sample_rate: int = 16000,
    **kwargs
) -> str
```

**返回：** 请求ID（字符串）

#### 设置回调

```python
set_callback(callback: Callable[[RecognitionResult], None])
```

#### 切换引擎

```python
switch_engine(engine_type: str) -> bool
```

**参数：**
- `engine_type`: 'local' 或 'cloud'

**返回：** 是否切换成功

#### 获取统计信息

```python
get_statistics() -> dict
```

**返回：** 统计数据字典，包含：
- `total_requests`: 总请求数
- `successful_requests`: 成功数
- `failed_requests`: 失败数
- `success_rate`: 成功率
- `engine`: 引擎统计信息

#### 重置统计

```python
reset_statistics()
```

### RecognitionResult 类

识别结果对象包含以下字段：

| 字段 | 类型 | 说明 |
|-----|------|------|
| request_id | str | 请求唯一标识 |
| success | bool | 识别是否成功 |
| text | str | 识别的文本内容 |
| confidence | float | 置信度（0.0-1.0） |
| language | str | 检测到的语言 |
| duration | float | 音频时长（秒） |
| processing_time | float | 处理耗时（毫秒） |
| engine_type | str | 使用的引擎 |
| speaker_id | str | 说话人ID |
| segments | list | 分段结果（可选） |
| error_message | str | 错误信息（失败时） |

## 与其他模块集成

### 接收 VAD 模块输出

```python
from src.vad import VADDetector
from src.stt import STTRecognizer

# 初始化模块
vad = VADDetector()
stt = STTRecognizer()

# VAD 回调函数
def on_speech_segment(audio_segment, metadata):
    # 执行语音识别
    result = stt.recognize(
        audio_segment,
        timestamp=metadata['start_time']
    )
    
    if result.success:
        print(f"识别文本: {result.text}")

# 设置 VAD 回调
vad.set_callback(on_speech_segment)
```

### 结合说话人识别

```python
from src.speaker_recognition import SpeakerRecognizer

speaker_rec = SpeakerRecognizer()

def on_speech_with_speaker(audio_segment, metadata):
    # 先识别说话人
    speaker_result = speaker_rec.recognize(audio_segment)
    
    # 再进行语音识别
    stt_result = stt.recognize(
        audio_segment,
        speaker_id=speaker_result.speaker_id if speaker_result.matched else None
    )
    
    print(f"说话人: {speaker_result.speaker_id}")
    print(f"文本: {stt_result.text}")
```

## 模型管理

### 自动下载

首次运行时，模型会自动从 HuggingFace 下载到 `models/stt/faster-whisper/` 目录。

### 手动下载

如果网络不佳，可以手动下载模型：

1. 访问 https://huggingface.co/guillaumekln/faster-whisper-medium
2. 下载所有文件
3. 放置到 `models/stt/faster-whisper/medium/` 目录

### 模型路径结构

```
models/
└── stt/
    └── faster-whisper/
        ├── tiny/
        ├── base/
        ├── small/
        ├── medium/      # 推荐使用
        └── large/
```

## 性能优化

### GPU 加速

**CUDA 环境检查：**

```python
import torch
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"CUDA 版本: {torch.version.cuda}")
print(f"GPU 设备: {torch.cuda.get_device_name(0)}")
```

**配置 GPU：**

```yaml
stt:
  local:
    device: cuda          # 强制使用 GPU
    compute_type: float16 # 使用混合精度（节省显存）
```

### 降低延迟

```yaml
stt:
  local:
    model_size: small     # 使用小模型
    beam_size: 1          # 减小 beam size
    temperature: 0.0      # 贪婪解码
```

### 提高准确率

```yaml
stt:
  local:
    model_size: medium    # 使用中等或大模型
    beam_size: 5          # 增加 beam size
    language: zh          # 指定语言（不使用 auto）
```

## 错误处理

### 常见错误

| 错误信息 | 原因 | 解决方法 |
|---------|------|---------|
| 音频过短 | 时长 < 0.3s | 增加音频长度 |
| 音频过长 | 时长 > 30s | 分段处理 |
| 模型加载失败 | 模型文件损坏/缺失 | 重新下载模型 |
| CUDA out of memory | GPU 显存不足 | 使用小模型或切换 CPU |
| 未识别到文本 | 音频质量差/静音 | 检查音频内容 |

### 降级处理

```python
# 本地引擎失败会自动降级到云端
result = recognizer.recognize(audio_data)

if not result.success:
    print(f"识别失败: {result.error_message}")
    # 检查是否发生降级
    stats = recognizer.get_statistics()
    print(f"降级次数: {stats['engine']['fallback_count']}")
```

## 测试

运行测试用例：

```bash
python tests/test_stt.py
```

测试包含：
- 初始化测试
- 音频验证测试
- 基本识别测试
- 回调机制测试
- 统计功能测试
- 引擎切换测试

## 注意事项

1. **音频格式**：必须是 numpy.ndarray, float32, 一维数组
2. **采样率**：推荐 16000 Hz
3. **时长限制**：0.3s - 30s
4. **模型加载**：首次运行会较慢（加载模型）
5. **GPU 显存**：medium 模型需要约 5GB 显存
6. **云端 API**：需要配置有效的 API 凭证

## 性能指标

### 延迟（medium 模型）

- CPU (Intel i7): ~2000ms
- GPU (RTX 3060): ~800ms
- 云端 API: ~600ms

### 准确率

- 中文清晰语音: > 90%
- 英文清晰语音: > 92%
- 中英混说: > 85%
- 噪声环境: > 75%

## 常见问题

**Q: 如何提高识别速度？**

A: 使用 GPU、减小模型尺寸（small）、降低 beam_size

**Q: 如何提高识别准确率？**

A: 使用大模型（medium/large）、指定语言、增大 beam_size

**Q: GPU 显存不足怎么办？**

A: 使用 small 模型、设置 `compute_type: int8`、或切换 CPU

**Q: 云端识别如何收费？**

A: 阿里云按音频时长计费，腾讯云按调用次数计费，建议优先使用本地引擎

## 后续计划

- [ ] 支持流式识别（边说边转）
- [ ] 支持自定义热词
- [ ] 支持更多云服务商（AWS, Azure）
- [ ] 优化模型加载速度
- [ ] 支持多语言自动检测

## 相关链接

- [Faster Whisper GitHub](https://github.com/guillaumekln/faster-whisper)
- [OpenAI Whisper 论文](https://arxiv.org/abs/2212.04356)
- [阿里云实时语音识别](https://help.aliyun.com/document_detail/90727.html)
- [腾讯云一句话识别](https://cloud.tencent.com/document/product/1093/37308)

---

**版本**: v1.0  
**更新时间**: 2025-11-25
