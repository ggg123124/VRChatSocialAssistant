# VRChat 社交辅助工具

一个基于AI的VRChat社交辅助系统，通过实时语音识别、声纹识别和大语言模型，为用户提供智能社交提示。

## 项目概述

本项目旨在帮助用户在VRChat社交场景中更好地交流，通过实时分析系统音频和麦克风输入，识别说话人并提供智能对话建议。

### 核心功能

- 🎤 **实时音频采集**：支持WASAPI Loopback采集系统音频和麦克风输入
- 🗣️ **语音活动检测（VAD）**：基于Silero模型的实时语音片段检测
- 👤 **说话人识别**：通过声纹识别区分不同说话人
- 📝 **语音转文本（STT）**：实时语音识别，支持中英文混合
- 🧠 **智能对话辅助**：基于LLM的上下文理解和建议生成
- 💾 **记忆管理**：RAG向量检索，记住好友信息和对话历史
- 🥽 **VR显示**：基于OpenXR的头显HUD提示展示

## 技术架构

### 技术栈

| 组件 | 技术选型 | 说明 |
|------|---------|------|
| 音频处理 | PyAudio + pyaudiowpatch | WASAPI Loopback支持 |
| 语音检测 | Silero VAD | 轻量级、高精度VAD模型 |
| 语音识别 | faster-whisper | 本地部署的高效STT |
| 声纹识别 | pyannote.audio | ECAPA-TDNN声纹模型 |
| 向量数据库 | Chroma | 记忆存储和检索 |
| 大语言模型 | OpenAI API / 本地模型 | 对话理解和建议生成 |
| VR渲染 | OpenXR + pyopenvr | 跨平台VR显示 |

### 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    VR 头显 HUD 显示                      │
└─────────────────────────────────────────────────────────┘
                            ▲
                            │
┌─────────────────────────────────────────────────────────┐
│              提示生成 & 智能建议模块                      │
│  - LLM推理引擎  - 记忆检索  - 提示格式化                 │
└─────────────────────────────────────────────────────────┘
                            ▲
                            │
┌─────────────────────────────────────────────────────────┐
│                   语音处理模块                           │
│  - VAD检测  - 说话人识别  - 流式STT                      │
└─────────────────────────────────────────────────────────┘
                            ▲
                            │
┌─────────────────────────────────────────────────────────┐
│                   音频采集模块                           │
│  - WASAPI Loopback  - 麦克风输入  - 音频预处理           │
└─────────────────────────────────────────────────────────┘
```

## 项目进度

### ✅ 已完成

#### 1. 音频采集模块 (完成度: 100%)
- ✅ `DeviceManager`：音频设备管理和枚举
- ✅ `AudioCapturer`：双通道音频采集（Loopback + 麦克风）
- ✅ 自动重采样功能（48kHz → 16kHz）
- ✅ 回调机制和队列管理
- ✅ 完整的单元测试和集成测试

#### 2. VAD（语音活动检测）模块 (完成度: 100%)
- ✅ `SileroVAD`：基于Silero模型的VAD检测
- ✅ `AudioBuffer`：音频缓冲区管理
- ✅ `VADDetector`：语音片段检测和切分
- ✅ 状态机管理（IDLE → SPEECH → SILENCE）
- ✅ 语音片段输出回调
- ✅ 性能优化（处理延迟 <2ms，零丢帧）
- ✅ 完整的单元测试（18个测试用例全部通过）
- ✅ 与音频采集模块的集成测试

#### 3. 配置管理
- ✅ YAML配置文件支持
- ✅ 音频参数配置
- ✅ VAD参数配置
- ✅ 记忆模块参数配置

#### 4. 记忆管理模块 (完成度: 100%)
- ✅ `MemoryManager`：统一的记忆操作入口
- ✅ `ProfileStore`：好友档案管理（JSON存储）
- ✅ `ConversationStore`：对话记录存储（SQLite）
- ✅ `VectorDatabase`：向量数据库封装（Chroma后端）
- ✅ `EmbeddingService`：文本嵌入服务（bge-m3模型）
- ✅ `Retriever`：多策略检索器（语义/混合/时间衰减）
- ✅ 模型自动下载功能
- ✅ 完整的单元测试和演示代码

#### 5. 项目基础设施
- ✅ 项目结构设计
- ✅ 依赖管理（requirements.txt）
- ✅ Git版本控制
- ✅ 开发环境配置

#### 6. 说话人识别模块 (完成度: 100%)
- ✅ `EmbeddingEngine`：声纹特征提取（基于pyannote.audio）
- ✅ `MatchingEngine`：声纹匹配引擎（余弦相似度）
- ✅ `ProfileDatabase`：说话人档案管理（JSON存储）
- ✅ `SpeakerRecognizer`：统一的说话人识别接口
- ✅ 多说话人管理和档案持久化
- ✅ 阈值控制和相似度计算
- ✅ 完整的演示代码

#### 7. 语音转文本（STT）模块 (完成度: 100%)
- ✅ `LocalEngine`：基于faster-whisper的本地识别引擎
- ✅ `CloudEngine`：基于阿里云的云端识别引擎
- ✅ `EngineManager`：多引擎管理和切换
- ✅ `STTRecognizer`：统一的STT接口
- ✅ 流式识别支持
- ✅ 中英文混合识别
- ✅ 性能监控和错误处理
- ✅ 完整的演示代码

### 🚧 进行中

目前所有已开发模块均已完成并通过测试。

### 📋 待开发

#### 1. 智能对话辅助模块 (优先级: 高)
- ⏳ LLM集成（OpenAI API / 本地模型）
- ⏳ 上下文管理
- ⏳ 对话理解和分析
- ⏳ 建议生成策略

#### 2. 系统集成 (优先级: 高)
- ⏳ 主控制流程
- ⏳ 模块间通信（音频→VAD→STT→记忆→LLM）
- ⏳ 错误处理和恢复
- ⏳ 性能监控
- ⏳ 端到端测试

#### 3. VR显示模块 (优先级: 中)
- ⏳ OpenXR集成
- ⏳ HUD Overlay渲染
- ⏳ 手柄交互
- ⏳ 视觉优化

## 快速开始

### 环境要求

- Python 3.11
- Windows 10/11（WASAPI支持）
- 支持CUDA的GPU（可选，用于加速）

### 安装依赖

```bash
pip install -r requirements.txt
```

### 模型文件

项目使用的模型文件会在首次运行时自动下载到 `models/` 目录下，无需手动下载。

**VAD模型**: 首次运行时会从 PyTorch Hub 自动下载 Silero VAD 模型（约1-2MB）到 `models/vad/` 目录。

**嵌入模型**: 首次运行记忆管理模块时会从 ModelScope 自动下载 bge-m3 模型（约2GB）到 `models/embeddings/bge-m3/` 目录。

**声纹识别模型**: 首次运行说话人识别模块时会从 Hugging Face 自动下载 pyannote.audio 模型到 `models/speaker_embeddings/` 目录。

**STT模型**: 首次运行STT模块时会从 Hugging Face 自动下载 faster-whisper 模型（约1-3GB，取决于选择的模型大小）。

### 运行测试

#### 集成测试工具（推荐）
```bash
# Windows 用户可以直接运行
python tests/integrated_test.py
# 或使用启动脚本
run_integrated_test.bat

# 仅运行初始化检查
python tests/integrated_test.py --init

# 直接运行完整流程测试
python tests/integrated_test.py --full
```

集成测试工具提供了交互式命令行界面，可以：
- 🔍 自动检查环境并初始化所有模块
- 🧪 单独测试各个模块功能
- 🔗 测试端到端的语音处理流程
- 📊 管理好友档案和对话数据
- 📊 查看系统状态和统计信息

详细使用说明请查看: [tests/README_TEST.md](tests/README_TEST.md)

#### 模块单元测试

#### 测试音频采集
```bash
python tests/test_audio_capture.py
```

#### 测试VAD模块
```bash
python tests/test_vad.py
```

#### 测试集成功能
```bash
python tests/test_vad_integration.py
```

#### VAD功能演示
```bash
python tests/demo_vad.py
```

#### 测试记忆管理模块
```bash
python tests/test_memory_basic.py
```

#### 测试说话人识别模块
```bash
python tests/test_speaker_recognition.py
```

#### 说话人识别演示
```bash
python tests/demo_speaker_recognition.py
```

#### 测试STT模块
```bash
python tests/test_stt.py
```

## 项目结构

```
VRChatSocialAssistant/
├── config/                 # 配置文件
│   ├── audio_config.yaml  # 音频和VAD配置
│   └── memory_config.yaml # 记忆模块配置
├── doc/                   # 文档
│   └── 架构设计.md        # 架构设计文档
├── models/                # 模型文件目录
│   ├── vad/              # VAD模型存储目录
│   └── embeddings/       # 嵌入模型存储目录
├── src/                   # 源代码
│   ├── audio_capture/     # 音频采集模块
│   │   ├── device_manager.py
│   │   └── audio_capturer.py
│   ├── vad/              # VAD模块
│   │   ├── audio_buffer.py
│   │   ├── silero_vad.py
│   │   └── vad_detector.py
│   ├── speaker_recognition/  # 说话人识别模块
│   │   ├── embedding_engine.py
│   │   ├── matching_engine.py
│   │   ├── profile_database.py
│   │   └── speaker_recognizer.py
│   ├── stt/              # 语音转文本模块
│   │   ├── local_engine.py
│   │   ├── cloud_engine.py
│   │   ├── engine_manager.py
│   │   └── stt_recognizer.py
│   └── memory/           # 记忆管理模块
│       ├── memory_manager.py
│       ├── profile_store.py
│       ├── conversation_store.py
│       ├── vector_database.py
│       ├── embedding_service.py
│       └── retriever.py
├── tests/                # 测试代码
│   ├── integrated_test.py        # 集成测试主程序
│   ├── test_utils.py             # 测试工具函数
│   ├── README_TEST.md            # 测试说明文档
│   ├── test_data/                # 测试数据
│   │   ├── sample_profiles.json
│   │   └── sample_conversations.json
│   ├── test_audio_capture.py
│   ├── test_vad.py
│   ├── test_vad_integration.py
│   ├── demo_vad.py
│   ├── test_speaker_recognition.py
│   ├── demo_speaker_recognition.py
│   ├── test_stt.py
│   └── test_memory_basic.py
├── run_integrated_test.bat   # Windows 启动脚本
└── requirements.txt          # 项目依赖
```

## 测试结果

### VAD模块测试
- ✅ 单元测试：18/18 通过
- ✅ 集成测试：成功
- ✅ 性能指标：
  - 处理延迟：1.48ms（目标 <30ms）
  - 丢帧率：0%
  - 准确率：成功检测语音片段，置信度 0.798

### 音频采集测试
- ✅ WASAPI Loopback采集正常
- ✅ 自动重采样功能正常（48kHz → 16kHz）
- ✅ 零溢出、零丢帧

## 开发计划

### 近期目标（1-2周）
1. 实现智能对话辅助模块（LLM集成）
2. 完成模块间集成（音频→VAD→说话人识别→STT→记忆→LLM）
3. 端到端功能测试

### 中期目标（1-2月）
1. VR显示模块开发
2. 系统性能优化
3. 用户体验改进

### 长期目标（3-6月）
1. 完整的VR头显HUD集成
2. 多场景适配和优化
3. 稳定性和可靠性改进
4. 用户文档和教程完善

## 贡献指南

欢迎提交Issue和Pull Request！

## 许可证

本项目采用MIT许可证，详见 [LICENSE](LICENSE) 文件。

## 致谢

- [Silero VAD](https://github.com/snakers4/silero-vad) - 高性能VAD模型
- [pyaudiowpatch](https://github.com/s0d3s/PyAudioWPatch) - WASAPI Loopback支持
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - 高效语音识别
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) - 声纹识别
- [ChromaDB](https://github.com/chroma-core/chroma) - 向量数据库
- [BGE-M3](https://github.com/FlagOpen/FlagEmbedding) - 文本嵌入模型

---

**注意**：本项目仍在积极开发中，API可能会发生变化。
