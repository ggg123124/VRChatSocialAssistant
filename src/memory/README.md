# 记忆与检索模块

记忆与检索模块是 VRChat 社交辅助系统的核心智能组件，负责存储、管理和检索好友相关信息及对话历史。

## 功能特性

- ✅ **好友档案管理**：创建、读取、更新、删除好友档案
- ✅ **对话记录存储**：SQLite 数据库存储对话历史
- ✅ **语义检索**：基于文本嵌入的相似度检索
- ✅ **时间衰减**：优先检索近期对话
- ✅ **模型自动下载**：首次运行自动下载 bge-m3 嵌入模型
- ✅ **多种检索策略**：语义检索、混合检索、关键词检索

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 基础使用

```python
from src.memory import MemoryManager

# 初始化记忆管理器
manager = MemoryManager()

# 创建好友档案
friend_id = manager.create_friend_profile(
    name="小明",
    voice_profile_path="data/speaker_profiles/xiaoming.npy",
    preferences=["游戏", "动漫"],
    avoid_topics=["政治"]
)

# 添加对话记录
conversation_id = manager.add_conversation(
    friend_id=friend_id,
    transcript="我最近在玩VRChat，超好玩的！",
    speaker_id=friend_id,
    event_type="STATEMENT"
)

# 检索相关记忆
memories = manager.retrieve_memories(
    query="VRChat游戏",
    friend_id=friend_id,
    top_k=5
)

# 打印检索结果
for memory in memories:
    print(f"相似度: {memory.similarity_score:.2f}")
    print(f"内容: {memory.content}")
    print(f"时间: {memory.timestamp}")
    print("-" * 40)
```

### 3. 运行测试

```bash
# 基础功能测试
python tests/test_memory_basic.py
```

## 模块结构

```
src/memory/
├── __init__.py                 # 模块导出
├── memory_manager.py           # 记忆管理器（核心协调者）
├── profile_store.py            # 好友档案存储
├── conversation_store.py       # 对话记录存储
├── vector_database.py          # 向量数据库封装
├── embedding_service.py        # 文本嵌入服务
├── retriever.py                # 检索器
└── models.py                   # 数据模型定义
```

## 核心组件

### MemoryManager

统一的记忆操作入口，协调各子组件的交互。

**主要方法**：
- `create_friend_profile()`: 创建好友档案
- `add_conversation()`: 添加对话记录
- `retrieve_memories()`: 检索相关记忆
- `get_friend_profile()`: 获取好友档案
- `update_friend_profile()`: 更新好友档案
- `delete_all_memories()`: 删除记忆

### ProfileStore

好友档案存储，采用每个好友一个 JSON 文件的方式。

### ConversationStore

对话记录存储，使用 SQLite 数据库。

### VectorDatabase

向量数据库封装，使用 Chroma 作为后端。

### EmbeddingService

文本嵌入服务，使用 bge-m3 模型。

**模型管理**：
- 首次运行自动从 ModelScope 下载模型（约2GB）
- 模型保存在 `models/embeddings/bge-m3/`
- 支持 GPU 加速（CUDA）

### Retriever

检索器，支持多种检索策略：
- **语义检索**：基于向量相似度
- **混合检索**：语义 + 关键词
- **时间衰减检索**：优先检索近期对话

## 配置

配置文件位于 `config/memory_config.yaml`

```yaml
embedding:
  model_path: "models/embeddings/bge-m3"  # 模型本地路径
  device: "cuda"  # 运行设备（cuda/cpu）
  auto_download: true  # 自动下载模型
  download_mirror: "modelscope"  # 下载镜像源

retrieval:
  default_top_k: 5  # 默认检索数量
  similarity_threshold: 0.6  # 相似度阈值
  time_decay_lambda: 0.1  # 时间衰减系数
  default_strategy: "semantic"  # 默认检索策略
```

## 数据存储

```
data/
├── profiles/               # 好友档案（JSON文件）
├── conversations/          # 对话记录（SQLite数据库）
├── vector_db/              # 向量数据库（Chroma）
└── cache/                  # 临时缓存
```

## 性能指标

| 操作 | 目标延迟 |
|------|---------|
| 添加对话（含嵌入） | < 100ms |
| 检索 Top-5 记忆 | < 80ms |
| 获取好友档案 | < 20ms |

## 注意事项

1. **首次运行**：会自动下载 bge-m3 模型（约2GB），需要网络连接
2. **GPU 加速**：如果有 CUDA，会自动使用 GPU 加速嵌入计算
3. **数据持久化**：所有数据保存在本地，确保隐私安全
4. **磁盘空间**：预留至少 3.5GB 空间（模型 2GB + 数据 1.5GB）

## 常见问题

### Q: 模型下载失败怎么办？

A: 可以手动下载模型：
1. 访问 https://modelscope.cn/models/Xorbits/bge-m3
2. 下载所有文件到 `models/embeddings/bge-m3/` 目录
3. 确保包含 `config.json` 和 `pytorch_model.bin`

### Q: 如何切换到 CPU 模式？

A: 修改配置文件 `config/memory_config.yaml`：
```yaml
embedding:
  device: "cpu"
```

### Q: 如何清除所有数据？

A: 删除 `data/` 目录下的所有文件即可。

## 开发计划

- [x] Phase 1: 基础框架
- [ ] Phase 2: 检索优化
- [ ] Phase 3: 功能完善
- [ ] Phase 4: 隐私增强

## 许可证

本项目采用 MIT 许可证。
