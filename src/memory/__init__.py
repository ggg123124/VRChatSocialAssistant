"""
记忆与检索模块

该模块负责存储、管理和检索好友相关信息及对话历史，为 LLM 推理层提供上下文增强能力（RAG）。

主要组件:
- MemoryManager: 记忆管理器（核心协调者）
- ProfileStore: 好友档案存储
- ConversationStore: 对话记录存储
- VectorDatabase: 向量数据库封装
- EmbeddingService: 文本嵌入服务
- Retriever: 检索器
"""

__version__ = "0.1.0"

# 导出主要接口
from .memory_manager import MemoryManager
from .models import FriendProfile, Conversation, Memory, EventType, Personality

__all__ = [
    "MemoryManager",
    "FriendProfile",
    "Conversation",
    "Memory",
    "EventType",
    "Personality",
]
