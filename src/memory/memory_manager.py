"""
记忆管理器模块

统一的记忆操作入口，协调各子组件的交互，提供高层抽象接口。
"""

import logging
import uuid
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from pathlib import Path

import yaml

from .models import FriendProfile, Conversation, Memory, EventType, Personality
from .profile_store import ProfileStore, NotFoundError, StorageError
from .conversation_store import ConversationStore, ConversationStoreError
from .vector_database import VectorDatabase, VectorDatabaseError
from .embedding_service import EmbeddingService, EmbeddingServiceError
from .retriever import Retriever, RetrieverError


logger = logging.getLogger(__name__)


class MemoryManagerError(Exception):
    """记忆管理器异常基类"""
    pass


class MemoryManager:
    """
    记忆管理器
    
    提供统一的记忆操作接口，协调各子组件：
    - ProfileStore: 好友档案管理
    - ConversationStore: 对话记录存储
    - VectorDatabase: 向量检索
    - EmbeddingService: 文本嵌入
    - Retriever: 检索策略
    """
    
    def __init__(self, config_path: str = "config/memory_config.yaml"):
        """
        初始化记忆管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # 初始化各组件
        self._initialize_components()
        
        logger.info("MemoryManager initialized successfully")
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                logger.warning(f"Config file not found: {self.config_path}, using defaults")
                return self._get_default_config()
            
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            logger.info(f"Config loaded from: {self.config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load config: {e}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'storage': {
                'profiles_dir': 'data/profiles',
                'conversations_db': 'data/conversations/conversations.db',
                'vector_db_dir': 'data/vector_db',
                'backup_dir': 'data/backups',
                'enable_encryption': False
            },
            'embedding': {
                'model_name': 'BAAI/bge-m3',
                'model_path': 'models/embeddings/bge-m3',
                'device': 'cuda',
                'batch_size': 32,
                'max_length': 512,
                'auto_download': True,
                'download_mirror': 'modelscope'
            },
            'vector_db': {
                'backend': 'chroma',
                'persist_interval': 300,
                'dimension': None
            },
            'retrieval': {
                'default_top_k': 5,
                'similarity_threshold': 0.6,
                'time_decay_lambda': 0.1,
                'enable_time_decay': True,
                'default_strategy': 'semantic'
            },
            'cache': {
                'enable_profile_cache': True,
                'max_cache_size': 100
            }
        }
    
    def _initialize_components(self):
        """初始化各子组件"""
        try:
            # ProfileStore
            self.profile_store = ProfileStore(
                profiles_dir=self.config['storage']['profiles_dir'],
                enable_cache=self.config['cache']['enable_profile_cache'],
                max_cache_size=self.config['cache']['max_cache_size']
            )
            
            # ConversationStore
            self.conversation_store = ConversationStore(
                db_path=self.config['storage']['conversations_db']
            )
            
            # EmbeddingService
            self.embedding_service = EmbeddingService(
                model_path=self.config['embedding']['model_path'],
                model_name=self.config['embedding']['model_name'],
                device=self.config['embedding']['device'],
                batch_size=self.config['embedding']['batch_size'],
                max_length=self.config['embedding']['max_length'],
                auto_download=self.config['embedding']['auto_download'],
                download_mirror=self.config['embedding']['download_mirror']
            )
            
            # VectorDatabase
            self.vector_db = VectorDatabase(
                persist_directory=self.config['storage']['vector_db_dir'],
                embedding_dimension=self.embedding_service.get_dimension()
            )
            
            # Retriever
            self.retriever = Retriever(
                vector_db=self.vector_db,
                embedding_service=self.embedding_service,
                profile_store=self.profile_store,
                default_top_k=self.config['retrieval']['default_top_k'],
                similarity_threshold=self.config['retrieval']['similarity_threshold'],
                time_decay_lambda=self.config['retrieval']['time_decay_lambda']
            )
            
            logger.info("All components initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise MemoryManagerError(f"Initialization failed: {e}")
    
    def add_conversation(
        self,
        friend_id: str,
        transcript: str,
        speaker_id: str,
        timestamp: Optional[datetime] = None,
        event_type: str = "STATEMENT",
        confidence: float = 0.9
    ) -> str:
        """
        添加新对话记录
        
        Args:
            friend_id: 好友唯一标识符
            transcript: 识别的对话文本
            speaker_id: 说话人ID
            timestamp: 对话发生时间（默认当前时间）
            event_type: 对话事件类型
            confidence: STT识别置信度
            
        Returns:
            对话记录的唯一ID
            
        Raises:
            MemoryManagerError: 添加失败
        """
        try:
            # 生成ID
            conversation_id = str(uuid.uuid4())
            vector_id = f"vec_{conversation_id}"
            
            if timestamp is None:
                timestamp = datetime.now()
            
            # 创建对话记录
            conversation = Conversation(
                id=conversation_id,
                friend_id=friend_id,
                timestamp=timestamp,
                speaker_id=speaker_id,
                transcript=transcript,
                event_type=EventType(event_type),
                confidence=confidence,
                vector_id=vector_id
            )
            
            # 生成嵌入向量
            embedding = self.embedding_service.encode(transcript)
            
            # 保存到对话存储
            self.conversation_store.add_conversation(conversation)
            
            # 添加到向量数据库
            metadata = {
                'friend_id': friend_id,
                'speaker_id': speaker_id,
                'timestamp': timestamp.isoformat(),
                'event_type': event_type,
                'conversation_id': conversation_id,
                'confidence': confidence
            }
            
            self.vector_db.add_vectors(
                ids=[vector_id],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[transcript]
            )
            
            # 更新好友档案（对话次数、最后见面时间）
            try:
                profile = self.profile_store.get_profile(friend_id)
                self.profile_store.update_profile(friend_id, {
                    'conversation_count': profile.conversation_count + 1,
                    'last_seen': timestamp
                })
            except NotFoundError:
                logger.warning(f"Friend profile not found: {friend_id}")
            
            logger.info(f"Conversation added: {conversation_id} (friend={friend_id})")
            return conversation_id
            
        except Exception as e:
            logger.error(f"Failed to add conversation: {e}")
            raise MemoryManagerError(f"Failed to add conversation: {e}")
    
    def retrieve_memories(
        self,
        query: str,
        friend_id: Optional[str] = None,
        top_k: Optional[int] = None,
        time_decay: bool = True,
        strategy: Optional[str] = None
    ) -> List[Memory]:
        """
        检索相关记忆
        
        Args:
            query: 查询文本
            friend_id: 限定好友ID
            top_k: 返回记忆数量
            time_decay: 是否应用时间衰减
            strategy: 检索策略
            
        Returns:
            记忆片段列表
            
        Raises:
            MemoryManagerError: 检索失败
        """
        try:
            if top_k is None:
                top_k = self.config['retrieval']['default_top_k']
            
            if strategy is None:
                strategy = self.config['retrieval']['default_strategy']
            
            if time_decay is None:
                time_decay = self.config['retrieval']['enable_time_decay']
            
            memories = self.retriever.retrieve(
                query=query,
                friend_id=friend_id,
                top_k=top_k,
                strategy=strategy,
                apply_time_decay=time_decay
            )
            
            logger.debug(f"Retrieved {len(memories)} memories for query: {query[:50]}...")
            return memories
            
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            raise MemoryManagerError(f"Failed to retrieve memories: {e}")
    
    def create_friend_profile(
        self,
        name: str,
        voice_profile_path: str,
        preferences: Optional[List[str]] = None,
        avoid_topics: Optional[List[str]] = None,
        personality: str = "随和",
        language_preference: str = "zh"
    ) -> str:
        """
        创建新好友档案
        
        Args:
            name: 好友昵称
            voice_profile_path: 声纹文件路径
            preferences: 兴趣偏好列表
            avoid_topics: 禁忌话题列表
            personality: 性格标签
            language_preference: 语言偏好
            
        Returns:
            新创建的好友ID
            
        Raises:
            MemoryManagerError: 创建失败
        """
        try:
            friend_id = str(uuid.uuid4())
            
            profile = FriendProfile(
                id=friend_id,
                name=name,
                voice_profile_path=voice_profile_path,
                preferences=preferences or [],
                avoid_topics=avoid_topics or [],
                personality=Personality(personality),
                language_preference=language_preference
            )
            
            self.profile_store.create_profile(profile)
            
            logger.info(f"Friend profile created: {friend_id} (name={name})")
            return friend_id
            
        except Exception as e:
            logger.error(f"Failed to create friend profile: {e}")
            raise MemoryManagerError(f"Failed to create friend profile: {e}")
    
    def get_friend_profile(self, friend_id: str) -> FriendProfile:
        """
        获取好友档案
        
        Args:
            friend_id: 好友唯一标识符
            
        Returns:
            FriendProfile 对象
            
        Raises:
            MemoryManagerError: 好友不存在或读取失败
        """
        try:
            return self.profile_store.get_profile(friend_id)
        except NotFoundError as e:
            raise MemoryManagerError(f"Friend not found: {friend_id}")
        except Exception as e:
            logger.error(f"Failed to get friend profile: {e}")
            raise MemoryManagerError(f"Failed to get friend profile: {e}")
    
    def update_friend_profile(self, friend_id: str, updates: Dict[str, Any]) -> bool:
        """
        更新好友档案属性
        
        Args:
            friend_id: 好友唯一标识符
            updates: 更新字段字典
            
        Returns:
            更新是否成功
            
        Raises:
            MemoryManagerError: 更新失败
        """
        try:
            return self.profile_store.update_profile(friend_id, updates)
        except Exception as e:
            logger.error(f"Failed to update friend profile: {e}")
            raise MemoryManagerError(f"Failed to update friend profile: {e}")
    
    def delete_all_memories(self, friend_id: Optional[str] = None, confirm: bool = False) -> bool:
        """
        删除记忆（隐私保护）
        
        Args:
            friend_id: 好友ID（为空则删除所有）
            confirm: 确认删除
            
        Returns:
            删除是否成功
            
        Raises:
            MemoryManagerError: 删除失败
        """
        if not confirm:
            raise MemoryManagerError("Must confirm deletion by setting confirm=True")
        
        try:
            if friend_id:
                # 删除指定好友的记忆
                self.conversation_store.delete_conversations_by_friend(friend_id)
                self.vector_db.delete_by_filter({'friend_id': friend_id})
                logger.info(f"Deleted all memories for friend: {friend_id}")
            else:
                # 删除所有记忆（危险操作！）
                logger.warning("Deleting ALL memories!")
                self.vector_db.reset()
                # 注意：这里没有清空 ConversationStore，需要手动处理
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete memories: {e}")
            raise MemoryManagerError(f"Failed to delete memories: {e}")
    
    def generate_conversation_summary(
        self,
        friend_id: str,
        time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> str:
        """
        生成对话摘要
        
        Args:
            friend_id: 好友ID
            time_range: 时间范围（起始时间，结束时间）
            
        Returns:
            对话摘要文本
        """
        try:
            # 获取对话记录
            start_time, end_time = time_range if time_range else (None, None)
            conversations = self.conversation_store.get_conversations_by_friend(
                friend_id=friend_id,
                limit=100,
                start_time=start_time,
                end_time=end_time
            )
            
            if not conversations:
                return "暂无对话记录"
            
            # 简单摘要（实际应调用 LLM）
            summary_lines = [
                f"对话数量: {len(conversations)}",
                f"时间跨度: {conversations[-1].timestamp.strftime('%Y-%m-%d')} 至 {conversations[0].timestamp.strftime('%Y-%m-%d')}",
                f"主要话题: [待LLM生成]"
            ]
            
            return "\n".join(summary_lines)
            
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return f"生成摘要失败: {e}"
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        try:
            return {
                'total_friends': self.profile_store.count(),
                'total_conversations': self.conversation_store.count_total(),
                'total_vectors': self.vector_db.count(),
                'embedding_dimension': self.embedding_service.get_dimension()
            }
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
