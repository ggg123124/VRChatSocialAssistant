"""
检索器模块

实现多种检索策略，包括语义检索、混合检索、时间衰减等。
"""

import logging
import math
from typing import List, Optional, Dict, Any
from datetime import datetime

from .models import Memory, EventType, SearchResult
from .vector_database import VectorDatabase
from .embedding_service import EmbeddingService
from .profile_store import ProfileStore


logger = logging.getLogger(__name__)


class RetrieverError(Exception):
    """检索器异常基类"""
    pass


class Retriever:
    """
    记忆检索器
    
    提供多种检索策略：
    - 语义检索（默认）
    - 混合检索（语义 + 关键词）
    - 时间衰减检索
    - 好友过滤检索
    """
    
    def __init__(
        self,
        vector_db: VectorDatabase,
        embedding_service: EmbeddingService,
        profile_store: ProfileStore,
        default_top_k: int = 5,
        similarity_threshold: float = 0.6,
        time_decay_lambda: float = 0.1
    ):
        """
        初始化检索器
        
        Args:
            vector_db: 向量数据库
            embedding_service: 嵌入服务
            profile_store: 档案存储
            default_top_k: 默认检索数量
            similarity_threshold: 相似度阈值
            time_decay_lambda: 时间衰减系数
        """
        self.vector_db = vector_db
        self.embedding_service = embedding_service
        self.profile_store = profile_store
        self.default_top_k = default_top_k
        self.similarity_threshold = similarity_threshold
        self.time_decay_lambda = time_decay_lambda
        
        logger.info("Retriever initialized")
    
    def retrieve(
        self,
        query: str,
        friend_id: Optional[str] = None,
        top_k: Optional[int] = None,
        strategy: str = "semantic",
        apply_time_decay: bool = True
    ) -> List[Memory]:
        """
        通用检索接口
        
        Args:
            query: 查询文本
            friend_id: 限定好友ID（可选）
            top_k: 返回记忆数量（可选）
            strategy: 检索策略（semantic/hybrid/keyword）
            apply_time_decay: 是否应用时间衰减
            
        Returns:
            记忆片段列表
            
        Raises:
            RetrieverError: 检索失败
        """
        if top_k is None:
            top_k = self.default_top_k
        
        # 根据策略选择检索方法
        if strategy == "semantic":
            return self._semantic_search(query, friend_id, top_k, apply_time_decay)
        elif strategy == "hybrid":
            return self._hybrid_search(query, friend_id, top_k, apply_time_decay)
        elif strategy == "keyword":
            return self._keyword_search(query, friend_id, top_k)
        else:
            raise RetrieverError(f"Unknown strategy: {strategy}")
    
    def _semantic_search(
        self,
        query: str,
        friend_id: Optional[str],
        top_k: int,
        apply_time_decay: bool
    ) -> List[Memory]:
        """
        语义检索
        
        Args:
            query: 查询文本
            friend_id: 限定好友ID
            top_k: 返回数量
            apply_time_decay: 是否应用时间衰减
            
        Returns:
            记忆片段列表
        """
        try:
            # 生成查询向量
            query_embedding = self.embedding_service.encode(query)
            
            # 构建过滤条件
            filter_condition = None
            if friend_id:
                filter_condition = {"friend_id": friend_id}
            
            # 向量检索
            search_results = self.vector_db.search(
                query_embedding=query_embedding,
                top_k=top_k * 2,  # 多取一些，应用过滤后可能不足
                filter=filter_condition
            )
            
            # 转换为 Memory 对象
            memories = self._convert_to_memories(search_results, apply_time_decay)
            
            # 按最终得分排序并过滤低分结果
            memories = [m for m in memories if m.similarity_score >= self.similarity_threshold]
            memories.sort(key=lambda x: x.final_score, reverse=True)
            
            return memories[:top_k]
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            raise RetrieverError(f"Semantic search failed: {e}")
    
    def _hybrid_search(
        self,
        query: str,
        friend_id: Optional[str],
        top_k: int,
        apply_time_decay: bool
    ) -> List[Memory]:
        """
        混合检索（语义 + 关键词）
        
        Args:
            query: 查询文本
            friend_id: 限定好友ID
            top_k: 返回数量
            apply_time_decay: 是否应用时间衰减
            
        Returns:
            记忆片段列表
        """
        # 先进行语义检索
        semantic_results = self._semantic_search(query, friend_id, top_k * 2, apply_time_decay)
        
        # 提取查询关键词
        keywords = self._extract_keywords(query)
        
        # 计算关键词匹配得分
        for memory in semantic_results:
            keyword_score = self._calculate_keyword_score(memory.content, keywords)
            # 混合得分：70% 语义 + 30% 关键词
            memory.final_score = 0.7 * memory.final_score + 0.3 * keyword_score
        
        # 重新排序
        semantic_results.sort(key=lambda x: x.final_score, reverse=True)
        
        return semantic_results[:top_k]
    
    def _keyword_search(
        self,
        query: str,
        friend_id: Optional[str],
        top_k: int
    ) -> List[Memory]:
        """
        关键词检索（降级方案）
        
        注意：此方法在向量检索不可用时使用，需要从 ConversationStore 查询
        """
        # 这是一个简化实现，实际应该从 ConversationStore 查询
        logger.warning("Keyword search is a fallback method with limited functionality")
        return []
    
    def retrieve_recent(
        self,
        friend_id: str,
        days: int = 7,
        limit: int = 10
    ) -> List[Memory]:
        """
        检索最近N天的记忆
        
        Args:
            friend_id: 好友ID
            days: 天数
            limit: 返回数量
            
        Returns:
            记忆片段列表
        """
        try:
            # 计算时间戳范围
            now = datetime.now()
            cutoff_timestamp = now.timestamp() - (days * 24 * 3600)
            
            # 从向量数据库检索（按时间过滤）
            # 注意：Chroma 的元数据过滤有限，这里做简化处理
            filter_condition = {"friend_id": friend_id}
            
            search_results = self.vector_db.search(
                query_embedding=[0.0] * self.embedding_service.get_dimension(),  # 占位符
                top_k=limit * 2,
                filter=filter_condition
            )
            
            # 转换并过滤时间
            memories = self._convert_to_memories(search_results, apply_time_decay=False)
            memories = [m for m in memories if m.timestamp.timestamp() >= cutoff_timestamp]
            
            # 按时间排序
            memories.sort(key=lambda x: x.timestamp, reverse=True)
            
            return memories[:limit]
            
        except Exception as e:
            logger.error(f"Failed to retrieve recent memories: {e}")
            return []
    
    def retrieve_by_keyword(
        self,
        keywords: List[str],
        friend_id: Optional[str] = None
    ) -> List[Memory]:
        """
        关键词检索
        
        Args:
            keywords: 关键词列表
            friend_id: 限定好友ID
            
        Returns:
            记忆片段列表
        """
        # 简化实现：使用关键词组合作为查询
        query = " ".join(keywords)
        return self._hybrid_search(query, friend_id, self.default_top_k, apply_time_decay=True)
    
    def _convert_to_memories(
        self,
        search_results: List[SearchResult],
        apply_time_decay: bool
    ) -> List[Memory]:
        """
        将搜索结果转换为 Memory 对象
        
        Args:
            search_results: 搜索结果列表
            apply_time_decay: 是否应用时间衰减
            
        Returns:
            记忆片段列表
        """
        memories = []
        
        for result in search_results:
            try:
                # 从元数据中提取信息
                metadata = result.metadata
                friend_id = metadata.get('friend_id', '')
                
                # 获取好友名称
                friend_name = ""
                try:
                    profile = self.profile_store.get_profile(friend_id)
                    friend_name = profile.name
                except Exception:
                    friend_name = "Unknown"
                
                # 解析时间戳
                timestamp_str = metadata.get('timestamp', '')
                if isinstance(timestamp_str, str):
                    timestamp = datetime.fromisoformat(timestamp_str)
                elif isinstance(timestamp_str, (int, float)):
                    timestamp = datetime.fromtimestamp(timestamp_str)
                else:
                    timestamp = datetime.now()
                
                # 计算相似度（距离转换为相似度）
                # Chroma 返回的是 L2 距离，需要转换
                similarity_score = 1.0 / (1.0 + result.distance)
                
                # 计算时间衰减
                time_decay_factor = 1.0
                if apply_time_decay:
                    time_decay_factor = self._calculate_time_decay(timestamp)
                
                # 计算最终得分
                final_score = similarity_score * time_decay_factor
                
                # 创建 Memory 对象
                memory = Memory(
                    id=result.id,
                    content=result.document,
                    friend_id=friend_id,
                    friend_name=friend_name,
                    timestamp=timestamp,
                    event_type=EventType(metadata.get('event_type', 'STATEMENT')),
                    similarity_score=similarity_score,
                    time_decay_factor=time_decay_factor,
                    final_score=final_score
                )
                
                memories.append(memory)
                
            except Exception as e:
                logger.warning(f"Failed to convert search result to memory: {e}")
                continue
        
        return memories
    
    def _calculate_time_decay(self, timestamp: datetime) -> float:
        """
        计算时间衰减系数
        
        使用公式: decay_factor = exp(-λ × 时间差（天）)
        
        Args:
            timestamp: 时间戳
            
        Returns:
            衰减系数（0-1）
        """
        now = datetime.now()
        time_diff_days = (now - timestamp).total_seconds() / (24 * 3600)
        
        # 指数衰减
        decay_factor = math.exp(-self.time_decay_lambda * time_diff_days)
        
        return decay_factor
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        提取关键词（简化版）
        
        实际应使用更复杂的 NLP 方法
        
        Args:
            text: 文本
            
        Returns:
            关键词列表
        """
        # 简单分词（按空格）
        words = text.split()
        
        # 过滤停用词（简化）
        stopwords = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        keywords = [w for w in words if w not in stopwords and len(w) > 1]
        
        return keywords[:5]  # 最多返回5个关键词
    
    def _calculate_keyword_score(self, content: str, keywords: List[str]) -> float:
        """
        计算关键词匹配得分
        
        Args:
            content: 内容文本
            keywords: 关键词列表
            
        Returns:
            得分（0-1）
        """
        if not keywords:
            return 0.0
        
        content_lower = content.lower()
        matched_count = sum(1 for kw in keywords if kw.lower() in content_lower)
        
        return matched_count / len(keywords)
