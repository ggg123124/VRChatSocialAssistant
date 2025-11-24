"""
向量数据库封装模块

使用 Chroma 作为向量数据库后端，提供向量的插入、检索、删除等操作。
"""

import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

from .models import SearchResult


logger = logging.getLogger(__name__)


class VectorDatabaseError(Exception):
    """向量数据库异常基类"""
    pass


class VectorDatabase:
    """
    向量数据库封装
    
    使用 Chroma 作为后端，支持向量的持久化存储和相似度检索
    """
    
    def __init__(
        self,
        persist_directory: str = "data/vector_db",
        collection_name: str = "conversations",
        embedding_dimension: Optional[int] = None
    ):
        """
        初始化向量数据库
        
        Args:
            persist_directory: 持久化目录
            collection_name: 集合名称
            embedding_dimension: 向量维度（可选）
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        
        self.client = None
        self.collection = None
        
        logger.info(f"VectorDatabase initializing: dir={persist_directory}, collection={collection_name}")
        
        # 初始化数据库
        self._initialize()
    
    def _initialize(self):
        """初始化 Chroma 客户端和集合"""
        try:
            import chromadb
            from chromadb.config import Settings
            
            # 确保目录存在
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            
            # 创建持久化客户端
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # 获取或创建集合
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Conversation memories"}
            )
            
            logger.info(f"VectorDatabase initialized. Collection: {self.collection_name}, Count: {self.collection.count()}")
            
        except ImportError:
            raise VectorDatabaseError(
                "chromadb not installed. Please install: pip install chromadb"
            )
        except Exception as e:
            logger.error(f"Failed to initialize vector database: {e}")
            raise VectorDatabaseError(f"Failed to initialize: {e}")
    
    def add_vectors(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: List[str]
    ) -> None:
        """
        批量插入向量
        
        Args:
            ids: 向量ID列表
            embeddings: 嵌入向量列表
            metadatas: 元数据列表
            documents: 原始文档列表
            
        Raises:
            VectorDatabaseError: 插入失败
        """
        if not ids:
            logger.warning("No vectors to add")
            return
        
        if len(ids) != len(embeddings) != len(metadatas) != len(documents):
            raise VectorDatabaseError("Length mismatch between ids, embeddings, metadatas, and documents")
        
        try:
            # 转换元数据中的数值为字符串（Chroma 要求）
            processed_metadatas = []
            for metadata in metadatas:
                processed_meta = {}
                for key, value in metadata.items():
                    # Chroma 不支持嵌套字典，需要转换
                    if isinstance(value, (int, float)):
                        processed_meta[key] = value
                    else:
                        processed_meta[key] = str(value)
                processed_metadatas.append(processed_meta)
            
            # 添加向量
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=processed_metadatas,
                documents=documents
            )
            
            logger.debug(f"Added {len(ids)} vectors to collection")
            
        except Exception as e:
            logger.error(f"Failed to add vectors: {e}")
            raise VectorDatabaseError(f"Failed to add vectors: {e}")
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        相似度检索
        
        Args:
            query_embedding: 查询向量
            top_k: 返回数量
            filter: 元数据过滤条件（可选）
            
        Returns:
            检索结果列表
            
        Raises:
            VectorDatabaseError: 检索失败
        """
        try:
            # 执行查询
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filter if filter else None
            )
            
            # 转换结果
            search_results = []
            if results and results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    search_result = SearchResult(
                        id=results['ids'][0][i],
                        distance=results['distances'][0][i] if results.get('distances') else 0.0,
                        metadata=results['metadatas'][0][i] if results.get('metadatas') else {},
                        document=results['documents'][0][i] if results.get('documents') else ""
                    )
                    search_results.append(search_result)
            
            logger.debug(f"Search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Failed to search vectors: {e}")
            raise VectorDatabaseError(f"Failed to search: {e}")
    
    def delete_by_id(self, ids: List[str]) -> None:
        """
        删除指定向量
        
        Args:
            ids: 向量ID列表
            
        Raises:
            VectorDatabaseError: 删除失败
        """
        if not ids:
            logger.warning("No IDs to delete")
            return
        
        try:
            self.collection.delete(ids=ids)
            logger.debug(f"Deleted {len(ids)} vectors")
            
        except Exception as e:
            logger.error(f"Failed to delete vectors: {e}")
            raise VectorDatabaseError(f"Failed to delete vectors: {e}")
    
    def delete_by_filter(self, filter: Dict[str, Any]) -> int:
        """
        按条件删除向量
        
        Args:
            filter: 元数据过滤条件
            
        Returns:
            删除的向量数量
            
        Raises:
            VectorDatabaseError: 删除失败
        """
        try:
            # 先查询符合条件的向量
            results = self.collection.get(where=filter)
            
            if results and results['ids']:
                ids_to_delete = results['ids']
                self.collection.delete(ids=ids_to_delete)
                deleted_count = len(ids_to_delete)
                logger.info(f"Deleted {deleted_count} vectors by filter")
                return deleted_count
            else:
                logger.debug("No vectors matched the filter")
                return 0
            
        except Exception as e:
            logger.error(f"Failed to delete by filter: {e}")
            raise VectorDatabaseError(f"Failed to delete by filter: {e}")
    
    def count(self, filter: Optional[Dict[str, Any]] = None) -> int:
        """
        统计向量数量
        
        Args:
            filter: 元数据过滤条件（可选）
            
        Returns:
            向量数量
        """
        try:
            if filter:
                results = self.collection.get(where=filter)
                return len(results['ids']) if results and results['ids'] else 0
            else:
                return self.collection.count()
            
        except Exception as e:
            logger.error(f"Failed to count vectors: {e}")
            return 0
    
    def get_by_id(self, id: str) -> Optional[Dict[str, Any]]:
        """
        根据ID获取向量信息
        
        Args:
            id: 向量ID
            
        Returns:
            向量信息字典，不存在返回 None
        """
        try:
            results = self.collection.get(ids=[id])
            
            if results and results['ids'] and results['ids'][0]:
                return {
                    'id': results['ids'][0],
                    'embedding': results['embeddings'][0] if results.get('embeddings') else None,
                    'metadata': results['metadatas'][0] if results.get('metadatas') else {},
                    'document': results['documents'][0] if results.get('documents') else ""
                }
            else:
                return None
            
        except Exception as e:
            logger.error(f"Failed to get vector by ID: {e}")
            return None
    
    def reset(self) -> None:
        """
        重置数据库（删除所有数据）
        
        警告：此操作不可恢复！
        """
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Conversation memories"}
            )
            logger.warning("Vector database reset completed")
            
        except Exception as e:
            logger.error(f"Failed to reset database: {e}")
            raise VectorDatabaseError(f"Failed to reset: {e}")
