"""
对话记录存储模块

使用 SQLite 数据库存储对话记录，支持复杂查询和时间过滤。
"""

import sqlite3
import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import contextmanager

from .models import Conversation


logger = logging.getLogger(__name__)


class ConversationStoreError(Exception):
    """对话存储异常基类"""
    pass


class ConversationStore:
    """
    对话记录存储管理器
    
    使用 SQLite 数据库存储对话记录
    支持按时间范围、好友ID查询，定期清理过期记录
    """
    
    # 数据库表结构
    CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY,
        friend_id TEXT NOT NULL,
        timestamp DATETIME NOT NULL,
        speaker_id TEXT NOT NULL,
        transcript TEXT NOT NULL,
        event_type TEXT NOT NULL,
        confidence REAL NOT NULL,
        vector_id TEXT UNIQUE,
        summary TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """
    
    # 创建索引
    CREATE_INDEXES_SQL = [
        "CREATE INDEX IF NOT EXISTS idx_friend_id ON conversations(friend_id);",
        "CREATE INDEX IF NOT EXISTS idx_timestamp ON conversations(timestamp);",
        "CREATE INDEX IF NOT EXISTS idx_vector_id ON conversations(vector_id);",
    ]
    
    def __init__(self, db_path: str = "data/conversations/conversations.db"):
        """
        初始化对话存储
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = Path(db_path)
        
        # 确保目录存在
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 初始化数据库
        self._initialize_database()
        
        logger.info(f"ConversationStore initialized: db={db_path}")
    
    def _initialize_database(self):
        """初始化数据库表和索引"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 创建表
            cursor.execute(self.CREATE_TABLE_SQL)
            
            # 创建索引
            for index_sql in self.CREATE_INDEXES_SQL:
                cursor.execute(index_sql)
            
            conn.commit()
            logger.debug("Database initialized")
    
    @contextmanager
    def _get_connection(self):
        """获取数据库连接（上下文管理器）"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # 启用字典式访问
        try:
            yield conn
        finally:
            conn.close()
    
    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """将数据库行转换为字典"""
        return dict(row)
    
    def add_conversation(self, conversation: Conversation) -> str:
        """
        添加新的对话记录
        
        Args:
            conversation: 对话记录对象
            
        Returns:
            对话记录ID
            
        Raises:
            ConversationStoreError: 添加失败
        """
        sql = """
        INSERT INTO conversations 
        (id, friend_id, timestamp, speaker_id, transcript, event_type, confidence, vector_id, summary)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # 处理 event_type：如果是枚举类型则取值，否则直接使用
                event_type_value = conversation.event_type.value if hasattr(conversation.event_type, 'value') else conversation.event_type
                cursor.execute(sql, (
                    conversation.id,
                    conversation.friend_id,
                    conversation.timestamp.isoformat(),
                    conversation.speaker_id,
                    conversation.transcript,
                    event_type_value,
                    conversation.confidence,
                    conversation.vector_id,
                    conversation.summary
                ))
                conn.commit()
            
            logger.debug(f"Conversation added: {conversation.id}")
            return conversation.id
            
        except sqlite3.IntegrityError as e:
            logger.error(f"Failed to add conversation (integrity error): {e}")
            raise ConversationStoreError(f"Conversation ID already exists: {conversation.id}")
        except Exception as e:
            logger.error(f"Failed to add conversation: {e}")
            raise ConversationStoreError(f"Failed to add conversation: {e}")
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """
        获取单条对话记录
        
        Args:
            conversation_id: 对话记录ID
            
        Returns:
            对话记录对象，不存在返回 None
        """
        sql = "SELECT * FROM conversations WHERE id = ?"
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, (conversation_id,))
                row = cursor.fetchone()
            
            if row is None:
                return None
            
            return Conversation.from_dict(self._row_to_dict(row))
            
        except Exception as e:
            logger.error(f"Failed to get conversation {conversation_id}: {e}")
            return None
    
    def get_conversations_by_friend(
        self, 
        friend_id: str, 
        limit: int = 50,
        offset: int = 0,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Conversation]:
        """
        获取指定好友的对话记录
        
        Args:
            friend_id: 好友ID
            limit: 返回数量限制
            offset: 偏移量
            start_time: 起始时间（可选）
            end_time: 结束时间（可选）
            
        Returns:
            对话记录列表
        """
        sql = "SELECT * FROM conversations WHERE friend_id = ?"
        params = [friend_id]
        
        # 添加时间过滤
        if start_time:
            sql += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        if end_time:
            sql += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        
        # 排序和分页
        sql += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, params)
                rows = cursor.fetchall()
            
            return [Conversation.from_dict(self._row_to_dict(row)) for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to get conversations for friend {friend_id}: {e}")
            return []
    
    def get_recent_conversations(self, days: int = 7, limit: int = 100) -> List[Conversation]:
        """
        获取最近N天的对话记录
        
        Args:
            days: 天数
            limit: 返回数量限制
            
        Returns:
            对话记录列表
        """
        start_time = datetime.now() - timedelta(days=days)
        sql = """
        SELECT * FROM conversations 
        WHERE timestamp >= ?
        ORDER BY timestamp DESC 
        LIMIT ?
        """
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, (start_time.isoformat(), limit))
                rows = cursor.fetchall()
            
            return [Conversation.from_dict(self._row_to_dict(row)) for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to get recent conversations: {e}")
            return []
    
    def update_vector_id(self, conversation_id: str, vector_id: str) -> bool:
        """
        更新对话记录的向量ID
        
        Args:
            conversation_id: 对话记录ID
            vector_id: 向量数据库中的ID
            
        Returns:
            是否更新成功
        """
        sql = "UPDATE conversations SET vector_id = ? WHERE id = ?"
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, (vector_id, conversation_id))
                conn.commit()
            
            logger.debug(f"Vector ID updated for conversation {conversation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update vector ID: {e}")
            return False
    
    def update_summary(self, conversation_id: str, summary: str) -> bool:
        """
        更新对话记录的摘要
        
        Args:
            conversation_id: 对话记录ID
            summary: 摘要文本
            
        Returns:
            是否更新成功
        """
        sql = "UPDATE conversations SET summary = ? WHERE id = ?"
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, (summary, conversation_id))
                conn.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update summary: {e}")
            return False
    
    def delete_conversations_by_friend(self, friend_id: str) -> int:
        """
        删除指定好友的所有对话记录
        
        Args:
            friend_id: 好友ID
            
        Returns:
            删除的记录数量
        """
        sql = "DELETE FROM conversations WHERE friend_id = ?"
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, (friend_id,))
                deleted_count = cursor.rowcount
                conn.commit()
            
            logger.info(f"Deleted {deleted_count} conversations for friend {friend_id}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to delete conversations: {e}")
            return 0
    
    def cleanup_old_records(self, retention_days: int = 30) -> int:
        """
        清理过期的对话记录
        
        Args:
            retention_days: 保留天数
            
        Returns:
            删除的记录数量
        """
        cutoff_time = datetime.now() - timedelta(days=retention_days)
        sql = "DELETE FROM conversations WHERE timestamp < ? AND summary IS NULL"
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, (cutoff_time.isoformat(),))
                deleted_count = cursor.rowcount
                conn.commit()
            
            logger.info(f"Cleaned up {deleted_count} old conversation records")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old records: {e}")
            return 0
    
    def count_by_friend(self, friend_id: str) -> int:
        """
        统计指定好友的对话数量
        
        Args:
            friend_id: 好友ID
            
        Returns:
            对话数量
        """
        sql = "SELECT COUNT(*) FROM conversations WHERE friend_id = ?"
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, (friend_id,))
                count = cursor.fetchone()[0]
            
            return count
            
        except Exception as e:
            logger.error(f"Failed to count conversations: {e}")
            return 0
    
    def count_total(self) -> int:
        """
        统计总对话数量
        
        Returns:
            对话总数
        """
        sql = "SELECT COUNT(*) FROM conversations"
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql)
                count = cursor.fetchone()[0]
            
            return count
            
        except Exception as e:
            logger.error(f"Failed to count total conversations: {e}")
            return 0
