"""
数据模型定义

定义记忆与检索模块使用的核心数据结构。
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class EventType(str, Enum):
    """对话事件类型枚举"""
    QUESTION = "QUESTION"  # 提问
    STATEMENT = "STATEMENT"  # 陈述
    TOPIC_CHANGE = "TOPIC_CHANGE"  # 话题转换
    SILENCE = "SILENCE"  # 静默


class Personality(str, Enum):
    """性格标签枚举"""
    OUTGOING = "外向"
    INTROVERTED = "内向"
    EASYGOING = "随和"
    LIVELY = "活泼"


class FriendProfile(BaseModel):
    """好友档案数据模型"""
    
    id: str = Field(..., description="唯一标识符（UUID）")
    name: str = Field(..., description="好友昵称")
    voice_profile_path: str = Field(..., description="声纹文件路径")
    preferences: List[str] = Field(default_factory=list, description="兴趣爱好列表")
    avoid_topics: List[str] = Field(default_factory=list, description="禁忌话题列表")
    personality: Personality = Field(default=Personality.EASYGOING, description="性格标签")
    language_preference: str = Field(default="zh", description="语言偏好（zh/en/ja）")
    custom_notes: Optional[str] = Field(default=None, description="用户自定义备注")
    last_seen: datetime = Field(default_factory=datetime.now, description="最后交互时间")
    conversation_count: int = Field(default=0, description="对话次数")
    created_at: datetime = Field(default_factory=datetime.now, description="档案创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="档案更新时间")
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return self.model_dump(mode='json')
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FriendProfile':
        """从字典创建对象"""
        # 转换时间戳字符串为 datetime 对象
        if isinstance(data.get('last_seen'), str):
            data['last_seen'] = datetime.fromisoformat(data['last_seen'])
        if isinstance(data.get('created_at'), str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if isinstance(data.get('updated_at'), str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)


class Conversation(BaseModel):
    """对话记录数据模型"""
    
    id: str = Field(..., description="唯一标识符（UUID）")
    friend_id: str = Field(..., description="好友ID（外键）")
    timestamp: datetime = Field(..., description="对话时间")
    speaker_id: str = Field(..., description="说话人ID（区分自己/好友）")
    transcript: str = Field(..., description="识别文本")
    event_type: EventType = Field(default=EventType.STATEMENT, description="对话事件类型")
    confidence: float = Field(default=0.9, ge=0.0, le=1.0, description="识别置信度")
    vector_id: Optional[str] = Field(default=None, description="向量数据库中的ID")
    summary: Optional[str] = Field(default=None, description="摘要（用于长对话压缩）")
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return self.model_dump(mode='json')
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Conversation':
        """从字典创建对象"""
        if isinstance(data.get('timestamp'), str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class Memory(BaseModel):
    """记忆片段数据模型（检索结果）"""
    
    id: str = Field(..., description="记忆ID")
    content: str = Field(..., description="文本内容")
    friend_id: str = Field(..., description="好友ID")
    friend_name: str = Field(..., description="好友昵称")
    timestamp: datetime = Field(..., description="时间戳")
    event_type: EventType = Field(..., description="事件类型")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="相似度评分")
    time_decay_factor: float = Field(default=1.0, ge=0.0, le=1.0, description="时间衰减系数")
    final_score: float = Field(..., ge=0.0, description="最终得分")
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return self.model_dump(mode='json')
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
        """从字典创建对象"""
        if isinstance(data.get('timestamp'), str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class SearchResult(BaseModel):
    """向量检索结果数据模型"""
    
    id: str = Field(..., description="向量ID")
    distance: float = Field(..., description="距离（或相似度）")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    document: str = Field(..., description="原始文档")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return self.model_dump(mode='json')
