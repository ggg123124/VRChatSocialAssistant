"""
STT 模块数据模型定义
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import numpy as np


class EngineType(Enum):
    """识别引擎类型"""
    LOCAL = "local"
    CLOUD = "cloud"


@dataclass
class RecognitionRequest:
    """
    识别请求模型
    """
    request_id: str
    audio_data: np.ndarray
    sample_rate: int = 16000
    timestamp: float = 0.0
    speaker_id: Optional[str] = None
    language: Optional[str] = "auto"
    priority: int = 5
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """验证数据"""
        if not isinstance(self.audio_data, np.ndarray):
            raise TypeError("audio_data must be numpy.ndarray")
        if self.audio_data.dtype != np.float32:
            self.audio_data = self.audio_data.astype(np.float32)
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if not 0 <= self.priority <= 10:
            raise ValueError("priority must be in range [0, 10]")


@dataclass
class SegmentResult:
    """
    分段识别结果
    """
    id: int
    text: str
    start: float
    end: float
    confidence: float
    tokens: Optional[List[Dict]] = None


@dataclass
class RecognitionResult:
    """
    识别结果模型
    """
    request_id: str
    success: bool
    text: str = ""
    confidence: float = 0.0
    language: Optional[str] = None
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    processing_time: float = 0.0
    engine_type: Optional[str] = None
    speaker_id: Optional[str] = None
    segments: Optional[List[SegmentResult]] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            'request_id': self.request_id,
            'success': self.success,
            'text': self.text,
            'confidence': self.confidence,
            'language': self.language,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'processing_time': self.processing_time,
            'engine_type': self.engine_type,
            'speaker_id': self.speaker_id,
            'error_message': self.error_message,
            'metadata': self.metadata,
        }
        
        if self.segments:
            result['segments'] = [
                {
                    'id': seg.id,
                    'text': seg.text,
                    'start': seg.start,
                    'end': seg.end,
                    'confidence': seg.confidence,
                }
                for seg in self.segments
            ]
        
        return result


@dataclass
class EngineConfig:
    """
    引擎配置模型
    """
    engine_type: str = "local"
    model_name: str = "medium"
    device: str = "auto"
    language: str = "auto"
    compute_type: str = "float16"
    beam_size: int = 5
    best_of: int = 5
    temperature: float = 0.0
    vad_filter: bool = False
    cloud_provider: Optional[str] = None
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    timeout: int = 5
    retry_count: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> tuple[bool, Optional[str]]:
        """
        验证配置有效性
        
        Returns:
            (是否有效, 错误信息)
        """
        if self.engine_type not in ["local", "cloud"]:
            return False, f"Invalid engine_type: {self.engine_type}"
        
        if self.engine_type == "local":
            valid_models = ["tiny", "base", "small", "medium", "large"]
            if self.model_name not in valid_models:
                return False, f"Invalid model_name: {self.model_name}"
            
            if self.device not in ["cpu", "cuda", "auto"]:
                return False, f"Invalid device: {self.device}"
            
            if self.compute_type not in ["float32", "float16", "int8"]:
                return False, f"Invalid compute_type: {self.compute_type}"
        
        elif self.engine_type == "cloud":
            if not self.cloud_provider:
                return False, "cloud_provider is required for cloud engine"
            
            if self.cloud_provider not in ["aliyun", "tencent"]:
                return False, f"Invalid cloud_provider: {self.cloud_provider}"
        
        if self.beam_size < 1:
            return False, "beam_size must be >= 1"
        
        if self.temperature < 0:
            return False, "temperature must be >= 0"
        
        return True, None


class STTError(Exception):
    """STT模块基础异常"""
    pass


class AudioFormatError(STTError):
    """音频格式错误"""
    pass


class ModelLoadError(STTError):
    """模型加载错误"""
    pass


class RecognitionError(STTError):
    """识别错误"""
    pass


class EngineError(STTError):
    """引擎错误"""
    pass
