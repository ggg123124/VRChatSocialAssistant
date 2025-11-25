"""
STT (Speech-to-Text) 语音识别模块

提供本地和云端语音转文本识别功能
"""

from .models import (
    RecognitionRequest,
    RecognitionResult,
    SegmentResult,
    EngineConfig,
    EngineType
)
from .stt_recognizer import STTRecognizer

__all__ = [
    'RecognitionRequest',
    'RecognitionResult',
    'SegmentResult',
    'EngineConfig',
    'EngineType',
    'STTRecognizer',
]
