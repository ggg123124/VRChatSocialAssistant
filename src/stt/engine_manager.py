"""
引擎管理器和降级策略
"""

import logging
from typing import Optional, Union
from enum import Enum

from .models import EngineConfig, EngineType, RecognitionResult, EngineError
from .local_engine import LocalSTTEngine
from .cloud_engine import CloudSTTEngine, CloudSTTEngineBase

logger = logging.getLogger(__name__)


class FallbackStrategy(Enum):
    """降级策略"""
    NONE = "none"  # 不降级
    LOCAL_TO_CLOUD = "local_to_cloud"  # 本地失败降级到云端
    CLOUD_TO_LOCAL = "cloud_to_local"  # 云端失败降级到本地
    AUTO = "auto"  # 自动选择（优先本地）


class EngineManager:
    """
    引擎管理器
    
    负责引擎创建、切换和降级策略
    """
    
    def __init__(
        self,
        primary_config: EngineConfig,
        fallback_config: Optional[EngineConfig] = None,
        fallback_strategy: FallbackStrategy = FallbackStrategy.AUTO,
        enable_fallback: bool = True
    ):
        """
        初始化引擎管理器
        
        Args:
            primary_config: 主要引擎配置
            fallback_config: 降级引擎配置（可选）
            fallback_strategy: 降级策略
            enable_fallback: 是否启用降级
        """
        self.primary_config = primary_config
        self.fallback_config = fallback_config
        self.fallback_strategy = fallback_strategy
        self.enable_fallback = enable_fallback
        
        # 引擎实例
        self.primary_engine: Optional[Union[LocalSTTEngine, CloudSTTEngineBase]] = None
        self.fallback_engine: Optional[Union[LocalSTTEngine, CloudSTTEngineBase]] = None
        
        # 当前活跃引擎
        self.current_engine_type: Optional[str] = None
        
        # 统计信息
        self.fallback_count = 0
        self.primary_success_count = 0
        self.primary_fail_count = 0
        self.fallback_success_count = 0
        self.fallback_fail_count = 0
        
        # 初始化引擎
        self._initialize_engines()
    
    def _initialize_engines(self):
        """初始化引擎"""
        # 创建主引擎
        try:
            self.primary_engine = self._create_engine(self.primary_config)
            self.current_engine_type = self.primary_config.engine_type
            logger.info(f"主引擎初始化成功: {self.primary_config.engine_type}")
        except Exception as e:
            logger.error(f"主引擎初始化失败: {e}", exc_info=True)
            raise EngineError(f"主引擎初始化失败: {e}")
        
        # 创建降级引擎
        if self.enable_fallback and self.fallback_config:
            try:
                self.fallback_engine = self._create_engine(self.fallback_config)
                logger.info(f"降级引擎初始化成功: {self.fallback_config.engine_type}")
            except Exception as e:
                logger.warning(f"降级引擎初始化失败: {e}")
                self.fallback_engine = None
    
    def _create_engine(
        self,
        config: EngineConfig
    ) -> Union[LocalSTTEngine, CloudSTTEngineBase]:
        """
        创建引擎实例
        
        Args:
            config: 引擎配置
        
        Returns:
            引擎实例
        """
        if config.engine_type == "local":
            return LocalSTTEngine(config)
        elif config.engine_type == "cloud":
            return CloudSTTEngine.create(config)
        else:
            raise ValueError(f"不支持的引擎类型: {config.engine_type}")
    
    def recognize(self, *args, **kwargs) -> RecognitionResult:
        """
        执行识别（带降级策略）
        
        Args:
            *args: 位置参数
            **kwargs: 关键字参数
        
        Returns:
            识别结果
        """
        # 尝试主引擎
        try:
            result = self.primary_engine.recognize(*args, **kwargs)
            
            if result.success:
                self.primary_success_count += 1
                return result
            else:
                self.primary_fail_count += 1
                logger.warning(f"主引擎识别失败: {result.error_message}")
                
                # 判断是否需要降级
                if self._should_fallback(result):
                    return self._fallback_recognize(*args, **kwargs)
                else:
                    return result
        
        except Exception as e:
            self.primary_fail_count += 1
            logger.error(f"主引擎识别异常: {e}", exc_info=True)
            
            # 异常情况也尝试降级
            if self.enable_fallback and self.fallback_engine:
                return self._fallback_recognize(*args, **kwargs)
            else:
                # 返回错误结果
                request_id = kwargs.get('request_id', '')
                return RecognitionResult(
                    request_id=request_id,
                    success=False,
                    error_message=f"主引擎异常: {str(e)}"
                )
    
    def _should_fallback(self, result: RecognitionResult) -> bool:
        """
        判断是否应该降级
        
        Args:
            result: 主引擎识别结果
        
        Returns:
            是否需要降级
        """
        if not self.enable_fallback:
            return False
        
        if not self.fallback_engine:
            return False
        
        # 根据错误类型判断
        if result.error_message:
            # 模型加载失败、设备不可用等错误需要降级
            error_keywords = ['模型', 'cuda', 'gpu', '显存', '内存', '超时', '网络']
            error_lower = result.error_message.lower()
            
            for keyword in error_keywords:
                if keyword in error_lower:
                    logger.info(f"检测到需要降级的错误: {result.error_message}")
                    return True
        
        # 置信度过低也可以尝试降级
        if result.confidence < 0.3:
            logger.info(f"置信度过低({result.confidence:.2f})，尝试降级")
            return True
        
        return False
    
    def _fallback_recognize(self, *args, **kwargs) -> RecognitionResult:
        """
        使用降级引擎识别
        
        Args:
            *args: 位置参数
            **kwargs: 关键字参数
        
        Returns:
            识别结果
        """
        self.fallback_count += 1
        logger.info("启用降级引擎进行识别")
        
        try:
            result = self.fallback_engine.recognize(*args, **kwargs)
            
            if result.success:
                self.fallback_success_count += 1
                logger.info("降级引擎识别成功")
            else:
                self.fallback_fail_count += 1
                logger.warning(f"降级引擎识别失败: {result.error_message}")
            
            return result
        
        except Exception as e:
            self.fallback_fail_count += 1
            logger.error(f"降级引擎识别异常: {e}", exc_info=True)
            
            request_id = kwargs.get('request_id', '')
            return RecognitionResult(
                request_id=request_id,
                success=False,
                error_message=f"降级引擎异常: {str(e)}"
            )
    
    def switch_engine(self, engine_type: str) -> bool:
        """
        切换当前活跃引擎
        
        Args:
            engine_type: 引擎类型（local/cloud）
        
        Returns:
            是否切换成功
        """
        try:
            if engine_type == self.primary_config.engine_type:
                self.current_engine_type = engine_type
                logger.info(f"切换到主引擎: {engine_type}")
                return True
            
            elif self.fallback_config and engine_type == self.fallback_config.engine_type:
                if self.fallback_engine:
                    # 交换主引擎和降级引擎
                    self.primary_engine, self.fallback_engine = (
                        self.fallback_engine, self.primary_engine
                    )
                    self.primary_config, self.fallback_config = (
                        self.fallback_config, self.primary_config
                    )
                    self.current_engine_type = engine_type
                    logger.info(f"切换到降级引擎: {engine_type}")
                    return True
                else:
                    logger.error(f"降级引擎未初始化: {engine_type}")
                    return False
            else:
                logger.error(f"不支持的引擎类型: {engine_type}")
                return False
        
        except Exception as e:
            logger.error(f"切换引擎失败: {e}", exc_info=True)
            return False
    
    def get_current_engine(self) -> Union[LocalSTTEngine, CloudSTTEngineBase]:
        """
        获取当前活跃引擎
        
        Returns:
            当前引擎实例
        """
        return self.primary_engine
    
    def get_statistics(self) -> dict:
        """
        获取统计信息
        
        Returns:
            统计数据字典
        """
        primary_stats = self.primary_engine.get_statistics() if self.primary_engine else {}
        fallback_stats = self.fallback_engine.get_statistics() if self.fallback_engine else {}
        
        return {
            'current_engine_type': self.current_engine_type,
            'fallback_enabled': self.enable_fallback,
            'fallback_count': self.fallback_count,
            'primary': {
                'success_count': self.primary_success_count,
                'fail_count': self.primary_fail_count,
                'engine_stats': primary_stats,
            },
            'fallback': {
                'success_count': self.fallback_success_count,
                'fail_count': self.fallback_fail_count,
                'engine_stats': fallback_stats,
            }
        }
    
    def reset_statistics(self):
        """重置统计信息"""
        self.fallback_count = 0
        self.primary_success_count = 0
        self.primary_fail_count = 0
        self.fallback_success_count = 0
        self.fallback_fail_count = 0
        
        if self.primary_engine:
            self.primary_engine.reset_statistics()
        
        if self.fallback_engine:
            self.fallback_engine.reset_statistics()
        
        logger.info("引擎管理器统计信息已重置")
