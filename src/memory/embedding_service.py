"""
文本嵌入服务模块

负责将文本转换为向量表示，管理嵌入模型的下载、加载与推理。
支持批量处理和模型缓存。
"""

import os
import logging
import hashlib
from typing import List, Optional
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingServiceError(Exception):
    """嵌入服务异常基类"""
    pass


class ModelNotFoundError(EmbeddingServiceError):
    """模型未找到异常"""
    pass


class EmbeddingService:
    """
    文本嵌入服务
    
    使用 sentence-transformers 加载 bge-m3 模型
    支持本地模型加载和自动下载
    """
    
    def __init__(
        self,
        model_path: str = "models/embeddings/bge-m3",
        model_name: str = "BAAI/bge-m3",
        device: str = "cuda",
        batch_size: int = 32,
        max_length: int = 512,
        auto_download: bool = True,
        download_mirror: str = "modelscope"
    ):
        """
        初始化嵌入服务
        
        Args:
            model_path: 模型本地路径
            model_name: 模型名称
            device: 运行设备（cuda/cpu）
            batch_size: 批处理大小
            max_length: 文本最大长度
            auto_download: 是否自动下载模型
            download_mirror: 下载镜像源（huggingface/modelscope）
        """
        self.model_path = Path(model_path)
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.auto_download = auto_download
        self.download_mirror = download_mirror
        
        self.model = None
        self._dimension = None
        
        logger.info(f"EmbeddingService initializing: model={model_name}, device={device}")
        
        # 初始化模型
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化模型（加载或下载）"""
        # 检查模型是否存在
        if not self.check_model_exists(str(self.model_path)):
            if self.auto_download:
                logger.info(f"Model not found, downloading from {self.download_mirror}...")
                success = self.download_model(
                    self.model_name, 
                    str(self.model_path), 
                    self.download_mirror
                )
                if not success:
                    raise ModelNotFoundError(
                        f"Model not found and download failed. "
                        f"Please manually download to: {self.model_path}"
                    )
            else:
                raise ModelNotFoundError(
                    f"Model not found at: {self.model_path}. "
                    f"Please set auto_download=True or manually download the model."
                )
        
        # 加载模型
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading model from: {self.model_path}")
            self.model = SentenceTransformer(str(self.model_path), device=self.device)
            
            # 获取向量维度
            self._dimension = self.model.get_sentence_embedding_dimension()
            
            logger.info(f"Model loaded successfully. Dimension: {self._dimension}")
            
        except ImportError:
            raise EmbeddingServiceError(
                "sentence-transformers not installed. "
                "Please install: pip install sentence-transformers"
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise EmbeddingServiceError(f"Failed to load model: {e}")
    
    def check_model_exists(self, model_path: str) -> bool:
        """
        检查模型是否已下载
        
        Args:
            model_path: 模型路径
            
        Returns:
            是否存在
        """
        model_dir = Path(model_path)
        
        # 检查关键文件是否存在
        required_files = ["config.json"]
        # pytorch_model.bin 或 model.safetensors
        has_weights = (model_dir / "pytorch_model.bin").exists() or \
                     (model_dir / "model.safetensors").exists()
        
        if not has_weights:
            return False
        
        for filename in required_files:
            if not (model_dir / filename).exists():
                return False
        
        return True
    
    def download_model(
        self, 
        model_name: str, 
        save_path: str, 
        mirror: str = "modelscope"
    ) -> bool:
        """
        下载模型到项目目录
        
        Args:
            model_name: 模型名称
            save_path: 保存路径
            mirror: 镜像源（huggingface/modelscope）
            
        Returns:
            是否下载成功
        """
        try:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            if mirror == "modelscope":
                # 使用 ModelScope 下载
                logger.info(f"Downloading from ModelScope: {model_name}")
                return self._download_from_modelscope(model_name, str(save_dir))
            else:
                # 使用 HuggingFace 下载
                logger.info(f"Downloading from HuggingFace: {model_name}")
                return self._download_from_huggingface(model_name, str(save_dir))
                
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            return False
    
    def _download_from_modelscope(self, model_name: str, save_path: str) -> bool:
        """从 ModelScope 下载模型"""
        try:
            from modelscope import snapshot_download
            
            # ModelScope 上的模型名称映射
            modelscope_model_map = {
                "BAAI/bge-m3": "Xorbits/bge-m3",
                "BAAI/bge-small-zh-v1.5": "Xorbits/bge-small-zh-v1.5"
            }
            
            modelscope_name = modelscope_model_map.get(model_name, model_name)
            
            logger.info(f"Starting download: {modelscope_name}")
            model_dir = snapshot_download(modelscope_name, cache_dir=save_path)
            
            logger.info(f"Model downloaded to: {model_dir}")
            
            # 移动文件到目标目录（如果需要）
            # ModelScope 的 snapshot_download 会下载到子目录，需要调整
            import shutil
            source_dir = Path(model_dir)
            target_dir = Path(save_path)
            
            if source_dir != target_dir:
                for item in source_dir.iterdir():
                    target_item = target_dir / item.name
                    if item.is_file():
                        shutil.copy2(item, target_item)
                    elif item.is_dir():
                        if target_item.exists():
                            shutil.rmtree(target_item)
                        shutil.copytree(item, target_item)
            
            return True
            
        except ImportError:
            logger.error("modelscope package not installed. Please install: pip install modelscope")
            return False
        except Exception as e:
            logger.error(f"ModelScope download failed: {e}")
            return False
    
    def _download_from_huggingface(self, model_name: str, save_path: str) -> bool:
        """从 HuggingFace 下载模型"""
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Downloading model via sentence-transformers: {model_name}")
            
            # sentence-transformers 会自动下载模型
            model = SentenceTransformer(model_name, cache_folder=save_path)
            
            # 保存到指定路径
            model.save(save_path)
            
            logger.info(f"Model downloaded and saved to: {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"HuggingFace download failed: {e}")
            return False
    
    def encode(self, text: str) -> List[float]:
        """
        单文本嵌入
        
        Args:
            text: 输入文本
            
        Returns:
            嵌入向量
        """
        if self.model is None:
            raise EmbeddingServiceError("Model not initialized")
        
        # 文本预处理
        text = self._preprocess_text(text)
        
        try:
            embedding = self.model.encode(
                text, 
                normalize_embeddings=True,
                show_progress_bar=False
            )
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Failed to encode text: {e}")
            raise EmbeddingServiceError(f"Failed to encode text: {e}")
    
    def batch_encode(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        """
        批量文本嵌入
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小（None则使用默认值）
            
        Returns:
            嵌入向量列表
        """
        if self.model is None:
            raise EmbeddingServiceError("Model not initialized")
        
        if batch_size is None:
            batch_size = self.batch_size
        
        # 文本预处理
        texts = [self._preprocess_text(text) for text in texts]
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > 100
            )
            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"Failed to batch encode texts: {e}")
            raise EmbeddingServiceError(f"Failed to batch encode texts: {e}")
    
    def _preprocess_text(self, text: str) -> str:
        """
        文本预处理
        
        Args:
            text: 原始文本
            
        Returns:
            预处理后的文本
        """
        # 去除多余空格
        text = " ".join(text.split())
        
        # 截断长度（简单截断，实际应按 token 计数）
        if len(text) > self.max_length * 2:  # 粗略估计，字符数约为token数的2倍
            text = text[:self.max_length * 2]
        
        return text
    
    def get_dimension(self) -> int:
        """
        获取向量维度
        
        Returns:
            向量维度
        """
        if self._dimension is None:
            raise EmbeddingServiceError("Model not initialized")
        return self._dimension
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        计算两个向量的余弦相似度
        
        Args:
            embedding1: 向量1
            embedding2: 向量2
            
        Returns:
            相似度（0-1）
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # 余弦相似度
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        
        # 转换到 0-1 范围
        return float((similarity + 1) / 2)
