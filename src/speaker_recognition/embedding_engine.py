"""
声纹提取引擎

负责加载模型并从音频中提取声纹嵌入向量
"""

import logging
import os
from pathlib import Path
from typing import Optional, Union
import numpy as np
import torch
import torchaudio

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """
    声纹提取引擎
    
    使用 pyannote.audio 的 ECAPA-TDNN 模型提取声纹特征
    """
    
    def __init__(
        self,
        model_path: str = "models/speaker_recognition/ecapa-tdnn/",
        sample_rate: int = 16000,
        device: str = "auto",
        auto_download: bool = True,
        min_audio_length: float = 0.5,
        max_audio_length: float = 10.0,
    ):
        """
        初始化声纹提取引擎
        
        Args:
            model_path: 模型存储路径
            sample_rate: 音频采样率
            device: 推理设备 (auto/cpu/cuda)
            auto_download: 是否自动下载模型
            min_audio_length: 最短音频时长（秒）
            max_audio_length: 最长音频时长（秒）
        """
        self.model_path = Path(model_path)
        self.sample_rate = sample_rate
        self.min_audio_length = min_audio_length
        self.max_audio_length = max_audio_length
        
        # 确定设备
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"EmbeddingEngine 初始化: device={self.device}, "
                   f"sample_rate={sample_rate}")
        
        # 初始化模型
        self.model = None
        self._load_model(auto_download)
    
    def _load_model(self, auto_download: bool = True):
        """
        加载声纹提取模型
        
        Args:
            auto_download: 是否自动下载模型
        """
        try:
            # 尝试从本地加载
            if self._check_model_exists():
                self._load_from_local()
            elif auto_download:
                logger.info("本地模型不存在，开始自动下载...")
                self._download_model()
                self._load_from_local()
            else:
                raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
            
            logger.info("声纹提取模型加载成功")
            
        except Exception as e:
            logger.error(f"加载声纹提取模型失败: {e}", exc_info=True)
            raise
    
    def _check_model_exists(self) -> bool:
        """检查模型文件是否存在"""
        # 检查是否存在模型文件（简化版检查）
        if not self.model_path.exists():
            return False
        
        # 检查关键文件
        config_file = self.model_path / "config.yaml"
        return config_file.exists()
    
    def _download_model(self):
        """
        从 ModelScope 下载模型
        """
        try:
            from modelscope.hub.snapshot_download import snapshot_download
            
            logger.info("从 ModelScope 下载 pyannote.audio ECAPA-TDNN 模型...")
            
            # 创建目录
            self.model_path.mkdir(parents=True, exist_ok=True)
            
            # 下载模型
            # 注意：这里使用的是示例路径，实际需要找到正确的ModelScope模型ID
            model_dir = snapshot_download(
                'iic/speech_ecapa-tdnn_sv_en_voxceleb_16k',
                cache_dir=str(self.model_path.parent)
            )
            
            logger.info(f"模型下载成功: {model_dir}")
            
        except ImportError:
            logger.error("未安装 modelscope，无法自动下载模型")
            logger.info("请手动安装: pip install modelscope")
            raise
        except Exception as e:
            logger.error(f"下载模型失败: {e}", exc_info=True)
            raise
    
    def _load_from_local(self):
        """
        从本地加载模型
        """
        try:
            # 首先尝试加载 ModelScope 格式的模型
            modelscope_model_path = Path("models/speaker_recognition/iic/speech_ecapa-tdnn_sv_en_voxceleb_16k")
            if modelscope_model_path.exists():
                logger.info("检测到 ModelScope 模型，尝试加载...")
                self._load_modelscope_model(modelscope_model_path)
                return
            
            # 使用 pyannote.audio 加载模型
            # 注意：在 Windows 上可能会出现 PyTorch 死锁问题
            import os
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
            
            from pyannote.audio import Model
            
            model_file = self.model_path / "pytorch_model.bin"
            if not model_file.exists():
                # 尝试从目录加载
                self.model = Model.from_pretrained(str(self.model_path))
            else:
                self.model = Model.from_pretrained(str(model_file))
            
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"从本地加载模型: {self.model_path}")
            
        except ImportError as e:
            logger.warning(f"pyannote.audio 未安装，使用替代方案: {e}")
            self._load_alternative_model()
        except RuntimeError as e:
            # 捕获 PyTorch 死锁错误
            if 'deadlock' in str(e).lower():
                logger.warning(f"PyTorch 死锁错误（Windows 常见问题），使用替代方案")
                self._load_alternative_model()
            else:
                logger.error(f"从本地加载模型失败: {e}", exc_info=True)
                raise
        except Exception as e:
            logger.error(f"从本地加载模型失败: {e}", exc_info=True)
            # 尝试使用替代模型
            logger.warning("尝试使用替代模型...")
            self._load_alternative_model()
    
    def _load_modelscope_model(self, model_path: Path):
        """
        加载 ModelScope 格式的 ECAPA-TDNN 模型
        
        Args:
            model_path: ModelScope 模型目录路径
        """
        try:
            logger.info(f"从 ModelScope 加载模型: {model_path}")
            
            # 优先尝试使用 Resemblyzer （简单易用且开源）
            try:
                from resemblyzer import VoiceEncoder
                
                logger.info("使用 Resemblyzer 加载真实声纹识别模型...")
                
                # 创建 Resemblyzer 包装器
                class ResemblyzerWrapper:
                    def __init__(self, device):
                        # Resemblyzer 使用预训练的 GE2E 模型
                        device_str = 'cuda' if 'cuda' in str(device) else 'cpu'
                        self.encoder = VoiceEncoder(device=device_str)
                        self.device = device
                        self.embedding_dim = 256  # Resemblyzer 嵌入维度
                        logger.info(f"✓ Resemblyzer 初始化成功，设备: {device_str}")
                    
                    def to(self, device):
                        # Resemblyzer 不需要显式移动设备
                        self.device = device
                        return self
                    
                    def eval(self):
                        # Resemblyzer 默认就是 eval 模式
                        return self
                    
                    def __call__(self, waveform):
                        import torch
                        import numpy as np
                        
                        # 转换为 numpy
                        if isinstance(waveform, torch.Tensor):
                            audio = waveform.cpu().numpy()
                        else:
                            audio = np.array(waveform, dtype=np.float32)
                        
                        # 确保是 1D 数组
                        if audio.ndim > 1:
                            audio = audio.squeeze()
                        
                        # Resemblyzer 接受 [samples] 格式，采样率 16kHz
                        # 提取嵌入（返回 (256,) 的 numpy 数组）
                        embedding = self.encoder.embed_utterance(audio)
                        
                        # 转换为 torch tensor 并增加 batch 维度
                        embedding_tensor = torch.from_numpy(embedding).float()
                        if embedding_tensor.ndim == 1:
                            embedding_tensor = embedding_tensor.unsqueeze(0)
                        
                        return embedding_tensor
                
                self.model = ResemblyzerWrapper(self.device)
                logger.info("✓ 真实声纹识别模型加载成功 (Resemblyzer GE2E)")
                logger.info("✓ 使用真实的深度学习模型进行声纹识别")
                return
                
            except ImportError:
                logger.warning("Resemblyzer 未安装，尝试 SpeechBrain...")
            except Exception as e:
                logger.warning(f"Resemblyzer 加载失败: {e}，尝试 SpeechBrain...")
            
            # 尝试使用 SpeechBrain
            try:
                import speechbrain
                from speechbrain.inference.speaker import EncoderClassifier
                
                logger.info("使用 SpeechBrain 加载 ECAPA-TDNN 模型...")
                
                # 检查是否存在模型文件
                model_file = model_path / "ecapa_tdnn.bin"
                if not model_file.exists():
                    raise FileNotFoundError(f"模型文件不存在: {model_file}")
                
                # SpeechBrain 可以加载 ModelScope 的 ECAPA-TDNN 模型
                # 创建包装器来使用 SpeechBrain 的推理引擎
                class SpeechBrainWrapper:
                    def __init__(self, model_path, device):
                        # 加载预训练的 ECAPA-TDNN 编码器
                        # 注意：这需要 SpeechBrain 支持的模型格式
                        self.model_path = model_path
                        self.device = device
                        self.embedding_dim = 192
                        
                        # 尝试使用 SpeechBrain 的预训练模型
                        # 由于 ModelScope 格式不同，我们使用 HuggingFace 的预训练模型
                        try:
                            self.classifier = EncoderClassifier.from_hparams(
                                source="speechbrain/spkrec-ecapa-voxceleb",
                                savedir="models/speaker_recognition/speechbrain",
                                run_opts={"device": str(device)}
                            )
                            logger.info("✓ SpeechBrain ECAPA-TDNN 模型加载成功")
                        except Exception as e:
                            logger.warning(f"加载 SpeechBrain 预训练模型失败: {e}")
                            raise
                    
                    def to(self, device):
                        self.device = device
                        return self
                    
                    def eval(self):
                        return self
                    
                    def __call__(self, waveform):
                        import torch
                        import numpy as np
                        
                        # 转换为正确格式
                        if isinstance(waveform, np.ndarray):
                            waveform = torch.from_numpy(waveform).float()
                        
                        # SpeechBrain 期望 [batch, time] 格式
                        if waveform.ndim == 1:
                            waveform = waveform.unsqueeze(0)
                        
                        waveform = waveform.to(self.device)
                        
                        # 提取嵌入
                        with torch.no_grad():
                            embedding = self.classifier.encode_batch(waveform)
                        
                        return embedding
                
                self.model = SpeechBrainWrapper(model_path, self.device)
                logger.info("✓ 真实 ECAPA-TDNN 声纹识别模型加载成功")
                return
                
            except ImportError:
                logger.warning("SpeechBrain 未安装，尝试使用 pyannote.audio...")
            except Exception as e:
                logger.warning(f"SpeechBrain 加载失败: {e}，尝试使用 pyannote.audio...")
            
            # 尝试使用 pyannote.audio
            try:
                from pyannote.audio import Model, Inference
                
                logger.info("使用 pyannote.audio 加载说话人识别模型...")
                
                # pyannote.audio 可以使用预训练的说话人嵌入模型
                # 使用 HuggingFace Hub 上的模型
                try:
                    model = Model.from_pretrained(
                        "pyannote/embedding",
                        use_auth_token=False
                    )
                    model.to(self.device)
                    model.eval()
                    
                    # 创建推理包装器
                    class PyannoteWrapper:
                        def __init__(self, model, device):
                            self.model = model
                            self.device = device
                            self.inference = Inference(model, window="whole")
                            self.embedding_dim = 512  # pyannote embedding 维度
                        
                        def to(self, device):
                            self.device = device
                            self.model.to(device)
                            return self
                        
                        def eval(self):
                            self.model.eval()
                            return self
                        
                        def __call__(self, waveform):
                            import torch
                            import numpy as np
                            
                            # 转换格式
                            if isinstance(waveform, np.ndarray):
                                waveform = torch.from_numpy(waveform).float()
                            
                            if waveform.ndim == 1:
                                waveform = waveform.unsqueeze(0)
                            
                            waveform = waveform.to(self.device)
                            
                            # 使用 pyannote 推理
                            with torch.no_grad():
                                embedding = self.model(waveform)
                            
                            # 如果返回字典，提取嵌入
                            if isinstance(embedding, dict):
                                embedding = embedding.get('embedding', embedding)
                            
                            return embedding
                    
                    self.model = PyannoteWrapper(model, self.device)
                    logger.info("✓ pyannote.audio 说话人嵌入模型加载成功")
                    logger.info("✓ 使用真实的深度学习模型进行声纹识别")
                    return
                    
                except Exception as e:
                    logger.warning(f"从 HuggingFace 加载 pyannote 模型失败: {e}")
                    logger.info("提示：某些 pyannote 模型需要 HuggingFace 认证令牌")
                    logger.info("  - 访问 https://huggingface.co/pyannote/embedding")
                    logger.info("  - 接受使用条款后获取访问令牌")
                    raise
                    
            except ImportError:
                logger.warning("pyannote.audio 未正确安装")
            except Exception as e:
                logger.warning(f"pyannote.audio 加载失败: {e}")
            
            # 如果所有方法都失败，使用简化模型
            logger.warning("无法加载真实声纹识别模型，使用替代方案")
            logger.info("提示: 请安装以下任一库以启用真实声纹识别：")
            logger.info("  - SpeechBrain: pip install speechbrain")
            logger.info("  - pyannote.audio: pip install pyannote.audio")
            self._load_alternative_model()
            
        except Exception as e:
            logger.error(f"加载 ModelScope 模型失败: {e}", exc_info=True)
            raise
    
    def _load_alternative_model(self):
        """
        加载替代模型（简化实现，用于演示）
        
        注意：这是一个占位实现，实际应用中应使用真实的声纹模型
        """
        logger.warning("使用简化的替代模型（仅用于演示）")
        
        # 创建一个简单的随机嵌入提取器（仅用于测试）
        class DummyEmbeddingModel:
            def __init__(self, embedding_dim=192):
                self.embedding_dim = embedding_dim
            
            def to(self, device):
                return self
            
            def eval(self):
                return self
            
            def __call__(self, waveform):
                # 返回随机嵌入（仅用于测试）
                batch_size = waveform.shape[0] if len(waveform.shape) > 1 else 1
                # 使用音频数据的统计特征生成"嵌入"
                if len(waveform.shape) == 1:
                    waveform = waveform.unsqueeze(0)
                
                # 计算简单的音频特征
                mean = torch.mean(waveform, dim=-1)
                std = torch.std(waveform, dim=-1)
                max_val = torch.max(waveform, dim=-1)[0]
                min_val = torch.min(waveform, dim=-1)[0]
                
                # 重复特征以匹配嵌入维度
                features = torch.stack([mean, std, max_val, min_val], dim=-1)
                embedding = features.repeat(1, self.embedding_dim // 4)
                
                return embedding
        
        self.model = DummyEmbeddingModel(embedding_dim=192)
        logger.warning("⚠️ 当前使用的是演示用的简化模型，不具备真实的声纹识别能力")
        logger.warning("⚠️ 请安装 pyannote.audio 以使用真实模型: pip install pyannote.audio")
    
    def preprocess_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: Optional[int] = None
    ) -> torch.Tensor:
        """
        音频预处理
        
        Args:
            audio_data: 音频数据 (numpy array, float32)
            sample_rate: 原始采样率（如果与目标不同则重采样）
        
        Returns:
            预处理后的音频张量
        """
        # 确保是 numpy array
        if not isinstance(audio_data, np.ndarray):
            audio_data = np.array(audio_data, dtype=np.float32)
        
        # 确保是 float32
        audio_data = audio_data.astype(np.float32)
        
        # 归一化到 [-1, 1]
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / 32768.0
        
        # 转换为 torch tensor
        waveform = torch.from_numpy(audio_data).float()
        
        # 确保是2D张量 [channels, samples]
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        
        # 重采样（如果需要）
        if sample_rate is not None and sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.sample_rate
            )
            waveform = resampler(waveform)
        
        # 填充或截断到合适长度
        num_samples = waveform.shape[1]
        target_samples = int(self.sample_rate * 3.0)  # 默认3秒
        
        if num_samples < target_samples:
            # 填充
            padding = target_samples - num_samples
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif num_samples > target_samples:
            # 截断
            waveform = waveform[:, :target_samples]
        
        return waveform
    
    def extract_embedding(
        self,
        audio_data: Union[np.ndarray, torch.Tensor],
        sample_rate: Optional[int] = None
    ) -> np.ndarray:
        """
        从音频中提取声纹嵌入向量
        
        Args:
            audio_data: 音频数据 (numpy array 或 torch tensor)
            sample_rate: 音频采样率
        
        Returns:
            声纹嵌入向量 (numpy array, shape: (192,))
        """
        if self.model is None:
            raise RuntimeError("模型未加载")
        
        try:
            # 预处理音频
            if isinstance(audio_data, np.ndarray):
                waveform = self.preprocess_audio(audio_data, sample_rate)
            else:
                waveform = audio_data
            
            # 移动到设备
            waveform = waveform.to(self.device)
            
            # 提取嵌入
            with torch.no_grad():
                embedding = self.model(waveform)
            
            # 转换为 numpy array
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.cpu().numpy()
            
            # 确保是1D数组
            if embedding.ndim > 1:
                embedding = embedding.squeeze()
            
            # 归一化
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            logger.debug(f"提取声纹嵌入: shape={embedding.shape}, "
                        f"norm={np.linalg.norm(embedding):.3f}")
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.error(f"提取声纹嵌入失败: {e}", exc_info=True)
            raise
    
    def validate_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> tuple[bool, str]:
        """
        验证音频质量是否满足要求
        
        Args:
            audio_data: 音频数据
            sample_rate: 采样率
        
        Returns:
            (是否合格, 错误信息)
        """
        # 检查时长
        duration = len(audio_data) / sample_rate
        if duration < self.min_audio_length:
            return False, f"音频时长过短: {duration:.2f}s < {self.min_audio_length}s"
        if duration > self.max_audio_length:
            return False, f"音频时长过长: {duration:.2f}s > {self.max_audio_length}s"
        
        # 检查音量
        rms = np.sqrt(np.mean(audio_data ** 2))
        if rms < 0.001:  # 约 -60dB
            return False, f"音频音量过小: RMS={rms:.6f}"
        
        # 检查是否全为静音
        if np.max(np.abs(audio_data)) < 0.001:
            return False, "音频数据接近静音"
        
        return True, ""
    
    def get_model_info(self) -> dict:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        return {
            'model_path': str(self.model_path),
            'device': str(self.device),
            'sample_rate': self.sample_rate,
            'min_audio_length': self.min_audio_length,
            'max_audio_length': self.max_audio_length,
            'model_loaded': self.model is not None,
        }
