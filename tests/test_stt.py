"""
STT 模块测试用例
"""

import sys
import os
import logging
import numpy as np
import time

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.stt import STTRecognizer, RecognitionResult

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def generate_test_audio(duration: float = 1.0, sample_rate: int = 16000) -> np.ndarray:
    """
    生成测试音频（正弦波）
    
    Args:
        duration: 时长（秒）
        sample_rate: 采样率
    
    Returns:
        音频数据
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    # 生成 440Hz 的正弦波（A4音）
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio


def test_recognizer_initialization():
    """测试识别器初始化"""
    logger.info("=" * 60)
    logger.info("测试 1: 识别器初始化")
    logger.info("=" * 60)
    
    try:
        recognizer = STTRecognizer()
        logger.info("✓ 识别器初始化成功")
        
        # 获取统计信息
        stats = recognizer.get_statistics()
        logger.info(f"初始统计信息: {stats}")
        
        return True
    except Exception as e:
        logger.error(f"✗ 初始化失败: {e}", exc_info=True)
        return False


def test_audio_validation():
    """测试音频验证"""
    logger.info("=" * 60)
    logger.info("测试 2: 音频验证")
    logger.info("=" * 60)
    
    try:
        recognizer = STTRecognizer()
        
        # 测试过短音频
        short_audio = generate_test_audio(duration=0.1)
        result = recognizer.recognize(short_audio)
        logger.info(f"过短音频: success={result.success}, error={result.error_message}")
        
        # 测试正常音频
        normal_audio = generate_test_audio(duration=1.0)
        result = recognizer.recognize(normal_audio)
        logger.info(f"正常音频: success={result.success}")
        
        # 测试过长音频
        long_audio = generate_test_audio(duration=35.0)
        result = recognizer.recognize(long_audio)
        logger.info(f"过长音频: success={result.success}, error={result.error_message}")
        
        logger.info("✓ 音频验证测试完成")
        return True
        
    except Exception as e:
        logger.error(f"✗ 音频验证测试失败: {e}", exc_info=True)
        return False


def test_basic_recognition():
    """测试基本识别功能"""
    logger.info("=" * 60)
    logger.info("测试 3: 基本识别功能")
    logger.info("=" * 60)
    
    try:
        recognizer = STTRecognizer()
        
        # 生成测试音频
        audio = generate_test_audio(duration=2.0)
        
        logger.info("开始识别测试音频...")
        start_time = time.time()
        
        result = recognizer.recognize(audio)
        
        elapsed = (time.time() - start_time) * 1000
        
        logger.info(f"识别结果:")
        logger.info(f"  - 成功: {result.success}")
        logger.info(f"  - 文本: {result.text}")
        logger.info(f"  - 置信度: {result.confidence:.3f}")
        logger.info(f"  - 语言: {result.language}")
        logger.info(f"  - 引擎: {result.engine_type}")
        logger.info(f"  - 处理时间: {result.processing_time:.1f}ms")
        logger.info(f"  - 总耗时: {elapsed:.1f}ms")
        
        if result.error_message:
            logger.info(f"  - 错误信息: {result.error_message}")
        
        logger.info("✓ 基本识别测试完成")
        return True
        
    except Exception as e:
        logger.error(f"✗ 基本识别测试失败: {e}", exc_info=True)
        return False


def test_callback_mechanism():
    """测试回调机制"""
    logger.info("=" * 60)
    logger.info("测试 4: 回调机制")
    logger.info("=" * 60)
    
    try:
        recognizer = STTRecognizer()
        
        # 设置回调函数
        callback_called = [False]
        
        def on_result(result: RecognitionResult):
            logger.info(f"回调被调用: text='{result.text}', success={result.success}")
            callback_called[0] = True
        
        recognizer.set_callback(on_result)
        
        # 执行识别
        audio = generate_test_audio(duration=1.0)
        recognizer.recognize(audio)
        
        if callback_called[0]:
            logger.info("✓ 回调机制测试成功")
            return True
        else:
            logger.error("✗ 回调未被调用")
            return False
        
    except Exception as e:
        logger.error(f"✗ 回调机制测试失败: {e}", exc_info=True)
        return False


def test_statistics():
    """测试统计功能"""
    logger.info("=" * 60)
    logger.info("测试 5: 统计功能")
    logger.info("=" * 60)
    
    try:
        recognizer = STTRecognizer()
        
        # 执行多次识别
        audio = generate_test_audio(duration=1.0)
        
        for i in range(3):
            recognizer.recognize(audio)
            logger.info(f"完成第 {i+1} 次识别")
        
        # 获取统计信息
        stats = recognizer.get_statistics()
        
        logger.info("统计信息:")
        logger.info(f"  - 总请求数: {stats['total_requests']}")
        logger.info(f"  - 成功数: {stats['successful_requests']}")
        logger.info(f"  - 失败数: {stats['failed_requests']}")
        logger.info(f"  - 成功率: {stats['success_rate']:.2%}")
        
        # 重置统计
        recognizer.reset_statistics()
        stats = recognizer.get_statistics()
        
        logger.info("重置后统计:")
        logger.info(f"  - 总请求数: {stats['total_requests']}")
        
        logger.info("✓ 统计功能测试成功")
        return True
        
    except Exception as e:
        logger.error(f"✗ 统计功能测试失败: {e}", exc_info=True)
        return False


def test_engine_switching():
    """测试引擎切换"""
    logger.info("=" * 60)
    logger.info("测试 6: 引擎切换")
    logger.info("=" * 60)
    
    try:
        recognizer = STTRecognizer()
        
        # 获取当前引擎
        stats = recognizer.get_statistics()
        current_engine = stats.get('engine', {}).get('current_engine_type')
        logger.info(f"当前引擎: {current_engine}")
        
        # 尝试切换引擎
        if current_engine == 'local':
            success = recognizer.switch_engine('cloud')
            logger.info(f"切换到云端引擎: {success}")
        else:
            success = recognizer.switch_engine('local')
            logger.info(f"切换到本地引擎: {success}")
        
        # 获取切换后的引擎
        stats = recognizer.get_statistics()
        new_engine = stats.get('engine', {}).get('current_engine_type')
        logger.info(f"切换后引擎: {new_engine}")
        
        logger.info("✓ 引擎切换测试完成")
        return True
        
    except Exception as e:
        logger.error(f"✗ 引擎切换测试失败: {e}", exc_info=True)
        return False


def main():
    """运行所有测试"""
    logger.info("开始 STT 模块测试")
    logger.info("注意: 本测试使用合成音频，实际识别可能失败（这是正常的）")
    logger.info("")
    
    tests = [
        ("初始化测试", test_recognizer_initialization),
        ("音频验证测试", test_audio_validation),
        ("基本识别测试", test_basic_recognition),
        ("回调机制测试", test_callback_mechanism),
        ("统计功能测试", test_statistics),
        ("引擎切换测试", test_engine_switching),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            logger.info("")
        except Exception as e:
            logger.error(f"测试 {test_name} 异常: {e}", exc_info=True)
            results.append((test_name, False))
            logger.info("")
    
    # 输出总结
    logger.info("=" * 60)
    logger.info("测试总结")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        logger.info(f"{status} - {test_name}")
    
    logger.info("")
    logger.info(f"总计: {passed}/{total} 通过 ({passed/total*100:.1f}%)")


if __name__ == "__main__":
    main()
