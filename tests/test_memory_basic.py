"""
记忆模块基础功能测试

演示如何使用 MemoryManager 进行基本操作
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory import MemoryManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_basic_workflow():
    """测试基本工作流程"""
    
    logger.info("=" * 60)
    logger.info("开始记忆模块基础测试")
    logger.info("=" * 60)
    
    try:
        # 1. 初始化 MemoryManager
        logger.info("\n[步骤1] 初始化 MemoryManager...")
        manager = MemoryManager()
        
        # 2. 创建好友档案
        logger.info("\n[步骤2] 创建好友档案...")
        friend_id = manager.create_friend_profile(
            name="测试好友",
            voice_profile_path="data/speaker_profiles/test_friend.npy",
            preferences=["游戏", "动漫", "VRChat"],
            avoid_topics=["政治"],
            personality="活泼"
        )
        logger.info(f"✓ 好友档案已创建，ID: {friend_id}")
        
        # 3. 获取好友档案
        logger.info("\n[步骤3] 获取好友档案...")
        profile = manager.get_friend_profile(friend_id)
        logger.info(f"✓ 好友信息: {profile.name}, 偏好: {', '.join(profile.preferences)}")
        
        # 4. 添加对话记录
        logger.info("\n[步骤4] 添加对话记录...")
        conversations = [
            "我最近在玩一个很有趣的游戏",
            "你有看过最新的动漫吗？",
            "VRChat里面有很多有趣的世界",
            "我喜欢探索不同的VR场景",
            "今天天气真不错"
        ]
        
        for i, text in enumerate(conversations):
            conv_id = manager.add_conversation(
                friend_id=friend_id,
                transcript=text,
                speaker_id=friend_id,
                event_type="STATEMENT"
            )
            logger.info(f"  [{i+1}] 对话已添加: {text[:30]}...")
        
        logger.info(f"✓ 共添加 {len(conversations)} 条对话记录")
        
        # 5. 检索相关记忆
        logger.info("\n[步骤5] 检索相关记忆...")
        query = "VRChat游戏"
        memories = manager.retrieve_memories(
            query=query,
            friend_id=friend_id,
            top_k=3
        )
        
        logger.info(f"✓ 查询: '{query}' 找到 {len(memories)} 条相关记忆:")
        for i, memory in enumerate(memories):
            logger.info(f"  [{i+1}] 相似度={memory.similarity_score:.3f}, "
                       f"时间衰减={memory.time_decay_factor:.3f}, "
                       f"内容: {memory.content[:40]}...")
        
        # 6. 获取统计信息
        logger.info("\n[步骤6] 获取统计信息...")
        stats = manager.get_statistics()
        logger.info(f"✓ 统计信息:")
        logger.info(f"  - 好友数量: {stats.get('total_friends', 0)}")
        logger.info(f"  - 对话总数: {stats.get('total_conversations', 0)}")
        logger.info(f"  - 向量总数: {stats.get('total_vectors', 0)}")
        logger.info(f"  - 向量维度: {stats.get('embedding_dimension', 0)}")
        
        logger.info("\n" + "=" * 60)
        logger.info("✓ 所有测试通过！")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"\n✗ 测试失败: {e}", exc_info=True)
        raise


def test_without_model():
    """
    测试在没有模型的情况下的行为
    
    注意：如果模型未下载，EmbeddingService 会尝试自动下载
    """
    logger.info("\n测试模型自动下载功能...")
    logger.info("注意：首次运行会自动下载 bge-m3 模型（约2GB），可能需要一些时间")
    
    try:
        manager = MemoryManager()
        logger.info("✓ 模型加载成功（或已自动下载）")
    except Exception as e:
        logger.error(f"✗ 模型加载失败: {e}")
        logger.info("提示：请确保网络连接正常，或手动下载模型到 models/embeddings/bge-m3/")


if __name__ == "__main__":
    # 运行基础测试
    test_basic_workflow()
    
    # 如果需要测试模型下载，取消下面的注释
    # test_without_model()
