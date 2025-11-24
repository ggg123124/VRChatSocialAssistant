"""
好友档案存储模块

负责管理好友的静态与动态属性，提供档案的 CRUD 操作。
采用每个好友一个 JSON 文件的存储方式。
"""

import json
import os
import logging
from typing import Dict, List, Optional
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from .models import FriendProfile


logger = logging.getLogger(__name__)


class ProfileStoreError(Exception):
    """档案存储异常基类"""
    pass


class NotFoundError(ProfileStoreError):
    """好友不存在异常"""
    pass


class StorageError(ProfileStoreError):
    """存储操作失败异常"""
    pass


class ProfileStore:
    """
    好友档案存储管理器
    
    存储方式：每个好友一个 JSON 文件（data/profiles/{friend_id}.json）
    支持 LRU 缓存提升读取性能
    """
    
    def __init__(self, profiles_dir: str = "data/profiles", enable_cache: bool = True, max_cache_size: int = 100):
        """
        初始化档案存储
        
        Args:
            profiles_dir: 档案存储目录
            enable_cache: 是否启用缓存
            max_cache_size: 最大缓存数量
        """
        self.profiles_dir = Path(profiles_dir)
        self.enable_cache = enable_cache
        self.max_cache_size = max_cache_size
        
        # 确保目录存在
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        
        # 构建 name -> id 映射（用于按名称查找）
        self._name_to_id_map: Dict[str, str] = {}
        self._build_name_index()
        
        logger.info(f"ProfileStore initialized: dir={profiles_dir}, cache_enabled={enable_cache}")
    
    def _build_name_index(self):
        """构建名称到ID的映射索引"""
        try:
            for profile_id in self._list_all_ids():
                try:
                    profile = self._load_profile(profile_id)
                    if profile:
                        self._name_to_id_map[profile.name] = profile.id
                except Exception as e:
                    logger.warning(f"Failed to index profile {profile_id}: {e}")
        except Exception as e:
            logger.error(f"Failed to build name index: {e}")
    
    def _list_all_ids(self) -> List[str]:
        """
        列出所有好友ID
        
        Returns:
            好友ID列表
        """
        try:
            profile_files = list(self.profiles_dir.glob("*.json"))
            return [f.stem for f in profile_files]
        except Exception as e:
            logger.error(f"Failed to list profile IDs: {e}")
            return []
    
    def _get_profile_path(self, friend_id: str) -> Path:
        """获取档案文件路径"""
        return self.profiles_dir / f"{friend_id}.json"
    
    @lru_cache(maxsize=100)
    def _load_profile(self, friend_id: str) -> Optional[FriendProfile]:
        """
        从文件加载档案（带缓存）
        
        Args:
            friend_id: 好友ID
            
        Returns:
            好友档案对象，不存在返回 None
        """
        profile_path = self._get_profile_path(friend_id)
        
        if not profile_path.exists():
            return None
        
        try:
            with open(profile_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return FriendProfile.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load profile {friend_id}: {e}")
            raise StorageError(f"Failed to load profile: {e}")
    
    def _save_profile(self, friend_id: str, profile: FriendProfile) -> bool:
        """
        保存档案到文件
        
        Args:
            friend_id: 好友ID
            profile: 好友档案对象
            
        Returns:
            是否保存成功
        """
        profile_path = self._get_profile_path(friend_id)
        
        try:
            # 更新修改时间
            profile.updated_at = datetime.now()
            
            # 写入文件
            with open(profile_path, 'w', encoding='utf-8') as f:
                json.dump(profile.to_dict(), f, ensure_ascii=False, indent=2)
            
            # 清除缓存
            if self.enable_cache:
                self._load_profile.cache_clear()
            
            # 更新名称索引
            self._name_to_id_map[profile.name] = profile.id
            
            logger.debug(f"Profile saved: {friend_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save profile {friend_id}: {e}")
            raise StorageError(f"Failed to save profile: {e}")
    
    def create_profile(self, profile: FriendProfile) -> str:
        """
        创建新的好友档案
        
        Args:
            profile: 好友档案对象
            
        Returns:
            创建的好友ID
            
        Raises:
            StorageError: 创建失败
        """
        # 检查是否已存在
        if self._get_profile_path(profile.id).exists():
            raise StorageError(f"Profile already exists: {profile.id}")
        
        # 保存档案
        self._save_profile(profile.id, profile)
        logger.info(f"Profile created: {profile.id} (name={profile.name})")
        
        return profile.id
    
    def get_profile(self, friend_id: str) -> FriendProfile:
        """
        获取好友档案
        
        Args:
            friend_id: 好友ID
            
        Returns:
            好友档案对象
            
        Raises:
            NotFoundError: 好友不存在
            StorageError: 读取失败
        """
        profile = self._load_profile(friend_id)
        
        if profile is None:
            raise NotFoundError(f"Profile not found: {friend_id}")
        
        return profile
    
    def get_profile_by_name(self, name: str) -> Optional[FriendProfile]:
        """
        根据名称获取好友档案
        
        Args:
            name: 好友昵称
            
        Returns:
            好友档案对象，不存在返回 None
        """
        friend_id = self._name_to_id_map.get(name)
        if friend_id is None:
            return None
        
        try:
            return self.get_profile(friend_id)
        except NotFoundError:
            return None
    
    def update_profile(self, friend_id: str, updates: Dict[str, any]) -> bool:
        """
        更新好友档案
        
        Args:
            friend_id: 好友ID
            updates: 更新字段字典
            
        Returns:
            是否更新成功
            
        Raises:
            NotFoundError: 好友不存在
            ValueError: 更新字段不合法
            StorageError: 写入失败
        """
        # 加载现有档案
        profile = self.get_profile(friend_id)
        
        # 验证更新字段
        allowed_fields = {'name', 'preferences', 'avoid_topics', 'personality', 
                         'custom_notes', 'language_preference', 'last_seen', 'conversation_count'}
        invalid_fields = set(updates.keys()) - allowed_fields
        if invalid_fields:
            raise ValueError(f"Invalid update fields: {invalid_fields}")
        
        # 应用更新
        for key, value in updates.items():
            setattr(profile, key, value)
        
        # 保存档案
        return self._save_profile(friend_id, profile)
    
    def delete_profile(self, friend_id: str) -> bool:
        """
        删除好友档案
        
        Args:
            friend_id: 好友ID
            
        Returns:
            是否删除成功
            
        Raises:
            NotFoundError: 好友不存在
            StorageError: 删除失败
        """
        profile_path = self._get_profile_path(friend_id)
        
        if not profile_path.exists():
            raise NotFoundError(f"Profile not found: {friend_id}")
        
        try:
            # 从名称索引中移除
            profile = self._load_profile(friend_id)
            if profile and profile.name in self._name_to_id_map:
                del self._name_to_id_map[profile.name]
            
            # 删除文件
            profile_path.unlink()
            
            # 清除缓存
            if self.enable_cache:
                self._load_profile.cache_clear()
            
            logger.info(f"Profile deleted: {friend_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete profile {friend_id}: {e}")
            raise StorageError(f"Failed to delete profile: {e}")
    
    def list_all_profiles(self) -> List[FriendProfile]:
        """
        列出所有好友档案
        
        Returns:
            好友档案列表
        """
        profiles = []
        for friend_id in self._list_all_ids():
            try:
                profile = self._load_profile(friend_id)
                if profile:
                    profiles.append(profile)
            except Exception as e:
                logger.warning(f"Failed to load profile {friend_id}: {e}")
        
        return profiles
    
    def exists(self, friend_id: str) -> bool:
        """
        检查好友档案是否存在
        
        Args:
            friend_id: 好友ID
            
        Returns:
            是否存在
        """
        return self._get_profile_path(friend_id).exists()
    
    def count(self) -> int:
        """
        统计好友数量
        
        Returns:
            好友数量
        """
        return len(self._list_all_ids())
