"""
Base class of Tracklets, used in tracklet.py
"""
"""
轨迹基类，用于tracklet.py
"""

import numpy as np
from collections import OrderedDict

# 定义轨迹状态枚举
class TrackState(object):
    New = 0       # 新建
    Tracked = 1   # 跟踪中
    Lost = 2      # 丢失
    Removed = 3   # 移除

# 轨迹基类
class BaseTrack(object):
    _count = 0  # 轨迹ID计数器

    # 轨迹属性
    track_id = 0          # 轨迹ID
    is_activated = False  # 是否激活
    state = TrackState.New # 当前状态
    
    history = OrderedDict() # 历史记录
    features = []          # 特征列表
    curr_feature = None    # 当前特征
    score = 0              # 得分
    start_frame = 0        # 起始帧
    frame_id = 0           # 当前帧ID
    time_since_update = 0  # 自上次更新后的帧数
    
    # 多摄像头跟踪
    location = (np.inf, np.inf)  # 位置信息

    # 属性方法
    @property
    def end_frame(self):
        return self.frame_id  # 结束帧

    # 静态方法
    @staticmethod
    def next_id():
        BaseTrack._count += 1  # 获取下一个ID
        return BaseTrack._count
    
    @staticmethod
    def clear_count():
        BaseTrack._count = 0  # 重置ID计数器

    # 抽象方法
    def activate(self, *args):
        raise NotImplementedError  # 激活轨迹

    def predict(self):
        raise NotImplementedError  # 预测下一状态

    def update(self, *args, **kwargs):
        raise NotImplementedError  # 更新状态

    def mark_lost(self):
        self.state = TrackState.Lost  # 标记为丢失

    def mark_removed(self):
        self.state = TrackState.Removed  # 标记为移除
        
    # 边界框格式转换方法
    @property
    def tlwh(self):
        """获取当前边界框(top left x, top left y, width, height)格式"""
        raise NotImplementedError

    @property
    def tlbr(self):
        """转换为(min x, min y, max x, max y)格式"""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret
        
    @property
    def xywh(self):
        """转换为(center x, center y, width, height)格式"""
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        return ret

    # 静态转换方法
    @staticmethod
    def tlwh_to_xyah(tlwh):
        """转换为(center x, center y, aspect ratio, height)格式"""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """转换为(center x, center y, width, height)格式"""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret
    
    @staticmethod
    def tlwh_to_xysa(tlwh):
        """转换为(center x, center y, area, aspect ratio)格式"""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] = tlwh[2] * tlwh[3]  # 面积
        ret[3] = tlwh[2] / tlwh[3]   # 宽高比
        return ret
    
    @staticmethod
    def tlbr_to_tlwh(tlbr):
        """(min x, min y, max x, max y)转(top left x, top left y, width, height)"""
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        """(top left x, top left y, width, height)转(min x, min y, max x, max y)"""
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret
    
    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)  # 转换为xyah格式
    
    def to_xywh(self):
        return self.tlwh_to_xywh(self.tlwh)  # 转换为xywh格式

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)  # 对象表示