"""
Base class for every tracker, embedded some general codes
like init, get_features and tracklet merge
for code clearity
"""
"""
跟踪器基类，包含初始化、特征提取和轨迹合并等通用代码
"""

import numpy as np 
import cv2
from .matching import iou_distance


class BaseTracker(object):
    def __init__(self, args, frame_rate=30):
        # 初始化跟踪器
        self.tracked_tracklets = []  # 跟踪中的轨迹
        self.lost_tracklets = []     # 丢失的轨迹
        self.removed_tracklets = []  # 移除的轨迹

        self.frame_id = 0  # 当前帧ID
        self.args = args   # 参数

        # 阈值参数
        self.init_thresh = args.init_thresh  # 初始化阈值
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)  # 缓冲区大小
        self.max_time_lost = self.buffer_size  # 最大丢失时间

        self.motion = args.kalman_format  # 运动模型类型

    def update(self, output_results, img, ori_img):
        raise NotImplementedError  # 更新跟踪状态

    def crop_and_resize(self, bboxes, ori_img, input_format='tlwh', sz=(128, 64), 
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        """
        裁剪并调整边界框大小
        """
        if len(bboxes) == 0:
            return []
            
        bboxes = bboxes.copy()
        
        # 转换bbox格式为xyxy
        if input_format == 'tlwh':
            bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
            bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
        
        img_h, img_w = ori_img.shape[:2]
        crops = []
        
        for box in bboxes:
            x1, y1, x2, y2 = box.round().astype('int')
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)
            
            if x2 <= x1 or y2 <= y1:
                crops.append(None)
                continue
                
            crop = ori_img[y1:y2, x1:x2]
            crop = cv2.resize(crop, sz, interpolation=cv2.INTER_LINEAR)
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            
            # 不进行归一化，直接返回uint8图像
            crops.append(crop)
            
        return crops

    def get_feature(self, tlwhs, ori_img, crop_size=[128, 64]):
        """
        获取目标的外观特征
        tlwhs: 边界框数组 (num_of_objects, 4)
        ori_img: 原始图像 (H, W, C)
        crop_size: 裁剪尺寸 [宽, 高]
        """
        if len(tlwhs) == 0:
            return []
            
        # 裁剪并调整大小
        crop_bboxes = self.crop_and_resize(tlwhs, ori_img, input_format='tlwh', sz=(crop_size[1], crop_size[0]))
        
        # 提取特征
        features = []
        for crop in crop_bboxes:
            if crop is None:
                features.append(None)
                continue
            # 这里调用ReID模型提取特征
            # 假设self.reid_model.extract_features接受numpy数组
            feat = self.reid_model.extract_features(crop)
            features.append(feat)
            
        return features
    
    def merge_tracklets(self, activated_tracklets, refind_tracklets, lost_tracklets, removed_tracklets):
        """
        根据当前关联结果更新轨迹列表
        """
        # 合并激活和重新找到的轨迹
        self.tracked_tracklets = BaseTracker.joint_tracklets(self.tracked_tracklets, activated_tracklets)
        self.tracked_tracklets = BaseTracker.joint_tracklets(self.tracked_tracklets, refind_tracklets)
        # 更新丢失和移除的轨迹
        self.lost_tracklets = BaseTracker.sub_tracklets(self.lost_tracklets, self.tracked_tracklets)
        self.lost_tracklets.extend(lost_tracklets)
        self.lost_tracklets = BaseTracker.sub_tracklets(self.lost_tracklets, self.removed_tracklets)
        self.removed_tracklets.extend(removed_tracklets)
        # 移除重复轨迹
        self.tracked_tracklets, self.lost_tracklets = BaseTracker.remove_duplicate_tracklets(self.tracked_tracklets, self.lost_tracklets)

    @staticmethod
    def joint_tracklets(tlista, tlistb):
        """合并两个轨迹列表"""
        exists = {}
        res = []
        # 添加第一个列表
        for t in tlista:
            exists[t.track_id] = 1
            res.append(t)
        # 添加第二个列表中不重复的轨迹
        for t in tlistb:
            tid = t.track_id
            if not exists.get(tid, 0):
                exists[tid] = 1
                res.append(t)
        return res

    @staticmethod
    def sub_tracklets(tlista, tlistb):
        """从tlista中移除tlistb中的轨迹"""
        tracklets = {}
        for t in tlista:
            tracklets[t.track_id] = t
        for t in tlistb:
            tid = t.track_id
            if tracklets.get(tid, 0):
                del tracklets[tid]
        return list(tracklets.values())

    @staticmethod
    def remove_duplicate_tracklets(trackletsa, trackletsb):
        """移除重复的轨迹"""
        pdist = iou_distance(trackletsa, trackletsb)  # 计算IoU距离
        pairs = np.where(pdist < 0.15)  # 找到IoU小于0.15的轨迹对
        dupa, dupb = list(), list()
        # 根据存活时间决定保留哪个轨迹
        for p, q in zip(*pairs):
            timep = trackletsa[p].frame_id - trackletsa[p].start_frame
            timeq = trackletsb[q].frame_id - trackletsb[q].start_frame
            if timep > timeq:
                dupb.append(q)
            else:
                dupa.append(p)
        # 过滤掉重复轨迹
        resa = [t for i, t in enumerate(trackletsa) if not i in dupa]
        resb = [t for i, t in enumerate(trackletsb) if not i in dupb]
        return resa, resb
