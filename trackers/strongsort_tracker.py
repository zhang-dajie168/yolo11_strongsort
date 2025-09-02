"""
Strong Sort
"""
import numpy as np  
import cv2 
from .basetrack import BaseTrack, TrackState
from .tracklet import Tracklet, Tracklet_w_reid
from .matching import *
from .osnet_rdk_x5_inference import OSNetRDKX5Inference
from .basetracker import BaseTracker

class StrongSortTracker(BaseTracker):
    def __init__(self, args, frame_rate=30):
        super().__init__(args, frame_rate=frame_rate)
        self.with_reid = True  # in strong sort, reid model must be included
        self.reid_model = None
        if self.with_reid:
            self.reid_model = OSNetRDKX5Inference(args.reid_model_path,input_size=(64, 128))
            print("MobileNetV2 ReID模型加载完成")
            
        self.bbox_crop_size = (256, 128)
        self.lambda_ = 0.98  # the coef of cost mix in eq. 10 in paper 成本混合系数
        BaseTrack.clear_count()   # 重置轨迹ID计数器
        
    def get_feature(self, tlwhs, ori_img, crop_size=None):
    #"使用OSNet提取特征"
        if len(tlwhs) == 0:
            return []
            
        features = []
        for tlwh in tlwhs:
            x1, y1, w, h = map(int, tlwh)
            x2, y2 = x1 + w, y1 + h
            
            # 边界检查
            h_img, w_img = ori_img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)
            
            if x2 <= x1 or y2 <= y1:
                features.append(None)
                continue
                
            crop = ori_img[y1:y2, x1:x2]
            try:
                feat = self.reid_model.extract_features(crop)
                features.append(feat)
            except Exception as e:
                print(f"特征提取失败: {e}")
                features.append(None)
                
        return features
    
    def update(self, output_results, img, ori_img, keypoints_list=None):
        """更新跟踪状态"""
        self.frame_id += 1
        activated_tracklets = []
        refind_tracklets = []
        lost_tracklets = []
        removed_tracklets = []

        scores = output_results[:, 4]
        bboxes = output_results[:, :4]
        categories = output_results[:, -1]

        remain_inds = scores > self.args.conf_thresh
        dets = bboxes[remain_inds]
        cates = categories[remain_inds]
        scores_keep = scores[remain_inds]
        
        keypoints_keep = [keypoints_list[i] for i in np.where(remain_inds)[0]] if keypoints_list else [None] * len(dets)
        features_keep = self.get_feature(tlwhs=dets[:, :4], ori_img=ori_img, crop_size=self.args.reid_crop_size)

        if len(dets) > 0:
            detections = [Tracklet_w_reid(tlwh, s, cate, motion=self.motion, feat=feat, keypoints=kps) for
                (tlwh, s, cate, feat, kps) in zip(dets, scores_keep, cates, features_keep, keypoints_keep)]
        else:
            detections = []

        unconfirmed = []
        tracked_tracklets = []
        for track in self.tracked_tracklets:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_tracklets.append(track)

        tracklet_pool = BaseTracker.joint_tracklets(tracked_tracklets, self.lost_tracklets)
        for tracklet in tracklet_pool:
            tracklet.predict()

        cost_matrix = self.gated_metric(tracklet_pool, detections)
        matches, u_track, u_detection = linear_assignment(cost_matrix, thresh=0.9)

        for itracked, idet in matches:
            track = tracklet_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_tracklets.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_tracklets.append(track)

        tracklet_for_iou = [tracklet_pool[i] for i in u_track if tracklet_pool[i].state == TrackState.Tracked]
        detection_for_iou = [detections[i] for i in u_detection]

        dists = iou_distance(tracklet_for_iou, detection_for_iou)
        matches, u_track, u_detection = linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = tracklet_for_iou[itracked]
            det = detection_for_iou[idet]
            if track.state == TrackState.Tracked:
                track.update(detection_for_iou[idet], self.frame_id)
                activated_tracklets.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_tracklets.append(track)

        for it in u_track:
            track = tracklet_for_iou[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_tracklets.append(track)

        detections = [detection_for_iou[i] for i in u_detection]
        dists = iou_distance(unconfirmed, detections)
       
        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7)

        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_tracklets.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_tracklets.append(track)

        for inew in u_detection:
            track = detections[inew]
            if track.score < self.init_thresh:
                continue
            track.activate(self.frame_id)
            activated_tracklets.append(track)

        for track in self.lost_tracklets:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_tracklets.append(track)

        self.tracked_tracklets = [t for t in self.tracked_tracklets if t.state == TrackState.Tracked]
        self.merge_tracklets(activated_tracklets, refind_tracklets, lost_tracklets, removed_tracklets)

        output_tracklets = [track for track in self.tracked_tracklets if track.is_activated]
        return output_tracklets
    
    def gated_metric(self, tracks, dets):
        apperance_dist = embedding_distance(tracks=tracks, detections=dets, metric='cosine')
        cost_matrix = self.gate_cost_matrix(apperance_dist, tracks, dets)
        return cost_matrix
    
    def gate_cost_matrix(self, cost_matrix, tracks, dets, max_apperance_thresh=0.15, gated_cost=1e5, only_position=False):
        gating_dim = 2 if only_position else 4
        gating_threshold = chi2inv95[gating_dim]
        measurements = np.asarray([Tracklet.tlwh_to_xyah(det.tlwh) for det in dets])

        cost_matrix[cost_matrix > max_apperance_thresh] = gated_cost
        
        for row, track in enumerate(tracks):
            gating_distance = track.kalman_filter.gating_distance(measurements)
            cost_matrix[row, gating_distance > gating_threshold] = gated_cost
            cost_matrix[row] = self.lambda_ * cost_matrix[row] + (1 - self.lambda_) * gating_distance
        return cost_matrix