"""
Tracklet implementation for StrongSORT
"""

import numpy as np 
from collections import deque
from .basetrack import BaseTrack, TrackState 
from .strongsort_kalman import NSAKalman

class Tracklet(BaseTrack):
    def __init__(self, tlwh, score, category, motion='strongsort'):
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.is_activated = False
        self.score = score
        self.category = category   
        self.motion = motion
        self.kalman_filter = NSAKalman()
        self.convert_func = self.tlwh_to_xyah
        self.kalman_filter.initialize(self.convert_func(self._tlwh))

    def predict(self):
        self.kalman_filter.predict(is_activated=self.state == TrackState.Tracked)
        self.time_since_update += 1

    def activate(self, frame_id):
        self.track_id = self.next_id()
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.frame_id = frame_id
        self.kalman_filter.update(self.convert_func(new_track.tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        self.frame_id = frame_id
        new_tlwh = new_track.tlwh
        self.score = new_track.score
        self.kalman_filter.update(self.convert_func(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True
        self.time_since_update = 0
    
    @property
    def tlwh(self):
        return self.xyah_to_tlwh()
    
    def xyah_to_tlwh(self):
        x = self.kalman_filter.kf.x 
        ret = x[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

class Tracklet_w_reid(Tracklet):
    def __init__(self, tlwh, score, category, motion='strongsort', 
                 feat=None, feat_history=50, keypoints=None):
        super().__init__(tlwh, score, category, motion)
        self.smooth_feat = None
        self.curr_feat = None
        self.features = deque([], maxlen=feat_history)
        if feat is not None:
            self.update_features(feat)
        self.keypoints = keypoints
        self.alpha = 0.9

    def update_features(self, feat, alpha=None):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            alpha_ = self.alpha if alpha is None else alpha
            self.smooth_feat = alpha_ * self.smooth_feat + (1 - alpha_) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def re_activate(self, new_track, frame_id, new_id=False):
        self.kalman_filter.update(self.convert_func(new_track.tlwh), new_track.score)
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.keypoints = new_track.keypoints
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        self.frame_id = frame_id
        new_tlwh = new_track.tlwh
        self.score = new_track.score
        self.keypoints = new_track.keypoints
        self.kalman_filter.update(self.convert_func(new_tlwh), self.score)
        self.state = TrackState.Tracked
        self.is_activated = True
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.time_since_update = 0