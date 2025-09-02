import cv2
import numpy as np
from scipy.special import softmax
from hobot_dnn import pyeasy_dnn as dnn
import logging

class YOLOPose:
    def __init__(self, model_path, conf_thres=0.25, iou_thres=0.45):
        self.model_path = model_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.logger = logging.getLogger("YOLOv11_Pose")
        
        # 初始化模型
        self.model = self._load_model()
        
        # COCO类别名称
        self.coco_names = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", 
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", 
            "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
            "scissors", "teddy bear", "hair drier", "toothbrush"
        ]
    
    def _load_model(self):
        """加载BPU量化模型"""
        try:
            model = dnn.load(self.model_path)
            self.logger.info(f"成功加载YOLOv11 Pose模型: {self.model_path}")
            
            # 获取输入输出形状
            input_shape = model[0].inputs[0].properties.shape[2:4]
            self.input_height, self.input_width = input_shape
            self.logger.info(f"模型输入尺寸: {input_shape}")
            
            # 准备反量化系数
            self.s_bboxes_scale = model[0].outputs[0].properties.scale_data[np.newaxis, :]
            self.m_bboxes_scale = model[0].outputs[2].properties.scale_data[np.newaxis, :]
            self.l_bboxes_scale = model[0].outputs[4].properties.scale_data[np.newaxis, :]
            
            # DFL系数
            self.weights_static = np.array([i for i in range(16)]).astype(np.float32)[np.newaxis, np.newaxis, :]
            
            # anchors
            self.s_anchor = np.stack([np.tile(np.linspace(0.5, 79.5, 80), reps=80), 
                                np.repeat(np.arange(0.5, 80.5, 1), 80)], axis=0).transpose(1,0)
            self.m_anchor = np.stack([np.tile(np.linspace(0.5, 39.5, 40), reps=40), 
                                np.repeat(np.arange(0.5, 40.5, 1), 40)], axis=0).transpose(1,0)
            self.l_anchor = np.stack([np.tile(np.linspace(0.5, 19.5, 20), reps=20), 
                                np.repeat(np.arange(0.5, 20.5, 1), 20)], axis=0).transpose(1,0)
            
            # 置信度阈值转换
            self.conf_inverse = -np.log(1/self.conf_thres - 1)
            
            return model
        except Exception as e:
            self.logger.error(f"加载模型失败: {e}")
            raise e
    
    def preprocess(self, img):
        """预处理图像为NV12格式"""
        img_resized = cv2.resize(img, (self.input_width, self.input_height), interpolation=cv2.INTER_NEAREST)
        height, width = img_resized.shape[0], img_resized.shape[1]
        area = height * width
        yuv420p = cv2.cvtColor(img_resized, cv2.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))
        y = yuv420p[:area]
        uv_planar = yuv420p[area:].reshape((2, area // 4))
        uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))
        nv12 = np.zeros_like(yuv420p)
        nv12[:height * width] = y
        nv12[height * width:] = uv_packed
        return nv12
    
    def detect(self, img):
        """执行检测"""
        # 预处理
        input_tensor = self.preprocess(img)
        
        # 前向推理
        outputs = self.model[0].forward(input_tensor)
        outputs = [output.buffer for output in outputs]
        
        # 后处理
        detections, keypoints = self.postprocess(outputs, img.shape[1], img.shape[0])
        
        return detections, keypoints
    
    def postprocess(self, outputs, img_width, img_height):
        """后处理输出结果"""
        # 计算缩放比例
        y_scale = img_height / self.input_height
        x_scale = img_width / self.input_width
        
        # reshape输出
        s_bboxes = outputs[0].reshape(-1, 64)
        s_clses = outputs[1].reshape(-1, 1)
        m_bboxes = outputs[2].reshape(-1, 64)
        m_clses = outputs[3].reshape(-1, 1)
        l_bboxes = outputs[4].reshape(-1, 64)
        l_clses = outputs[5].reshape(-1, 1)
        s_kpts = outputs[6].reshape(-1, 51)
        m_kpts = outputs[7].reshape(-1, 51)
        l_kpts = outputs[8].reshape(-1, 51)

        # 分类分支处理
        s_max_scores = np.max(s_clses, axis=1)
        s_valid_indices = np.flatnonzero(s_max_scores >= self.conf_inverse)
        s_ids = np.argmax(s_clses[s_valid_indices, :], axis=1)
        s_scores = 1 / (1 + np.exp(-s_max_scores[s_valid_indices]))

        m_max_scores = np.max(m_clses, axis=1)
        m_valid_indices = np.flatnonzero(m_max_scores >= self.conf_inverse)
        m_ids = np.argmax(m_clses[m_valid_indices, :], axis=1)
        m_scores = 1 / (1 + np.exp(-m_max_scores[m_valid_indices]))

        l_max_scores = np.max(l_clses, axis=1)
        l_valid_indices = np.flatnonzero(l_max_scores >= self.conf_inverse)
        l_ids = np.argmax(l_clses[l_valid_indices, :], axis=1)
        l_scores = 1 / (1 + np.exp(-l_max_scores[l_valid_indices]))

        # 边界框处理
        s_bboxes_float32 = s_bboxes[s_valid_indices,:].astype(np.float32) * self.s_bboxes_scale
        s_ltrb = np.sum(softmax(s_bboxes_float32.reshape(-1, 4, 16), axis=2) * self.weights_static, axis=2)
        s_anchor = self.s_anchor[s_valid_indices, :]
        s_x1y1 = s_anchor - s_ltrb[:, 0:2]
        s_x2y2 = s_anchor + s_ltrb[:, 2:4]
        s_dbboxes = np.hstack([s_x1y1, s_x2y2])*8.0

        m_bboxes_float32 = m_bboxes[m_valid_indices,:].astype(np.float32) * self.m_bboxes_scale
        m_ltrb = np.sum(softmax(m_bboxes_float32.reshape(-1, 4, 16), axis=2) * self.weights_static, axis=2)
        m_anchor = self.m_anchor[m_valid_indices, :]
        m_x1y1 = m_anchor - m_ltrb[:, 0:2]
        m_x2y2 = m_anchor + m_ltrb[:, 2:4]
        m_dbboxes = np.hstack([m_x1y1, m_x2y2])*16.0

        l_bboxes_float32 = l_bboxes[l_valid_indices,:].astype(np.float32) * self.l_bboxes_scale
        l_ltrb = np.sum(softmax(l_bboxes_float32.reshape(-1, 4, 16), axis=2) * self.weights_static, axis=2)
        l_anchor = self.l_anchor[l_valid_indices,:]
        l_x1y1 = l_anchor - l_ltrb[:, 0:2]
        l_x2y2 = l_anchor + l_ltrb[:, 2:4]
        l_dbboxes = np.hstack([l_x1y1, l_x2y2])*32.0

        # 关键点处理
        s_kpts = s_kpts[s_valid_indices,:].reshape(-1, 17, 3)
        s_kpts_xy = (s_kpts[:, :, :2] * 2.0 + (self.s_anchor[s_valid_indices,:][:,np.newaxis,:] - 0.5)) * 8.0
        s_kpts_score = s_kpts[:, :, 2:3]

        m_kpts = m_kpts[m_valid_indices,:].reshape(-1, 17, 3)
        m_kpts_xy = (m_kpts[:, :, :2] * 2.0 + (self.m_anchor[m_valid_indices,:][:,np.newaxis,:] - 0.5)) * 16.0
        m_kpts_score = m_kpts[:, :, 2:3]

        l_kpts = l_kpts[l_valid_indices,:].reshape(-1, 17, 3)
        l_kpts_xy = (l_kpts[:, :, :2] * 2.0 + (self.l_anchor[l_valid_indices,:][:,np.newaxis,:] - 0.5)) * 32.0
        l_kpts_score = l_kpts[:, :, 2:3]

        # 合并结果
        dbboxes = np.concatenate((s_dbboxes, m_dbboxes, l_dbboxes), axis=0)
        scores = np.concatenate((s_scores, m_scores, l_scores), axis=0)
        ids = np.concatenate((s_ids, m_ids, l_ids), axis=0)
        kpts_xy = np.concatenate((s_kpts_xy, m_kpts_xy, l_kpts_xy), axis=0)
        kpts_score = np.concatenate((s_kpts_score, m_kpts_score, l_kpts_score), axis=0)

        # NMS
        indices = cv2.dnn.NMSBoxes(dbboxes, scores, self.conf_thres, self.iou_thres)

        # 缩放回原图尺寸
        bboxes = dbboxes[indices] * np.array([x_scale, y_scale, x_scale, y_scale])
        bboxes = bboxes.astype(np.int32)
        kpts_xy = kpts_xy[indices] * np.array([[x_scale, y_scale]])
        kpts_score = kpts_score[indices]

        # 格式化检测结果
        formatted_dets = []
        keypoints_list = []
        for i in range(len(ids[indices])):
            x1, y1, x2, y2 = bboxes[i]
            score = scores[indices][i]
            class_id = ids[indices][i]
            
            # 只保留person类(class_id=0)
            if class_id == 0:
                formatted_dets.append([x1, y1, x2-x1, y2-y1, score, class_id])
                
                # 格式化关键点 (17, 3)
                kpts = np.zeros((17, 3))
                for j in range(17):
                    kpts[j, 0] = kpts_xy[i, j, 0]
                    kpts[j, 1] = kpts_xy[i, j, 1]
                    kpts[j, 2] = kpts_score[i, j, 0]
                keypoints_list.append(kpts)

        return np.array(formatted_dets), keypoints_list
