# osnet_rdk_x5_inference_fixed.py
import cv2
import numpy as np
import os
import time
from hobot_dnn import pyeasy_dnn as dnn

class OSNetRDKX5Inference:
    def __init__(self, bin_model_path, input_size=(64, 128)):
        """
        初始化OSNet RDK X5推理器（修复版本）
        """
        self.input_size = input_size
        self.width, self.height = input_size
        
        if not os.path.exists(bin_model_path):
            raise FileNotFoundError(f"模型文件不存在: {bin_model_path}")
        
        # 预分配内存
        self.nv12_buffer = np.zeros((self.height * 3 // 2, self.width), dtype=np.uint8)
        self.y_plane = self.nv12_buffer[:self.height, :]
        self.uv_plane = self.nv12_buffer[self.height:, :].reshape(self.height // 2, self.width // 2, 2)
        
        start_time = time.time()
        try:
            self.models = dnn.load(bin_model_path)
            self.model = self.models[0]
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {e}")
        
        print(f"RDK X5模型加载成功: {bin_model_path}")
        print(f"模型加载耗时: {(time.time() - start_time)*1000:.2f}ms")
        
        # 获取模型信息
        try:
            self.input_tensor = self.model.inputs[0]
            self.output_tensor = self.model.outputs[0]
            input_shape = self.input_tensor.properties.shape
            output_shape = self.output_tensor.properties.shape
            print(f"输入形状: {input_shape}")
            print(f"输出形状: {output_shape}")
            print(f"输入数据类型: {self.input_tensor.properties.dtype}")
            print(f"输出数据类型: {self.output_tensor.properties.dtype}")
            
        except Exception as e:
            print(f"获取模型信息失败: {e}")
        
        # 性能统计
        self.total_preprocess_time = 0
        self.total_inference_time = 0
        self.process_count = 0
    
    # def bgr2nv12_fastest(self, image):
    #     """
    #     最快的BGR到NV12转换方法（修复版本）
    #     """
    #     # 使用OpenCV的高效函数
    #     resized = cv2.resize(image, self.input_size, interpolation=cv2.INTER_LINEAR)
        
    #     # 直接转换到YUV_I420
    #     yuv_i420 = cv2.cvtColor(resized, cv2.COLOR_BGR2YUV_I420)
        
    #     height, width = self.height, self.width
    #     y_size = width * height
        
    #     # YUV_I420格式：YYYYYYYY UUUU VVVV
    #     # 总大小应该是：width * height * 3 // 2
    #     total_size = width * height * 3 // 2
    #     if len(yuv_i420) != total_size:
    #         # 如果大小不匹配，重新调整
    #         yuv_i420 = yuv_i420[:total_size]
        
    #     # 提取Y分量 (前width*height个字节)
    #     y_plane = yuv_i420[:y_size].reshape(height, width)
        
    #     # 提取U分量 (接下来的width*height//4个字节)
    #     u_start = y_size
    #     u_end = u_start + (width * height // 4)
    #     u_plane = yuv_i420[u_start:u_end].reshape(height // 2, width // 2)
        
    #     # 提取V分量 (最后的width*height//4个字节)
    #     v_start = u_end
    #     v_plane = yuv_i420[v_start:].reshape(height // 2, width // 2)
        
    #     # 复制Y平面
    #     np.copyto(self.y_plane, y_plane)
        
    #     # 交错UV平面
    #     self.uv_plane[:, :, 0] = u_plane
    #     self.uv_plane[:, :, 1] = v_plane
        
    #     return self.nv12_buffer
    
    def bgr2nv12_simple_fast(self, image):
        """
        简单快速的BGR到NV12转换方法
        """
        # 调整大小
        resized = cv2.resize(image, self.input_size, interpolation=cv2.INTER_LINEAR)
        
        # 转换为YUV
        yuv = cv2.cvtColor(resized, cv2.COLOR_BGR2YUV)
        
        # 复制Y平面
        np.copyto(self.y_plane, yuv[:, :, 0])
        
        # 下采样UV分量并交错
        u_down = cv2.resize(yuv[:, :, 1], (self.width // 2, self.height // 2), 
                           interpolation=cv2.INTER_LINEAR)
        v_down = cv2.resize(yuv[:, :, 2], (self.width // 2, self.height // 2), 
                           interpolation=cv2.INTER_LINEAR)
        
        # 交错排列UV
        self.uv_plane[:, :, 0] = u_down
        self.uv_plane[:, :, 1] = v_down
        
        return self.nv12_buffer
    
    # def bgr2nv12_optimized_final(self, image):
    #     """
    #     最终优化的BGR到NV12转换方法
    #     """
    #     # 一次性调整大小和转换
    #     resized = cv2.resize(image, self.input_size, interpolation=cv2.INTER_LINEAR)
        
    #     # 使用更高效的方法：先转换为YUV，然后手动处理UV分量
    #     yuv = cv2.cvtColor(resized, cv2.COLOR_BGR2YUV)
        
    #     # Y平面
    #     np.copyto(self.y_plane, yuv[:, :, 0])
        
    #     # UV平面 - 使用快速下采样
    #     u_channel = yuv[:, :, 1]
    #     v_channel = yuv[:, :, 2]
        
    #     # 快速2x2平均下采样
    #     for i in range(0, self.height, 2):
    #         for j in range(0, self.width, 2):
    #             u_val = (u_channel[i, j] + u_channel[i, j+1] + 
    #                     u_channel[i+1, j] + u_channel[i+1, j+1]) // 4
    #             v_val = (v_channel[i, j] + v_channel[i, j+1] + 
    #                     v_channel[i+1, j] + v_channel[i+1, j+1]) // 4
                
    #             self.uv_plane[i//2, j//2, 0] = u_val
    #             self.uv_plane[i//2, j//2, 1] = v_val
        
    #     return self.nv12_buffer
    
    def preprocess_image_optimized(self, image):
        """
        优化的图像预处理
        """
        start_time = time.perf_counter()
        
        # 使用最稳定的转换方法
        nv12 = self.bgr2nv12_simple_fast(image)
        
        # 直接返回视图，避免额外复制
        processed = nv12[np.newaxis, :]  # 添加batch维度
        
        self.total_preprocess_time += (time.perf_counter() - start_time) * 1000
        return processed
    
    def preprocess_batch_optimized(self, images):
        """
        优化的批量预处理
        """
        processed_images = np.zeros((len(images), self.height * 3 // 2, self.width), dtype=np.uint8)
        
        for i, img in enumerate(images):
            nv12 = self.bgr2nv12_simple_fast(img)
            processed_images[i] = nv12
        
        return processed_images
    
    def extract_features(self, image):
        """
        提取单张图像的特征（优化版本）
        """
        try:
            # 优化预处理
            input_tensor = self.preprocess_image_optimized(image)
            
            # 推理
            start_time = time.perf_counter()
            outputs = self.model.forward(input_tensor)
            self.total_inference_time += (time.perf_counter() - start_time) * 1000
            self.process_count += 1
            
            # 提取特征
            if hasattr(outputs[0], 'buffer'):
                features = outputs[0].buffer
            else:
                features = outputs[0]
                
            features = np.squeeze(features)
            if features.ndim > 1:
                features = features.flatten()
            
            return features
            
        except Exception as e:
            print(f"特征提取错误: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def extract_features_batch(self, images):
        """
        提取批量图像的特征（优化版本）
        """
        try:
            # 优化批量预处理
            input_tensor = self.preprocess_batch_optimized(images)
            
            # 推理
            start_time = time.perf_counter()
            outputs = self.model.forward(input_tensor)
            self.total_inference_time += (time.perf_counter() - start_time) * 1000
            self.process_count += len(images)
            
            # 提取特征
            if hasattr(outputs[0], 'buffer'):
                features = outputs[0].buffer
            else:
                features = outputs[0]
            
            if features.ndim == 4:
                features = features.reshape(features.shape[0], features.shape[1])
            
            return features
            
        except Exception as e:
            print(f"批量特征提取错误: {e}")
            import traceback
            traceback.print_exc()
            raise

    # # 其他方法保持不变...
    # def extract_features_from_bboxes(self, bboxes, ori_img, input_format='tlwh'):
    #     bboxes = bboxes.copy()
    #     img_h, img_w = ori_img.shape[:2]
        
    #     if input_format == 'tlwh':
    #         bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
    #         bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
        
    #     crops = []
    #     for box in bboxes:
    #         x1, y1, x2, y2 = box.round().astype('int')
    #         x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)
            
    #         if x2 > x1 and y2 > y1:
    #             crop = ori_img[y1:y2, x1:x2]
    #             crops.append(crop)
        
    #     if not crops:
    #         return np.array([]), []
        
    #     features = self.extract_features_batch(crops)
    #     return features, crops
    
    def compute_similarity(self, features1, features2):
        if features1 is None or features2 is None:
            return 0.0
            
        if len(features1.shape) == 1:
            norm1 = np.linalg.norm(features1)
            norm2 = np.linalg.norm(features2)
            if norm1 > 0 and norm2 > 0:
                features1 = features1 / norm1
                features2 = features2 / norm2
                similarity = np.dot(features1, features2)
            else:
                similarity = 0.0
        else:
            norm1 = np.linalg.norm(features1, axis=1, keepdims=True)
            norm2 = np.linalg.norm(features2, axis=1, keepdims=True)
            features1 = features1 / np.where(norm1 > 0, norm1, 1)
            features2 = features2 / np.where(norm2 > 0, norm2, 1)
            similarity = np.dot(features1, features2.T)
        
        return similarity
    
    def get_performance_stats(self):
        if self.process_count == 0:
            return {
                'avg_preprocess_time': 0,
                'avg_inference_time': 0,
                'avg_total_time': 0,
                'total_process_count': 0
            }
        
        avg_preprocess = self.total_preprocess_time / self.process_count
        avg_inference = self.total_inference_time / self.process_count
        
        return {
            'avg_preprocess_time': avg_preprocess,
            'avg_inference_time': avg_inference,
            'avg_total_time': avg_preprocess + avg_inference,
            'total_process_count': self.process_count
        }

# 测试优化效果
if __name__ == "__main__":
    # 初始化RDK X5推理器
    rdk_inference = OSNetRDKX5Inference(
        bin_model_path="osnet_64x128_nv12.bin",
        input_size=(64, 128)
    )
    
    # 预热
    warmup_img = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
    for _ in range(3):
        _ = rdk_inference.extract_features(warmup_img)
    
    # 正式测试
    image1 = cv2.imread("./test_image/0001_c5_f0051487.jpg")
    if image1 is not None:
        print(f"原始图像形状: {image1.shape}")
        
        # 多次测试取平均
        test_runs = 10
        preprocess_times = []
        
        for i in range(test_runs):
            start_time = time.perf_counter()
            features1 = rdk_inference.extract_features(image1)
            end_time = time.perf_counter()
            if i > 0:  # 跳过第一次（可能有初始化开销）
                preprocess_times.append((end_time - start_time) * 1000)
        
        avg_time = np.mean(preprocess_times[1:]) if len(preprocess_times) > 1 else 0
        print(f"平均处理时间: {avg_time:.2f}ms")
        print(f"特征向量形状: {features1.shape}")
        print(f"特征向量范数: {np.linalg.norm(features1)}")
        
        

    # 示例1: 提取单张图像特征
    image1 = cv2.imread("./test_image/0001_c5_f0051487.jpg")
    if image1 is not None:
        print(f"原始图像形状: {image1.shape}")
        features1 = rdk_inference.extract_features(image1)
        print(f"特征向量形状: {features1.shape}")
        print(f"特征向量范数: {np.linalg.norm(features1)}")
        print(f"特征数据类型: {features1.dtype}")
    else:
        print("无法加载图像1")
    
    image2 = cv2.imread("./test_image/0001_c5_f0051487.jpg")
    if image2 is not None:
        features2 = rdk_inference.extract_features(image2)
        print(f"特征向量形状: {features2.shape}")
        print(f"特征向量范数: {np.linalg.norm(features2)}")
    else:
        print("无法加载图像2")
    
    # 示例2: 计算相似度
    if image1 is not None and image2 is not None:
        similarity = rdk_inference.compute_similarity(features1, features2)
        print(f"相似度: {similarity:.4f}")
    

    # # 输出性能统计
    # stats = rdk_inference.get_performance_stats()
    # print(f"\n性能统计:")
    # print(f"平均预处理时间: {stats['avg_preprocess_time']:.2f}ms")
    # print(f"平均推理时间: {stats['avg_inference_time']:.2f}ms")
    # print(f"平均总时间: {stats['avg_total_time']:.2f}ms")
    # print(f"总处理次数: {stats['total_process_count']}")
