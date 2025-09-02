import os
import cv2
import numpy as np
#from tqdm import tqdm
#from loguru import logger
import argparse
import time
from tqdm import tqdm
from loguru import logger
# Import StrongSORT tracker
from trackers.strongsort_tracker import StrongSortTracker
from tracking_utils.visualization import plot_img

# Import our YOLOv11 Pose wrapper
from trackers.YOLO_Pose import YOLOPose

def get_args():
    parser = argparse.ArgumentParser()

    # Input/output
    parser.add_argument("--source", type=str, required=True, help="视频文件或图片文件夹路径")
    parser.add_argument("--output", type=str, default="track_results", help="输出文件夹路径")

    # Detection parameters
    parser.add_argument(
        "--pose_model_path", 
        type=str,
        default="/home/sunrise/yolo11_strongsort/weights/yolov8n_pose_bayese_640x640_nv12_modified.bin",
        help="YOLOv11 Pose模型路径"
    )
    parser.add_argument("--conf_thresh", type=float, default=0.25, help="检测置信度阈值")
    parser.add_argument("--iou_thresh", type=float, default=0.45, help="检测IOU阈值")

    # Tracking parameters
    parser.add_argument("--track_buffer", type=int, default=30, help="跟踪缓冲区大小")
    parser.add_argument("--min_area", type=int, default=100, help="跟踪的最小边界框面积")
    parser.add_argument("--init_thresh", type=float, default=0.3, help="初始化新轨迹的阈值")
    parser.add_argument("--max_time_lost", type=int, default=60, help="最大丢失帧数")
    parser.add_argument("--gamma", type=float, default=0.1, help="外观和运动特征的融合系数")
    parser.add_argument("--nms_thresh", type=float, default=0.7, help="NMS阈值")

    # ReID parameters
    parser.add_argument("--reid_model", type=str, default="osnet_x0_25", help="ReID模型名称")
    parser.add_argument(
        "--reid_model_path",
        type=str,
        default="/home/sunrise/yolo11_strongsort/weights/osnet_64x128_nv12.bin",
        help="ReID模型路径",
    )
    parser.add_argument(
        "--reid_crop_size",
        type=int,
        nargs="+",
        default=[256, 128],
        help="ReID裁剪尺寸[h, w]",
    )

    # Kalman filter
    parser.add_argument("--kalman_format", type=str, default="strongsort", help="卡尔曼滤波器格式")

    # TensorRT
    parser.add_argument("--trt", action="store_true", help="使用TensorRT加速")

    # Device
    parser.add_argument("--device", type=str, default="0", help="cuda设备，如0或0,1,2,3或cpu")

    # Visualization
    parser.add_argument("--show", action="store_true", help="实时显示结果")
    parser.add_argument("--save_dir", type=str, default="track_demo_results")
    parser.add_argument("--save_video", action="store_true", help="保存输出视频")
    parser.add_argument("--save_txt", action="store_true", help="保存跟踪结果到txt文件")

    # 时间统计
    parser.add_argument("--time_stats", action="store_true", help="记录时间统计信息")
    parser.add_argument("--time_log", type=str, default="time_stats.txt", help="时间统计日志文件")

    return parser.parse_args()

class YOLOStrongSORT:
    def __init__(self, args):
        self.args = args

        self.device = "cpu"

        # Initialize YOLOv11 Pose detector
        logger.info(f"加载YOLOv11 Pose模型: {args.pose_model_path}")
        self.detector = YOLOPose(
            model_path=args.pose_model_path,
            conf_thres=args.conf_thresh,
            iou_thres=args.iou_thresh
        )

        # Initialize StrongSORT tracker
        logger.info("初始化StrongSORT跟踪器")
        if not hasattr(args, "trt"):
            args.trt = False  # 默认不使用TensorRT
        self.tracker = StrongSortTracker(args)

        # 预热模型
        self._warmup_models()

    def _warmup_models(self):
        """预热模型以获得更准确的时间统计"""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.process_frame(dummy_frame, 0)

    def process_frame(self, frame, frame_id):
        # 检测时间统计
        detection_start = time.time()

        # 执行检测
        detections, keypoints_list = self.detector.detect(frame)
        
        # 过滤小目标
        filtered_dets = []
        filtered_kpts = []
        for det, kpts in zip(detections, keypoints_list):
            if det[2] * det[3] > self.args.min_area:  # w * h > min_area
                filtered_dets.append(det)
                filtered_kpts.append(kpts)
        
        detections = np.array(filtered_dets) if len(filtered_dets) > 0 else np.empty((0, 6))
        detection_time = time.time() - detection_start

        # 跟踪时间统计
        tracking_start = time.time()
        
        # 执行跟踪
        current_tracks = self.tracker.update(
            detections, frame, frame, filtered_kpts
        )
        tracking_time = time.time() - tracking_start
        
                # 打印当前帧的时间统计
        logger.info(f"帧 {frame_id} - YOLO检测时间: {detection_time*1000:.1f}ms | StrongSORT跟踪时间: {tracking_time*1000:.1f}ms")


        # 准备结果
        tracked_bboxes, track_ids, class_ids, keypoints_list = [], [], [], []
        for trk in current_tracks:
            bbox = trk.tlwh
            track_id = trk.track_id
            cls_id = trk.category
            kps = trk.keypoints if hasattr(trk, "keypoints") else None

            if bbox[2] * bbox[3] > self.args.min_area and cls_id == 0:
                tracked_bboxes.append(bbox)
                track_ids.append(track_id)
                class_ids.append(cls_id)
                keypoints_list.append(kps)
                
        return (
            tracked_bboxes,
            track_ids,
            class_ids,
            keypoints_list,
            detection_time,
            tracking_time,
        )


def main(args):
    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # 初始化时间统计
    total_det_time = 0.0
    total_track_time = 0.0
    frame_count = 0

    # Initialize tracker
    tracker = YOLOStrongSORT(args)

    # 定义 save_obj_name（从输入文件名提取）
    save_obj_name = os.path.splitext(os.path.basename(args.source))[
        0
    ]  # 如 "track_person"
    os.makedirs(
        os.path.join(args.save_dir, save_obj_name, "vis_results"), exist_ok=True
    )  # 创建可视化目录

    # Check if source is video or image folder
    is_video = args.source.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))

    # Initialize video writer if needed
    video_writer = None
    if is_video and args.save_video:
        cap = cv2.VideoCapture(args.source)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 修复视频输出路径 - 使用正确的输出目录
        video_output_path = os.path.join(args.output, f"{save_obj_name}_output.mp4")
        video_writer = cv2.VideoWriter(
            video_output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        logger.info(f"视频输出路径: {video_output_path}")
        cap.release()

    # Process video
    if is_video:
        cap = cv2.VideoCapture(args.source)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for frame_idx in tqdm(range(total_frames), desc="处理视频中"):
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame with timing
            start_time = time.time()
            bboxes, track_ids, class_ids, keypoints, det_time, track_time = (
                tracker.process_frame(frame, frame_idx + 1)
            )
            total_time = time.time() - start_time

            # 更新时间统计
            total_det_time += det_time
            total_track_time += track_time
            frame_count += 1

            # Visualization
            vis_frame = plot_img(
                img=frame,
                frame_id=frame_idx + 1,
                results=[bboxes, track_ids, class_ids, keypoints],
                save_dir=os.path.join(args.save_dir, save_obj_name, "vis_results"),
            )

            # 在帧上显示时间信息
            if args.time_stats:
                fps = 1 / total_time if total_time > 0 else 0
                cv2.putText(
                    vis_frame,
                    f"Det: {det_time*1000:.1f}ms",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    vis_frame,
                    f"Track: {track_time*1000:.1f}ms",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    vis_frame,
                    f"FPS: {fps:.1f}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            if args.show:
                cv2.imshow("Tracking", vis_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # 保存视频帧 - 修复：确保写入视频文件
            if args.save_video and video_writer is not None:
                video_writer.write(vis_frame)

            # 保存可视化图片
            if args.save_dir:
                cv2.imwrite(
                    os.path.join(
                        args.save_dir,
                        save_obj_name,
                        "vis_results",
                        f"frame_{frame_idx+1:06d}.jpg",
                    ),
                    vis_frame,
                )

            # Save results to txt if needed
            if args.save_txt:
                with open(os.path.join(args.output, "track_results.txt"), "a") as f:
                    for bbox, track_id in zip(bboxes, track_ids):
                        f.write(
                            f"{frame_idx+1},{track_id},{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},1,-1,-1,-1\n"
                        )

        cap.release()
        if args.save_video and video_writer is not None:
            video_writer.release()
            logger.info(f"视频已保存至: {video_output_path}")

    # Process image folder
    else:
        image_files = sorted(
            [
                f
                for f in os.listdir(args.source)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        )

        for frame_idx, img_file in enumerate(tqdm(image_files, desc="处理图片中")):
            frame = cv2.imread(os.path.join(args.source, img_file))

            # Process frame with timing
            start_time = time.time()
            bboxes, track_ids, class_ids, keypoints, det_time, track_time = (
                tracker.process_frame(frame, frame_idx + 1)
            )
            total_time = time.time() - start_time

            # 更新时间统计
            total_det_time += det_time
            total_track_time += track_time
            frame_count += 1

            # Visualization
            vis_frame = plot_img(
                img=frame,
                frame_id=frame_idx + 1,
                results=[bboxes, track_ids, class_ids, keypoints],
                save_dir=os.path.join(args.save_dir, save_obj_name, "vis_results"),
            )

            # 在帧上显示时间信息
            if args.time_stats:
                fps = 1 / total_time if total_time > 0 else 0
                cv2.putText(
                    vis_frame,
                    f"Det: {det_time*1000:.1f}ms",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    vis_frame,
                    f"Track: {track_time*1000:.1f}ms",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    vis_frame,
                    f"FPS: {fps:.1f}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            if args.show:
                cv2.imshow("Tracking", vis_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # Save visualized image
            output_img_path = os.path.join(args.output, f"frame_{frame_idx+1:05d}.jpg")
            cv2.imwrite(output_img_path, vis_frame)

            # Save results to txt if needed
            if args.save_txt:
                with open(os.path.join(args.output, "track_results.txt"), "a") as f:
                    for bbox, track_id in zip(bboxes, track_ids):
                        f.write(
                            f"{frame_idx+1},{track_id},{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},1,-1,-1,-1\n"
                        )

    # 输出时间统计
    if frame_count > 0 and args.time_stats:
        avg_det_time = total_det_time / frame_count
        avg_track_time = total_track_time / frame_count
        avg_total_time = (total_det_time + total_track_time) / frame_count

        time_log_path = os.path.join(args.output, args.time_log)
        with open(time_log_path, "w") as f:
            f.write(f"总帧数: {frame_count}\n")
            f.write(f"平均检测时间: {avg_det_time*1000:.2f}ms (每帧)\n")
            f.write(f"平均跟踪时间: {avg_track_time*1000:.2f}ms (每帧)\n")
            f.write(f"平均总处理时间: {avg_total_time*1000:.2f}ms (每帧)\n")
            f.write(f"平均FPS: {1/avg_total_time:.2f}\n")

        logger.info("=" * 50)
        logger.info(f"时间统计已保存至: {time_log_path}")
        logger.info(f"平均检测时间: {avg_det_time*1000:.2f}ms/帧")
        logger.info(f"平均跟踪时间: {avg_track_time*1000:.2f}ms/帧")
        logger.info(f"平均总处理时间: {avg_total_time*1000:.2f}ms/帧")
        logger.info(f"平均FPS: {1/avg_total_time:.2f}")
        logger.info("=" * 50)

    if args.show:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    args = get_args()
    main(args)
