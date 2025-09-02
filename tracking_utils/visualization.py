import cv2 
import os 
import numpy as np 
from PIL import Image

def plot_img1(img, frame_id, results, save_dir):
    """
    img: np.ndarray: (H, W, C)
    frame_id: int
    results: [tlwhs, ids, clses]
    save_dir: sr

    plot images with bboxes of a seq
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    assert img is not None

    if len(img.shape) > 3:
        img = img.squeeze(0)

    img_ = np.ascontiguousarray(np.copy(img))

    tlwhs, ids, clses = results[0], results[1], results[2]
    for tlwh, id, cls in zip(tlwhs, ids, clses):

        # convert tlwh to tlbr
        tlbr = tuple([int(tlwh[0]), int(tlwh[1]), int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])])
        # draw a rect
        cv2.rectangle(img_, tlbr[:2], tlbr[2:], get_color(id), thickness=3, )
        # note the id and cls
        text = f'{int(cls)}_{id}'
        cv2.putText(img_, text, (tlbr[0], tlbr[1]), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, 
                        color=(255, 164, 0), thickness=2)

    cv2.imwrite(filename=os.path.join(save_dir, f'{frame_id:05d}.jpg'), img=img_)
    return img_


### plot keypoint

def plot_img(img, frame_id, results, save_dir=None):
    """
    可视化跟踪结果，包含：
    - 边界框（红色=跟踪中，绿色=空闲）
    - 关键点（红色圆点）
    - 骨骼连接线（黄色）
    - ID和置信度标签
    - 跟踪状态标签
    """
    bboxes, track_ids, class_ids, keypoints = results
    # COCO关键点骨架连接关系（1-based索引）
    skeleton = [
        (16, 14), (14, 12), (17, 15), (15, 13), (12, 13),
        (6, 12), (7, 13), (6, 7), (6, 8), (7, 9),
        (8, 10), (9, 11), (2, 3), (1, 2), (1, 3),
        (2, 4), (3, 5), (4, 6), (5, 7)
    ]

    for bbox, track_id, cls_id, kps in zip(bboxes, track_ids, class_ids, keypoints):
        # 转换bbox格式 [x1,y1,w,h] -> [x1,y1,x2,y2]
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h
        bbox_int = [int(x1), int(y1), int(x2), int(y2)]
        # 根据跟踪状态设置颜色（示例逻辑，需根据实际跟踪状态调整）
        is_tracking = False  # 这里需要替换为实际的跟踪状态判断
        box_color = (0, 0, 255) if is_tracking else (0, 255, 0)
        text_color = (0, 0, 255) if is_tracking else (0, 255, 0)
        
        status_color = (0, 0, 255) if is_tracking else (255, 0, 0)

        # 绘制边界框
        cv2.rectangle(img, (bbox_int[0], bbox_int[1]), (bbox_int[2], bbox_int[3]), box_color, 2)

        # 绘制ID和置信度标签（示例中conf固定为0.9，实际应从结果获取）
        label = f"ID:{track_id}"  # 替换为实际的conf值
        
        cv2.putText(img, label, (bbox_int[0], bbox_int[1]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

        # 绘制关键点（如果存在）
        if kps is not None:
            # 绘制每个关键点
            for kp in kps:
                if len(kp) >= 2 and not np.isnan(kp[:2]).any():  # 至少需要x,y坐标
                    x, y = int(kp[0]), int(kp[1])
                    if x > 0 and y > 0:  # 过滤无效点
                        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

            # 绘制骨骼连接线
            for connection in skeleton:
                idx1, idx2 = connection[0]-1, connection[1]-1  # 转换为0-based索引
                if (idx1 < len(kps) and idx2 < len(kps) and 
                    not np.isnan(kps[idx1][:2]).any() and 
                    not np.isnan(kps[idx2][:2]).any()):
                    pt1 = (int(kps[idx1][0]), int(kps[idx1][1]))
                    pt2 = (int(kps[idx2][0]), int(kps[idx2][1]))
                    if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                        cv2.line(img, pt1, pt2, (0, 255, 255), 2)

    # 保存结果（如果指定了保存目录）
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, f"frame_{frame_id:06d}.jpg"), img)
    
    return img


def get_color(idx):
    """
    aux func for plot_seq
    get a unique color for each id
    """
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color

def save_video(images_path):
    """
    save images (frames) to a video
    """

    images_list = sorted(os.listdir(images_path))
    save_video_path = os.path.join(images_path, images_path.split('/')[-1] + '.mp4')

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    img0 = Image.open(os.path.join(images_path, images_list[0]))
    vw = cv2.VideoWriter(save_video_path, fourcc, 15, img0.size)

    for image_name in images_list:
        image = cv2.imread(filename=os.path.join(images_path, image_name))
        vw.write(image)
