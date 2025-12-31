from ultralytics import YOLO
import cv2
# 加载YOLO模型（根据需要更改路径）

def detect_objects(yolo_model, image_path):
    """
    使用YOLO模型检测图像中的对象
    :param image_path: 输入图像路径或 ndarray(frame)
    :return: list[dict]
    """
    det_list = []
    results = yolo_model.predict(image_path, save=False)
    for r in results:
        names = r.names
        boxes = r.boxes
        if boxes is None:
            continue
        cls_ids = boxes.cls.cpu().numpy()
        xywh = boxes.xywh.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        for i in range(len(cls_ids)):
            det_list.append({
                "class": names[int(cls_ids[i])],
                "xywh": [round(float(v), 3) for v in xywh[i]],
                "confidence": round(float(conf[i]), 3)
            })
    return det_list


if __name__ == "__main__":
    # 测试检测函数
    yolo_model = YOLO('xiangjiaba_best.pt')
    test_image_path = '/Users/well/Pictures/向家坝-端子识别/Xiangjiaba_connector_2.jpg'  # 替换为你的图像路径
    img = cv2.imread(test_image_path)
    yolo_result = detect_objects(yolo_model=yolo_model, image_path=test_image_path)
    for r in yolo_result:
        print(r)
        x, y, w, h = r['xywh']
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)
        # 根据类别选择颜色
        color = (0, 255, 0) if r['class'] == 'tag' else (255, 0, 0)
        # 画矩形框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        # 显示类别和置信度
        label = f"{r['class']}:{r['confidence']:.2f}"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    # 展示图片
    cv2.imshow("YOLO Results", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()