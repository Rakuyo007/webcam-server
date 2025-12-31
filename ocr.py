from ultralytics import YOLO
import cv2
from paddleocr import PaddleOCR

from yolo import detect_objects

def perform_ocr_on_image(ocr_model ,image_path, yolo_results=None):
    """
    使用PaddleOCR对图像进行文字识别
    :param image_path: 输入图像路径 或 ndarray(frame)
    :param yolo_results: YOLO检测结果列表
    :return: 处理后的 yolo_results（带 OCR 结果）
    """
    # 1. 读取图像
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
    else:
        image = image_path  # ndarray

    img_h, img_w = image.shape[:2]

    final_results = []

    for item in yolo_results:
        result_item = item.copy()

        # 默认值（非 tag）
        result_item["ocr_text"] = ""
        result_item["ocr_score"] = 0

        if item["class"] == "tag":
            x, y, w, h = item["xywh"]

            # 2. xywh → xyxy（像素坐标）
            x1 = int(max(0, x - w / 2))
            y1 = int(max(0, y - h / 2))
            x2 = int(min(img_w, x + w / 2))
            y2 = int(min(img_h, y + h / 2))

            # 防止非法区域
            if x2 > x1 and y2 > y1:
                crop_img = image[y1:y2, x1:x2]

                # 3. OCR 识别
                ocr_result = ocr_model.predict(crop_img)

                # 4. 提取最高置信度文本
                best_text = ""
                best_conf = 0

                if ocr_result:
                    final_texts = []
                    final_scores = []

                    # 遍历每个 OCR 结果
                    for line in ocr_result:
                        rec_texts = line.get('rec_texts', [])
                        rec_scores = line.get('rec_scores', [])

                        # 筛选 score > 0.9 的文字
                        for text, score in zip(rec_texts, rec_scores):
                            if score > 0.9:
                                final_texts.append(text)
                                final_scores.append(score)

                    if final_texts:
                        best_text = ' '.join(final_texts)  # 拼接文字
                        best_conf = min(final_scores)  # 取最小置信度
                    else:
                        best_text = ''
                        best_conf = 0.0

                    result_item["ocr_text"] = best_text
                    result_item["ocr_score"] = round(best_conf, 3)

        final_results.append(result_item)

    return final_results


if __name__ == "__main__":
    # 测试 OCR 函数
    ocr = PaddleOCR(
        text_recognition_model_name="PP-OCRv5_server_rec",
        text_recognition_model_dir="./PP-OCRv5_server_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False
    )
    yolo_model = YOLO('xiangjiaba_best.pt')
    test_image_path = '/Users/well/Pictures/向家坝-端子识别/Xiangjiaba_connector_2.jpg'  # 替换为你的图像路径
    yolo_result = detect_objects(yolo_model, test_image_path)
    ocr_results = perform_ocr_on_image(ocr, test_image_path, yolo_result)
    for res in ocr_results:
        print(res)