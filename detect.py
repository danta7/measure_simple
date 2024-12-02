
import cv2
import cv2

import cv2

import cv2

def manual_draw_rectangle(img, scale_factor=0.5):
    """
    手动绘制矩形ROI（感兴趣区域），支持窗口缩小显示，并实时显示矩形框。
    :param img: 输入图像（灰度图或二值图像）
    :param scale_factor: 缩放比例（默认缩小50%）
    :return: 矩形区域的左上角和右下角坐标 (top_left, bottom_right)
    """
    rect_params = None
    drawing = False  # 标记是否正在绘制

    # 缩放图像
    original_size = img.shape[1], img.shape[0]  # 原始尺寸 (宽, 高)
    resized_img = cv2.resize(img, (int(original_size[0] * scale_factor), int(original_size[1] * scale_factor)))

    def draw_rectangle(event, x, y, flags, param):
        """
        鼠标回调函数，用于绘制矩形ROI，并实时显示矩形框。
        """
        nonlocal rect_params, drawing, resized_img
        if event == cv2.EVENT_LBUTTONDOWN:
            # 鼠标按下，开始绘制
            drawing = True
            # 将鼠标点击位置从缩放图像坐标转换回原始图像坐标
            rect_params = [(x, y), (x, y)]  # 初始化矩形框的起始位置
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            # 鼠标移动过程中更新矩形的右下角
            rect_params[1] = (x, y)  # 更新矩形的右下角为当前鼠标位置
            # 在缩放后的图像上实时显示矩形框
            temp_img = resized_img.copy()  # 创建图像副本
            cv2.rectangle(temp_img, rect_params[0], rect_params[1], (255, 255, 255), 2)  # 绘制矩形框
            cv2.imshow("Draw ROI", temp_img)  # 显示带矩形框的图像
        elif event == cv2.EVENT_LBUTTONUP:
            # 鼠标释放，确认矩形ROI
            drawing = False
            # 打印缩放后的矩形框坐标
            print(f"Rectangle selected (scaled): Top-left={rect_params[0]}, Bottom-right={rect_params[1]}")

    # 显示缩放后的图像
    cv2.imshow("Draw ROI", resized_img)
    cv2.setMouseCallback("Draw ROI", draw_rectangle)

    while True:
        # 确保只有在绘制完成后输出矩形坐标
        if rect_params and not drawing:
            # 计算并输出原始图像上的矩形坐标
            original_top_left = (int(rect_params[0][0] / scale_factor), int(rect_params[0][1] / scale_factor))
            original_bottom_right = (int(rect_params[1][0] / scale_factor), int(rect_params[1][1] / scale_factor))
            print(f"Current selected rectangle (original): Top-left={original_top_left}, Bottom-right={original_bottom_right}")
            break  # 输出后退出循环

        # 等待按键，按下 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    # 返回原始图像上的坐标
    if rect_params:
        original_top_left = (int(rect_params[0][0] / scale_factor), int(rect_params[0][1] / scale_factor))
        original_bottom_right = (int(rect_params[1][0] / scale_factor), int(rect_params[1][1] / scale_factor))
        return original_top_left, original_bottom_right  # 返回的是原始图像的坐标
    else:
        return None

