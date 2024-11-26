import cv2
import numpy as np


def manual_draw_circle(img, scale_factor=0.5):
    """
    手动绘制圆形ROI（感兴趣区域），支持窗口缩小显示。
    :param img: 输入图像（灰度图或二值图像）
    :param scale_factor: 缩放比例（默认缩小50%）
    :return: 圆心坐标和半径 (center, radius)
    """
    global circle_params, drawing
    circle_params = None
    drawing = False  # 标记是否正在绘制

    # 缩放图像
    original_size = img.shape[1], img.shape[0]  # 原始尺寸 (宽, 高)
    resized_img = cv2.resize(img, (int(original_size[0] * scale_factor), int(original_size[1] * scale_factor)))

    def draw_circle(event, x, y, flags, param):
        """
        鼠标回调函数，用于绘制圆形ROI。
        """
        global circle_params, drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            # 鼠标按下，开始绘制
            drawing = True
            circle_params = [(int(x / scale_factor), int(y / scale_factor)), 0]  # 初始化圆心和半径
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            # 鼠标移动过程中动态更新半径
            cx, cy = circle_params[0]
            radius = int(np.sqrt(((x / scale_factor) - cx) ** 2 + ((y / scale_factor) - cy) ** 2))
            circle_params[1] = radius
            temp_img = resized_img.copy()
            cv2.circle(temp_img, (int(cx * scale_factor), int(cy * scale_factor)), int(radius * scale_factor), (255, 0, 0), 2)
            cv2.imshow("Draw ROI", temp_img)
        elif event == cv2.EVENT_LBUTTONUP:
            # 鼠标释放，确认圆形ROI
            drawing = False
            print(f"Circle selected: Center={circle_params[0]}, Radius={circle_params[1]}")

    # 显示缩放后的图像
    temp_img = resized_img.copy()
    cv2.imshow("Draw ROI", temp_img)
    cv2.setMouseCallback("Draw ROI", draw_circle)

    while True:
        if circle_params and not drawing:
            # 绘制完成后在缩小图像上显示圆形ROI
            cx, cy = circle_params[0]
            radius = circle_params[1]
            temp_img = resized_img.copy()
            cv2.circle(temp_img, (int(cx * scale_factor), int(cy * scale_factor)), int(radius * scale_factor), (255, 0, 0), 2)
            cv2.imshow("Draw ROI", temp_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # 按下 'q' 确认并退出
            break

    cv2.destroyAllWindows()
    return circle_params


def display_result_with_scaling(img, center, radius, scale_factor=0.5):
    """
    显示结果窗口并缩小显示。
    :param img: 输入图像（灰度图或二值图像）
    :param center: 圆心坐标
    :param radius: 半径
    :param scale_factor: 缩放比例（默认缩小50%）
    """
    # 绘制圆形ROI
    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.circle(result, center, radius, (0, 255, 0), 2)

    # 缩小显示
    original_size = result.shape[1], result.shape[0]  # 原始尺寸 (宽, 高)
    resized_result = cv2.resize(result, (int(original_size[0] * scale_factor), int(original_size[1] * scale_factor)))

    # 显示结果
    cv2.imshow("Selected ROI (Resized)", resized_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
