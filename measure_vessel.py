from detect import *
import cv2
import numpy as np


def calculate_diameter(segmented_img, top_left, bottom_right):
    """
    使用选择的矩形区域，在每条血管上选择 5 个等距离的测量点，计算血管直径。
    :param segmented_img: 已分割的血管二值图像
    :param top_left: 矩形左上角坐标
    :param bottom_right: 矩形右下角坐标
    :return: 平均直径及标注测量点后的图像
    """
    # 提取感兴趣区域 (ROI)
    mask = np.zeros_like(segmented_img, dtype=np.uint8)
    cv2.rectangle(mask, top_left, bottom_right, 255, -1)
    roi = cv2.bitwise_and(segmented_img, segmented_img, mask=mask)

    # 提取血管骨架
    skeleton = cv2.ximgproc.thinning(roi)

    # 查找所有血管分支
    contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # 用于绘制的图像
    result_img = cv2.cvtColor(segmented_img, cv2.COLOR_GRAY2BGR)
    all_diameters = []

    # 遍历每条血管分支
    for contour in contours:
        if len(contour) < 5:  # 忽略较短的血管分支
            continue

        # 在每条血管上选择 5 个等距离点
        contour = contour.reshape(-1, 2)
        step = len(contour) // 5  # 确定等距点之间的步长
        points = contour[::step][:5]  # 选择 5 个点

        # 在每个点的垂直方向计算直径
        diameters = []
        for point in points:
            x, y = point
            # 在垂直方向搜索边界
            left = right = x
            while left > 0 and segmented_img[y, left] > 0:
                left -= 1
            while right < segmented_img.shape[1] and segmented_img[y, right] > 0:
                right += 1
            diameter = right - left
            diameters.append(diameter)

            # 在图像上标记测量点
            cv2.circle(result_img, (x, y), 6, (0, 0, 255), -1)  # 红点为测量点
            cv2.line(result_img, (left, y), (right, y), (255, 0, 0), 2)  # 蓝线为直径线

        # 计算血管分支的平均直径
        if diameters:
            branch_avg_diameter = np.mean(diameters)
            all_diameters.append(branch_avg_diameter)

    # 返回结果
    overall_avg_diameter = np.mean(all_diameters) if all_diameters else 0
    return overall_avg_diameter, result_img


if __name__ == "__main__":
    # 加载已分割的二值血管图像
    segmented_img = cv2.imread("24.png", cv2.IMREAD_GRAYSCALE)

    # 用户选择感兴趣区域
    print("Draw a rectangle to define the region of interest (ROI).")
    top_left, bottom_right = manual_draw_rectangle(segmented_img)

    # 打印选择的区域
    print(f"Selected Rectangle: Top-left={top_left}, Bottom-right={bottom_right}")

    # 在选择的区域内进行血管直径测量
    avg_diameter, result_img = calculate_diameter(segmented_img, top_left, bottom_right)

    # 显示测量结果
    print(f"Overall average vessel diameter: {avg_diameter:.2f} pixels")
    cv2.imshow("Measured Vessel Diameters", cv2.resize(result_img, (result_img.shape[1] // 2, result_img.shape[0] // 2)))

    # 等待用户按键
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):  # 如果按下 'q' 键，退出程序
        cv2.destroyAllWindows()
    elif key == ord('s'):  # 如果按下 's' 键，保存结果并退出
        cv2.imwrite("vessel_diameter_result.png", result_img)
        print("Result saved as vessel_diameter_result.png")
        cv2.destroyAllWindows()


