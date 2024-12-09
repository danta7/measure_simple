import time
from detect import *
import numpy as np
import cv2
from scipy.optimize import curve_fit, OptimizeWarning
import json
import os
from datetime import datetime
import warnings

# 过滤掉 curve_fit 的协方差警告
warnings.filterwarnings('ignore', category=OptimizeWarning)


class VesselMeasurement:
    def __init__(self):
        self.reference_points = []
        self.diameters = []
        self.img = None
        self.roi = None
        self.top_left = None
        self.bottom_right = None

    def load_image(self, image_path):
        """Load and store the binary vessel image"""
        self.img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.img is None:
            raise ValueError("Failed to load image")
        return self.img

    def select_vessel(self):
        """Select a vessel using rectangular ROI"""
        self.top_left, self.bottom_right = manual_draw_rectangle(self.img, scale_factor=0.8)
        if self.top_left and self.bottom_right:
            x1, y1 = self.top_left
            x2, y2 = self.bottom_right
            self.roi = self.img[y1:y2, x1:x2].copy()
            return True
        return False

    def select_reference_points(self):
        """Allow user to select reference points along the vessel"""

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Convert coordinates to original image space
                orig_x = x + self.top_left[0]
                orig_y = y + self.top_left[1]
                self.reference_points.append((orig_x, orig_y))
                # Draw the point
                cv2.circle(display_img, (x, y), 2, (255, 0, 0), -1)
                cv2.imshow("Select Reference Points", display_img)

        # Create a copy of ROI for display
        display_img = cv2.cvtColor(self.roi, cv2.COLOR_GRAY2BGR)
        cv2.imshow("Select Reference Points", display_img)
        cv2.setMouseCallback("Select Reference Points", mouse_callback)

        print("Select reference points along the vessel. Press 'q' when done.")
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cv2.destroyAllWindows()
        return len(self.reference_points) > 0

    def auto_select_reference_points(self, num_points=10):
        """自动在血管上选择参考点"""
        # 获取ROI的骨架
        from skimage.morphology import skeletonize
        binary_roi = (self.roi > 128).astype(np.uint8)
        skeleton = skeletonize(binary_roi)

        # 转换为OpenCV格式
        skeleton = (skeleton * 255).astype(np.uint8)

        # 查找所有骨架点
        points = np.where(skeleton > 0)
        if len(points[0]) < num_points:
            print("找到的骨架点太少，无法选择足够的参考点")
            return False

        # 均匀选择指定数量的点
        total_points = len(points[0])
        indices = np.linspace(0, total_points - 1, num_points).astype(int)

        # 清除之前的参考点
        self.reference_points = []

        # 添加新的参考点（转换回原始图像坐标）
        for idx in indices:
            y, x = points[0][idx], points[1][idx]
            # 转换回原始图像坐标
            orig_x = x + self.top_left[0]
            orig_y = y + self.top_left[1]
            self.reference_points.append((orig_x, orig_y))

        # 显示选择的参考点
        display_img = cv2.cvtColor(self.roi, cv2.COLOR_GRAY2BGR)
        for x, y in [(p[0] - self.top_left[0], p[1] - self.top_left[1]) for p in self.reference_points]:
            cv2.circle(display_img, (int(x), int(y)), 2, (0, 0, 255), -1)

        cv2.imshow("Selected Reference Points", display_img)
        cv2.waitKey(1000)  # 显示1秒
        cv2.destroyAllWindows()

        return True

    def measure_diameter_at_point(self, point, window_size=12):
        """Measure vessel diameter at a specific point with sub-pixel accuracy"""
        x, y = point

        # Extract local region around the point
        half_window = window_size // 2
        local_region = self.img[max(0, y - half_window):min(self.img.shape[0], y + half_window),
                       max(0, x - half_window):min(self.img.shape[1], x + half_window)]

        # 增加对比度以突出边缘
        local_region = cv2.normalize(local_region, None, 0, 255, cv2.NORM_MINMAX)

        # Calculate local vessel direction using structure tensor with optimal window
        gy, gx = np.gradient(local_region.astype(float))
        Ixx = gx * gx
        Ixy = gx * gy
        Iyy = gy * gy

        window = 5
        Ixx_avg = cv2.boxFilter(Ixx, -1, (window, window))
        Ixy_avg = cv2.boxFilter(Ixy, -1, (window, window))
        Iyy_avg = cv2.boxFilter(Iyy, -1, (window, window))

        # Get vessel direction (perpendicular to gradient)
        center_y, center_x = local_region.shape[0] // 2, local_region.shape[1] // 2
        A = np.array([[Ixx_avg[center_y, center_x], Ixy_avg[center_y, center_x]],
                      [Ixy_avg[center_y, center_x], Iyy_avg[center_y, center_x]]])
        eigenvals, eigenvecs = np.linalg.eigh(A)
        vessel_direction = eigenvecs[:, np.argmin(eigenvals)]
        normal_direction = np.array([-vessel_direction[1], vessel_direction[0]])

        # 优化采样策略
        num_samples = 150
        positions = np.linspace(-window_size / 2, window_size / 2, num_samples)
        samples = []
        intensities_list = []

        # 减少平行线的偏移量以降低测量偏差
        offsets = [-0.5, 0, 0.5]
        for offset in offsets:
            offset_direction = vessel_direction * offset
            for pos in positions:
                sample_x = x + normal_direction[0] * pos + offset_direction[0]
                sample_y = y + normal_direction[1] * pos + offset_direction[1]

                if 0 <= sample_x < self.img.shape[1] - 1 and 0 <= sample_y < self.img.shape[0] - 1:
                    x0, y0 = int(sample_x), int(sample_y)
                    if x0 > 0 and y0 > 0 and x0 < self.img.shape[1] - 2 and y0 < self.img.shape[0] - 2:
                        value = cv2.getRectSubPix(self.img, (1, 1), (sample_x, sample_y))[0][0]
                        samples.append((pos, value))
                        intensities_list.append(value)

        if not samples:
            return None

        # 更严格的异常值过滤
        intensities_array = np.array(intensities_list)
        q1, q3 = np.percentile(intensities_array, [25, 75])
        iqr = q3 - q1
        valid_samples = [(pos, val) for pos, val in samples
                         if (q1 - 1.2 * iqr) <= val <= (q3 + 1.2 * iqr)]

        if len(valid_samples) < num_samples * 0.6:
            return None

        positions, intensities = zip(*valid_samples)
        positions = np.array(positions)
        intensities = np.array(intensities)

        def sigmoid(x, x0, k, a, b):
            z = -k * (x - x0)
            z = np.clip(z, -500, 500)
            return a / (1.0 + np.exp(z)) + b

        try:
            # 归一化并增强对比度
            intensities_norm = (intensities - np.min(intensities)) / (np.max(intensities) - np.min(intensities))
            intensities_norm = np.power(intensities_norm, 1.2)

            # 使用较小的窗口进行平滑
            window_length = 3
            intensities_smooth = np.array([np.median(
                intensities_norm[max(0, i - window_length):min(len(intensities_norm), i + window_length + 1)])
                for i in range(len(intensities_norm))])

            # 拟合左边缘
            left_idx = len(positions) // 2
            left_samples = [(p, i) for p, i in zip(positions[:left_idx], intensities_smooth[:left_idx])]
            left_pos, left_int = zip(*left_samples)

            # 调整边界条件使拟合更敏感
            p0_left = [-window_size / 4, 2.0, 0.9, 0.1]
            bounds_left = (
                [-window_size / 2, 0.5, 0.6, 0.0],
                [0, 15.0, 1.0, 0.2]
            )
            popt_left, _ = curve_fit(sigmoid, left_pos, left_int,
                                     p0=p0_left, bounds=bounds_left,
                                     maxfev=5000)
            left_edge = popt_left[0]

            # 拟合右边缘
            right_samples = [(p, i) for p, i in zip(positions[left_idx:], intensities_smooth[left_idx:])]
            right_pos, right_int = zip(*right_samples)

            p0_right = [window_size / 4, 2.0, 0.9, 0.1]
            bounds_right = (
                [0, 0.5, 0.6, 0.0],
                [window_size / 2, 15.0, 1.0, 0.2]
            )
            popt_right, _ = curve_fit(sigmoid, right_pos, right_int,
                                      p0=p0_right, bounds=bounds_right,
                                      maxfev=5000)
            right_edge = popt_right[0]

            # 计算直径并进行更严格的合理性检查
            diameter = abs(right_edge - left_edge)
            if diameter < 1 or diameter > window_size * 0.8:
                print(f"Warning: Suspicious diameter measurement at point {point}: {diameter}")
                return None

            # 对直径进行微调校正
            diameter = diameter * 0.95

            return diameter

        except Exception as e:
            print(f"拟合失败: {str(e)}")
            return None

    def measure_all_points(self):
        """Measure diameters at all reference points"""
        self.diameters = []
        for point in self.reference_points:
            diameter = self.measure_diameter_at_point(point)
            if diameter is not None:
                self.diameters.append(diameter)
            else:
                print(f"Failed to measure diameter at point {point}")

    def save_selection_result(self, output_path="measurement_results"):
        """保存血管选择和参考点的结果图像"""
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # 创建彩色显示图像
        display_img = cv2.cvtColor(self.img.copy(), cv2.COLOR_GRAY2BGR)

        # 绘制选择的矩形区域
        cv2.rectangle(display_img,
                      self.top_left,
                      self.bottom_right,
                      (0, 255, 0), 2)

        # 绘制参考点
        for point in self.reference_points:
            cv2.circle(display_img,
                       (int(point[0]), int(point[1])),
                       3, (0, 0, 255), -1)

            # 保存图像
        image_filename = os.path.join(output_path, f"vessel_selection_{int(time.time())}.png")
        cv2.imwrite(image_filename, display_img)

        # 显示结果
        cv2.imshow("Selected Vessel and Reference Points", display_img)
        print("显示选择结果，按任意键继续...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return image_filename

    def save_results(self, output_path="measurement_results"):
        """Save measurement results to a JSON file"""
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        results = {
            "timestamp": datetime.now().isoformat(),
            "reference_points": [(int(x), int(y)) for x, y in self.reference_points],
            "diameters": [float(d) for d in self.diameters],
            "average_diameter": float(np.mean(self.diameters)) if self.diameters else None,
            "std_diameter": float(np.std(self.diameters)) if self.diameters else None
        }

        filename = os.path.join(output_path, f"vessel_measurements_{int(time.time())}.json")
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4)
        return filename


def main():
    # 创建测量实例
    vm = VesselMeasurement()

    # 加载图像
    image_path = input("请输入血管图像路径: ")
    vm.load_image(image_path)

    # 选择血管
    print("请通过绘制矩形来选择血管。点击并拖动，然后释放。")
    if not vm.select_vessel():
        print("血管选择失败")
        return

    # 自动选择参考点
    print("正在自动选择参考点...")
    if not vm.auto_select_reference_points(num_points=10):
        print("自动选择参考点失败")
        return

    # 保存并显示选择结果
    selection_image = vm.save_selection_result()
    print(f"选择结果图像已保存到: {selection_image}")

    # 测量直径
    print("正在测量血管直径...")
    vm.measure_all_points()

    # 保存结果
    if vm.diameters:
        result_file = vm.save_results()
        print(f"测量结果已保存到: {result_file}")
        print(f"平均直径: {np.mean(vm.diameters):.2f} 像素")
        print(f"标准差: {np.std(vm.diameters):.2f} 像素")
    else:
        print("没有有效的测量结果可保存")


if __name__ == '__main__':
    main()