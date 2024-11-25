import cv2


def detct_optic_disc_center(segmented_img):
    """
    自动检测视盘中心，基于血管密度
    :param segmented_img:已分割的血管图像
    :return:自动检测的实盘中心坐标(x,y)
    """
    # 扩大血管区域
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    dilated = cv2.dilate(segmented_img, kernel, iterations=2)

    # 计算局部密度
    blurred = cv2.GaussianBlur(dilated, (15, 15), 0)
    _, _, _, max_loc = cv2.minMaxLoc(blurred)
    return max_loc  # 返回视盘中心坐标

def manual_adjustment(img,auto_center):
    """
    手动调整视盘中心
    :param img: 输入图像
    :param auto_center: 自动检测的视盘中心
    :return:最终调整的视盘中心坐标
    """
    global optic_disc_center
    optic_disc_center = auto_center

    def click_event(event, x, y, flags, param):
        global optic_disc_center
        if event == cv2.EVENT_LBUTTONDOWN:
            optic_disc_center = (x,y)
            print(f"Manual adjustment: ({x},{y})")
    # 显示图像
    temp_img = img.copy()
    cv2.circle(temp_img,auto_center,10,(255,0,0),-1)
    cv2.imshow("Adjust Optic Disc Center", temp_img)
    cv2.setMouseCallback("Adjust Optic Disc Center", click_event)

    while True:
        cv2.imshow("Adjust Optic Disc Center", temp_img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


    cv2.destroyAllWindows()
    return optic_disc_center


if __name__ == '__main__':
    semented_img = cv2.imread("24.png")

    auto_center = detct_optic_disc_center(semented_img)
    print("auto_center is", auto_center)

    # 手动调整
    optic_disc_center = manual_adjustment(semented_img, auto_center)
    print("Final optic disc center", optic_disc_center)

    # 显示结构
    result = cv2.cvtColor(semented_img, cv2.COLOR_BGR2GRAY)
    cv2.circle(result,optic_disc_center,20,(0,255,0),2)
    cv2.imshow("Detected Optic Disc Center", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()