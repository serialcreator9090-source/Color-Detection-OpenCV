
import cv2
import numpy as np


class ColorDetector:
    def __init__(self, minArea=300):
        self.minArea = minArea

        # HSV color ranges from the original project
        self.colorRanges = {
            "Red": (
                np.array([136, 87, 111], dtype=np.uint8),
                np.array([180, 255, 255], dtype=np.uint8),
                (0, 0, 255)
            ),
            "Green": (
                np.array([25, 52, 72], dtype=np.uint8),
                np.array([102, 255, 255], dtype=np.uint8),
                (0, 255, 0)
            ),
            "Blue": (
                np.array([94, 80, 2], dtype=np.uint8),
                np.array([120, 255, 255], dtype=np.uint8),
                (255, 0, 0)
            )
        }

        self.kernel = np.ones((5, 5), np.uint8)

    def findColors(self, img, draw=True):
        hsvFrame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        detections = []

        for colorName, (lower, upper, boxColor) in self.colorRanges.items():
            mask = cv2.inRange(hsvFrame, lower, upper)
            mask = cv2.dilate(mask, self.kernel)

            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                area = cv2.contourArea(contour)

                if area > self.minArea:
                    x, y, w, h = cv2.boundingRect(contour)

                    detections.append({
                        "color": colorName,
                        "bbox": (x, y, w, h),
                        "area": area
                    })

                    if draw:
                        cv2.rectangle(
                            img,
                            (x, y),
                            (x + w, y + h),
                            boxColor,
                            2
                        )

                        cv2.putText(
                            img,
                            f"{colorName} Color",
                            (x, max(y - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            boxColor,
                            2
                        )

        return img, detections


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    detector = ColorDetector()

    try:
        while True:
            success, img = cap.read()

            if not success:
                print("Error: Could not read webcam frame.")
                break

            img, detections = detector.findColors(img)

            cv2.imshow("Real-Time Color Detection", img)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
