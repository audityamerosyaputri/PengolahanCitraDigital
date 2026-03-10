import cv2
import numpy as np
import time

class RealTimeEnhancement:
    def __init__(self, target_fps=30):
        """
        Initialize real-time enhancement system
        """
        self.target_fps = target_fps
        self.history_buffer = []
        self.max_history = 5   # limit buffer untuk menjaga memori

    def enhance_frame(self, frame, enhancement_type='adaptive'):
        """
        Enhance single frame with real-time constraints

        Parameters:
        frame : Input video frame
        enhancement_type : Type of enhancement

        Returns:
        Enhanced frame
        """

        # Convert ke grayscale untuk processing lebih cepat
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ===============================
        # ENHANCEMENT METHODS
        # ===============================

        if enhancement_type == 'adaptive':

            # CLAHE untuk real-time contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)

        elif enhancement_type == 'sharpen':

            kernel = np.array([[0,-1,0],
                               [-1,5,-1],
                               [0,-1,0]])

            enhanced = cv2.filter2D(gray, -1, kernel)

        elif enhancement_type == 'denoise':

            enhanced = cv2.GaussianBlur(gray,(5,5),0)

        else:
            enhanced = gray

        # ===============================
        # TEMPORAL CONSISTENCY
        # ===============================

        self.history_buffer.append(enhanced)

        if len(self.history_buffer) > self.max_history:
            self.history_buffer.pop(0)

        # rata-rata frame sebelumnya untuk stabilitas video
        if len(self.history_buffer) > 1:
            avg_frame = np.mean(self.history_buffer, axis=0).astype(np.uint8)
        else:
            avg_frame = enhanced

        # convert kembali ke BGR agar bisa ditampilkan bersama original
        avg_frame = cv2.cvtColor(avg_frame, cv2.COLOR_GRAY2BGR)

        return avg_frame


# =========================
# MAIN PROGRAM
# =========================

def main():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Kamera tidak dapat dibuka")
        return

    enhancer = RealTimeEnhancement(target_fps=30)

    print("Tekan 'q' untuk keluar")

    while True:

        start_time = time.time()

        ret, frame = cap.read()

        if not ret:
            print("Error membaca frame")
            break

        # Resize agar processing lebih cepat
        frame = cv2.resize(frame, (640,480))

        # Enhancement
        enhanced_frame = enhancer.enhance_frame(frame, 'adaptive')

        # Gabungkan original dan enhanced
        combined = np.hstack((frame, enhanced_frame))

        cv2.putText(combined,
                    "Original (Left) | Enhanced (Right)",
                    (20,30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0),
                    2)

        cv2.imshow("Real-Time Video Enhancement", combined)

        # FPS Control
        elapsed = time.time() - start_time
        wait_time = max(1, int((1/enhancer.target_fps - elapsed)*1000))

        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# =========================
# RUN PROGRAM
# =========================

if __name__ == "__main__":
    main()