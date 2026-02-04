import cv2
import time
import os

def main():
    print("Attempting to open camera...")
    cap = cv2.VideoCapture(0)  # 0 represents the primary camera
    
    if not cap.isOpened():
        print("Error: Could not open camera. Please check if camera is connected and permissions are granted.")
        return

    print("Camera opened successfully. Capturing frames...")
    
    # Warm up camera
    for i in range(10):
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to grab frame {i}")
            break
        time.sleep(0.1)
    
    # Capture a test frame
    ret, frame = cap.read()
    if ret:
        output_path = os.path.join(os.path.dirname(__file__), "camera_test_result.png")
        cv2.imwrite(output_path, frame)
        print(f"Successfully captured image and saved to {output_path}")
    else:
        print("Failed to capture final frame.")

    cap.release()
    cv2.destroyAllWindows()
    print("Camera test completed.")

if __name__ == '__main__':
    main()
