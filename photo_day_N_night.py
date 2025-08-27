import cv2
import numpy as np
import os

def day_night_convert(image_path):
    """
    Convert an input image from day→night or night→day effect using HSV manipulation.
    Saves the output in the same directory with a new name.
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Error: Could not load image. Check the path.")
        return
    
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Detect average brightness
    avg_brightness = np.mean(v)
    
    if avg_brightness > 100:
        # Day → Night
        v = cv2.convertScaleAbs(v, alpha=0.3, beta=0)
        s = cv2.convertScaleAbs(s, alpha=0.4, beta=0)
        effect = "night"
    else:
        # Night → Day
        v = cv2.convertScaleAbs(v, alpha=1.8, beta=30)
        s = cv2.convertScaleAbs(s, alpha=1.7, beta=0)
        effect = "day"
    
    # Merge channels back
    hsv_modified = cv2.merge([h, s, v])
    output = cv2.cvtColor(hsv_modified, cv2.COLOR_HSV2BGR)
    
    # Show result
    cv2.imshow(f"Converted to {effect}", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save in same directory with new name
    base, ext = os.path.splitext(image_path)
    new_filename = f"{base}_{effect}{ext}"
    cv2.imwrite(new_filename, output)
    
    print(f"✅ Converted image saved as: {new_filename}")


if __name__ == "__main__":
    img_path = "A:\computer science\mlprojects\ML_practice\Screenshot (67)_night.png"
    day_night_convert(img_path)
