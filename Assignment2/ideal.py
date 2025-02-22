import cv2
import os

def stitch_images_opencv(input_folder, output_folder="outputImages"):
    """
    Stitches multiple images into a panorama using OpenCV's built-in stitcher.
    """
    # Read images from folder
    image_files = sorted([f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg'))])
    images = [cv2.imread(os.path.join(input_folder, f)) for f in image_files]

    if len(images) < 2:
        print("Error: Need at least two images for stitching.")
        return

    # Use OpenCV's built-in stitcher
    stitcher = cv2.Stitcher_create()
    status, panorama = stitcher.stitch(images)

    if status == cv2.Stitcher_OK:
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, "final_panorama.jpg")
        cv2.imwrite(output_path, panorama)
        print(f"Panorama saved at: {output_path}")
    else:
        print("Error: Stitching failed. Try different images or check alignment.")

if __name__ == "__main__":
    input_folder = "inputImages"
    stitch_images_opencv(input_folder)
