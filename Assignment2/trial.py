import cv2
import os
import numpy as np

def compute_homography(img1, img2):
    """
    Computes the homography matrix between two images using feature matching.
    """
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Use SIFT for feature detection
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        print("Error: Could not compute descriptors.")
        return None

    # Feature matching
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good_matches) < 10:
        print("Not enough good matches found.")
        return None

    # Get matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Compute homography
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
    return H

def warp_images(images, homographies, middle_index):
    """
    Warps all images into the same coordinate space using the middle image as reference.
    """
    h, w = images[middle_index].shape[:2]
    all_corners = []

    for i, H in enumerate(homographies):
        corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners, H)
        all_corners.append(transformed_corners)

    all_corners = np.concatenate(all_corners, axis=0)
    x_min, y_min = np.int32(all_corners.min(axis=0).flatten())
    x_max, y_max = np.int32(all_corners.max(axis=0).flatten())

    # Compute translation matrix
    translation_matrix = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

    # Warp all images into a common space
    panorama_size = (x_max - x_min, y_max - y_min)
    panorama = np.zeros((y_max - y_min, x_max - x_min, 3), dtype=np.uint8)

    for i, (img, H) in enumerate(zip(images, homographies)):
        H_translated = translation_matrix @ H
        warped_img = cv2.warpPerspective(img, H_translated, panorama_size)
        panorama = np.maximum(panorama, warped_img)  # Overlay images

    return panorama

def main():
    input_folder = "inputImages/Set2"

    if not os.path.exists(input_folder):
        print(f"Folder {input_folder} does not exist.")
        return

    # Read images from the folder
    image_files = sorted([f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg'))])
    images = [cv2.imread(os.path.join(input_folder, f)) for f in image_files]

    if len(images) < 2:
        print("Error: Need at least two images for stitching.")
        return

    # Select the middle image as reference
    middle_index = len(images) // 2
    homographies = [np.eye(3)]  # Identity matrix for the reference image

    # Compute homographies for left and right images relative to the middle
    for i in range(middle_index - 1, -1, -1):  # Left-side images
        H = compute_homography(images[i], images[i + 1])
        if H is None:
            print(f"Skipping image {i} due to homography failure.")
            continue
        homographies.insert(0, homographies[0] @ H)  # Accumulate homographies

    for i in range(middle_index + 1, len(images)):  # Right-side images
        H = compute_homography(images[i], images[i - 1])
        if H is None:
            print(f"Skipping image {i} due to homography failure.")
            continue
        homographies.append(homographies[-1] @ np.linalg.inv(H))  # Accumulate in reverse

    # Warp images into common space
    final_panorama = warp_images(images, homographies, middle_index)

    # Save the final panorama
    output_path = "outputImages"
    os.makedirs(output_path, exist_ok=True)
    cv2.imwrite(output_path + "/finalPanorama.jpg", final_panorama)
    print(f"Panorama saved at '{output_path}'.")

if __name__ == "__main__":
    main()
