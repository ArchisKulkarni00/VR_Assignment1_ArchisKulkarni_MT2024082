import cv2
import os
import numpy as np

def compute_panorama(img1, img2):
    """
    Computes the panorama by stitching img1 and img2 using feature matching and homography.
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
        return img1

    # Feature matching
    bf = cv2.DescriptorMatcher_create("BruteForce")
    matches = bf.knnMatch(des1, des2, 2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good_matches) < 10:
        print("Not enough good matches found.")
        return img1

    # Get matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Compute homography
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
    if M is None:
        print("Error: Homography matrix not found.")
        return img1

    # Warp img2 to align with img1
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Define corners of both images
    corners_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    corners_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    
    # Transform img2 corners to panorama space
    transformed_corners = cv2.perspectiveTransform(corners_img2, M)
    
    # Compute final bounding box
    all_corners = np.concatenate((corners_img1, transformed_corners), axis=0)
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # Compute translation matrix
    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])

    # Warp img1 into the panorama space
    panorama_width = xmax - xmin
    panorama_height = ymax - ymin
    maska = np.ones((img1.shape[0],img1.shape[1]))*255
    maskb = np.ones((img1.shape[0],img1.shape[1]))*255
    warped_img1 = cv2.warpPerspective(img2, Ht @ np.eye(3), (panorama_width, panorama_height))
    warped_img2 = cv2.warpPerspective(img1, Ht @ M, (panorama_width, panorama_height))
    mask1 = cv2.warpPerspective(maska, Ht @ np.eye(3), (panorama_width, panorama_height))
    mask2 = cv2.warpPerspective(maskb, Ht @ M, (panorama_width, panorama_height))
    mask = mask1+mask2

    cv2.imwrite(f"outputImages/wimg1.jpg", warped_img1)
    cv2.imwrite(f"outputImages/wimg2.jpg", warped_img2)
    cv2.imwrite(f"outputImages/mask.jpg", mask)

    print(xmax,"   ",xmin)
    print(ymax,"   ",ymin)

    mask = mask.astype(np.uint8)
    result = cv2.bitwise_or(warped_img1, warped_img2, mask=mask)
    # result = warped_img2+warped_img1

    return result

def main():
    input_folder = "inputImages"

    if not os.path.exists(input_folder):
        print(f"Folder {input_folder} does not exist.")
        return

    # Read images from the folder
    image_files = sorted([f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg'))])
    print(image_files)
    images = [cv2.imread(os.path.join(input_folder, f)) for f in image_files]

    if len(images) < 2:
        print("No valid images found in the folder.")
        return

    # Initialize the first image as the stitched result
    # img_stitched = images[0]
    img_stitched = compute_panorama(images[0], images[1])
    cv2.imwrite(f"outputImages/trial.jpg", img_stitched)

    # Iterate over all images and compute panoramas
    # for i in range(1, len(images)):
    #     img_stitched = compute_panorama(img_stitched, images[i])
    #     print(f"Stitched {i + 1} images.")
    #     cv2.imwrite(f"outputImages/image{i}.jpg", img_stitched)

    # Save the final panorama
    output_path = "outputImages/final_panorama.jpg"
    os.makedirs("outputImages", exist_ok=True)
    cv2.imwrite(output_path, img_stitched)
    print(f"Panorama saved as '{output_path}'.")

if __name__ == "__main__":
    main()
