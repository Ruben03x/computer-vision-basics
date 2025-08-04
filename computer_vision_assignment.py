import cv2, numpy as np, matplotlib.pyplot as plt
import os

def gaussian_filter(size, sigma):
    size = int(size)
    if size % 2 == 0:
        size += 1
    ax = np.linspace(-(size // 2), size // 2, size, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    f = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return f / f.sum()

def convolution_2d(img, filter):
    k = filter.shape[0] // 2
    h, w = img.shape
    padded = np.pad(img, ((k, k), (k, k)), mode="constant")
    out = np.zeros_like(img, dtype=np.float32)
    for i in range(h):
        for j in range(w):
            out[i, j] = np.sum(
                padded[i : i + filter.shape[0], j : j + filter.shape[1]] * filter
            )
    return out

class ComputerVisionAssignment:
    def __init__(self):
        self.output_dir = "output"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def save_image(self, image, filename):
        """Save image to output directory as a high-quality PDF"""
        filepath = os.path.join(self.output_dir, filename)
        
        # Auto-convert to RGB if image is BGR
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image  # grayscale

        # Use matplotlib to save as PDF
        plt.figure(figsize=(image.shape[1] / 100, image.shape[0] / 100), dpi=100)
        if len(image_rgb.shape) == 2:
            plt.imshow(image_rgb, cmap='gray', vmin=0, vmax=255)
        else:
            plt.imshow(image_rgb)
        plt.axis('off')
        plt.tight_layout(pad=0)

        # Replace extension with .pdf if not already
        if not filepath.lower().endswith(".pdf"):
            filepath = os.path.splitext(filepath)[0] + ".pdf"

        plt.savefig(filepath, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved PDF image: {filepath}")
    
    # Question 1
    def color_thresholding_greenscreen(self, image_path, background_path):
        gs_image = cv2.imread(image_path)

        # We get HSV from the original image
        hue, saturation, value   = cv2.split(cv2.cvtColor(gs_image, cv2.COLOR_BGR2HSV))

        # Then we create a mask with the thresholds that found the best balance
        mask = ((hue >= 68) &
                (saturation >= 100) &
                (value >= 177)).astype(np.uint8)

        # We create the inverse mask
        green_mask = mask.astype(np.uint8) * 255
        inv_mask = 255 - green_mask

        # We apply the mask to the original image
        actors = (gs_image * (inv_mask[:, :, None] / 255)).astype(np.uint8)

        # We read the background image and resize it to match the original image size
        bg = cv2.imread(background_path)
        bg = cv2.resize(bg, (gs_image.shape[1], gs_image.shape[0]))
        
        # Create the composite through taking the actors and blending with the background
        composite = (actors + bg * (green_mask[:, :, None] / 255)).astype(np.uint8)

        # save & display
        self.save_image(gs_image, "Question1/1_original_greenscreen.jpg")
        self.save_image(inv_mask, "Question1/1_green_mask.jpg")
        self.save_image(actors, "Question1/1_actors_extracted.jpg")
        self.save_image(bg, "Question1/1_new_background.jpg")
        self.save_image(composite, "Question1/1_final_composite.jpg")

        return composite

    
    # Question 2
    def add_salt_pepper_noise(self, img: np.ndarray, d: float = 0.2) -> np.ndarray:
        noisy_image = img.copy()
        p = d / 2.0
        height, width = img.shape[:2]
        
        # For each channel, we add salt and pepper noise, intended for a grayscale image but we will apply it to each channel of a color image
        for channel in range(3):
            random_values = np.random.rand(height, width)

            for row in range(height):
                for col in range(width):
                    if random_values[row, col] < p:
                        noisy_image[row, col, channel] = 0
                    elif random_values[row, col] > 1 - p:
                        noisy_image[row, col, channel] = 255

        return noisy_image

    @staticmethod
    def median_filter_channel(channel: np.ndarray, filter_size: int = 3) -> np.ndarray:
        padding = filter_size // 2
        height, width = channel.shape

        padded_channel = np.pad(channel, padding, mode='edge')

        filtered_channel = np.empty_like(channel)

        # Iterate over each pixel in the original channel and apply the median filter
        for row in range(height):
            for col in range(width):
                row_start = row
                row_end = row + filter_size
                col_start = col
                col_end = col + filter_size
                
                local_window = padded_channel[row_start:row_end, col_start:col_end]
                median_value = np.median(local_window)
                filtered_channel[row, col] = median_value

        return filtered_channel
    
    def median_filter_denoising(self,
                                image_path: str,
                                densities=(0.1, 0.2, 0.3),
                                filter_sizes=(3, 5, 7)):

        original = cv2.imread(image_path)
        if original is None:
            raise FileNotFoundError(image_path)

        # For each density, we add noise and then apply median filtering
        for d in densities:
            noisy = self.add_salt_pepper_noise(original, d)
            self.save_image(noisy, f"Question2/2_noisy_d{d}.jpg")

            for k in filter_sizes:
                restored = np.zeros_like(noisy)
                for ch in range(3):
                    restored[:, :, ch] = self.median_filter_channel(noisy[:, :, ch], k)
                self.save_image(restored, f"Question2/2_filtered_d{d}_s{k}.jpg")

    # Question 3
    def unsharp_masking(self, image_path, filter_size=5, sigma=2.0, alpha=1.5):
        img = cv2.imread(image_path).astype(np.float32)
        # Extract the RGB channels and convert to grayscale using slide's values
        r, g, b = img[:, :, 2], img[:, :, 1], img[:, :, 0]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        # Apply Gaussian blur to the grayscale image
        filter = gaussian_filter(filter_size, sigma)
        # Convolve the grayscale image with the Gaussian filter
        blurred = convolution_2d(gray, filter)

        # Create the unsharp mask by subtracting the blurred image from the original grayscale image
        mask = gray - blurred
        mask_vis = mask - mask.min()
        if mask_vis.max() > 0:
            mask_vis = mask_vis / mask_vis.max() * 255
        mask_vis = mask_vis.astype(np.uint8)
        
        # Sharpen the original grayscale image by adding the scaled mask
        sharp = np.clip(gray + alpha * mask, 0, 255).astype(np.uint8)
        gray_u8 = gray.astype(np.uint8)
        blurred_u8 = blurred.astype(np.uint8)

        self.save_image(gray_u8, "Question3/3_original_gray.jpg")
        self.save_image(mask_vis, "Question3/3_mask_vis.jpg")
        self.save_image(sharp, "Question3/3_sharpened.jpg")
        self.save_image(blurred_u8, "Question3/3_blurred.jpg")
        return sharp, mask_vis
    
    # Question 4
    def _sizes(self, img, s):
        h, w = img.shape[:2]
        return h, w, int(h * s), int(w * s)

    def nearest_neighbor_interpolation(self, img, s):
        h, w = img.shape[:2]
        hn, wn = int(h * s), int(w * s)
        out = np.zeros((hn, wn, 3), dtype=img.dtype)

        # Use formula from slides
        for y_new in range(hn):
            for x_new in range(wn):
                x_old = x_new / s
                y_old = y_new / s

                x_r = min(round(x_old), w - 1)
                y_r = min(round(y_old), h - 1)

                out[y_new, x_new] = img[y_r, x_r]

        return out

    def bilinear_interpolation(self, img, s):
        h, w = img.shape[:2]
        hn, wn = int(h * s), int(w * s)
        out = np.zeros((hn, wn, 3), dtype=np.float32)

        for i in range(hn):
            for j in range(wn):
                x_new, y_new = j, i
                x_old, y_old = x_new / s, y_new / s

                x_f = int(np.floor(x_old))
                y_f = int(np.floor(y_old))
                x_c = min(x_f + 1, w - 1)
                y_c = min(y_f + 1, h - 1)

                dx = x_old - x_f
                dy = y_old - y_f

                for c in range(3):
                    # Use formula from slides
                    out[i, j, c] = \
                        (1 - dx) * (1 - dy) * img[y_f, x_f, c] + \
                        dx * (1 - dy) * img[y_f, x_c, c] + \
                        (1 - dx) * dy * img[y_c, x_f, c] + \
                        dx * dy * img[y_c, x_c, c]

        return out.astype(img.dtype)

    def image_resizing_comparison(self, path, scales=(0.5, 1.5, 2.0)):
        img = cv2.imread(path)
        results = []
        for scale in scales:
            # Perform nearest neighbor and bilinear interpolation
            nn = self.nearest_neighbor_interpolation(img, scale)
            bl = self.bilinear_interpolation(img, scale)
            results.append((scale, nn, bl))
            if scale > 1:
                height, width = nn.shape[:2]
                k = min(200, height // 2, width // 2)
                ch, cw = height // 2, width // 2
                self.save_image(nn[ch - k // 2 : ch + k // 2, cw - k // 2 : cw + k // 2], f"Question4/4_nn_closeup_scale_{scale}.jpg")
                self.save_image(bl[ch - k // 2 : ch + k // 2, cw - k // 2 : cw + k // 2], f"Question4/4_bl_closeup_scale_{scale}.jpg")
        return results
    
    # Question 5
    def detect_and_match_features(self, p1, p2):
        # Initially get the images and convert them to grayscale using slide values
        img1 = cv2.imread(p1)
        img2 = cv2.imread(p2)
        r1, g1, b1 = img1[:, :, 2], img1[:, :, 1], img1[:, :, 0]
        r2, g2, b2 = img2[:, :, 2], img2[:, :, 1], img2[:, :, 0]
        g1 = (0.2989 * r1 + 0.5870 * g1 + 0.1140 * b1).astype(np.uint8)
        g2 = (0.2989 * r2 + 0.5870 * g2 + 0.1140 * b2).astype(np.uint8)

        # Chose and use ORB to detect and compute the keypoints and descriptors
        orb = cv2.ORB_create(nfeatures=5000, scaleFactor=1.2, nlevels=4)
        kp1, des1 = orb.detectAndCompute(g1, None)
        kp2, des2 = orb.detectAndCompute(g2, None)

        # bf matcher to match the descriptors
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = sorted(bf.match(des1, des2), key=lambda m: m.distance) if des1 is not None and des2 is not None else []
        # Limit the number of matches to 100 for visualization
        good = matches[: min(100, len(matches))]

        # Draw the keypoints with very small circles
        img1_kp = img1.copy()
        img2_kp = img2.copy()
        for kp in kp1:
            cv2.circle(img1_kp, (int(kp.pt[0]), int(kp.pt[1])), 2, (0, 255, 0), 1)
        for kp in kp2:
            cv2.circle(img2_kp, (int(kp.pt[0]), int(kp.pt[1])), 2, (0, 255, 0), 1)
        img_m = cv2.drawMatches(
            img1,
            kp1,
            img2,
            kp2,
            good,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        self.save_image(img1_kp, "Question5/5_features_image1.jpg")
        self.save_image(img2_kp, "Question5/5_features_image2.jpg")
        self.save_image(img_m, "Question5/5_matches.jpg")

        plt.figure(figsize=(15, 8))
        plt.imshow(cv2.cvtColor(img_m, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

        return kp1, kp2, des1, des2, good
    
    # Question 6
    def evaluate_scale_robustness(self, path, scale_min=0.1, scale_max=3.0, step=0.1):
            imgA = cv2.imread(path)
            
            # Convert to grayscale using slide's values
            r, g, b = imgA[:, :, 2], imgA[:, :, 1], imgA[:, :, 0]
            gA = (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(np.uint8)
            
            # Initialize ORB detector with same parameters as in Question 5
            orb = cv2.ORB_create(nfeatures=5000, scaleFactor=1.2, nlevels=4)
            kpA, desA = orb.detectAndCompute(gA, None)

            scales = np.arange(scale_min, scale_max + step, step)
            acc = []
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

            # Iterate through each scale factor and compute the matching accuracy
            for scale in scales:
                imgB = self.bilinear_interpolation(imgA, scale)
                rB, gB, bB = imgB[:, :, 2], imgB[:, :, 1], imgB[:, :, 0]
                gB = (0.2989 * rB + 0.5870 * gB + 0.1140 * bB).astype(np.uint8)
                kpB, desB = orb.detectAndCompute(gB, None)
                if desB is None:
                    acc.append(0.0)
                    continue
                m = bf.match(desA, desB)
                if not m:
                    acc.append(0.0)
                    continue
                tol2 = 4.0
                ok = 0
                for mm in m:
                    xA, yA = kpA[mm.queryIdx].pt
                    xB, yB = kpB[mm.trainIdx].pt
                    if (xB - xA * scale) ** 2 + (yB - yA * scale) ** 2 <= tol2:
                        ok += 1
                acc.append(ok / len(m))

            plt.figure(figsize=(12, 6))
            plt.plot(scales, acc)
            plt.xlabel("Scale Factor")
            plt.ylabel("Matching Accuracy")
            plt.ylim(0, 1)
            plt.grid(alpha=0.3)
            plt.savefig("output/Question6/6_accuracy_vs_scale.pdf", dpi=300, bbox_inches="tight")
            plt.show()

            # For each scale, save the matches
            for scale in (0.5, 1.0, 2.0):
                if scale < scale_min or scale > scale_max:
                    continue
                imgB = self.bilinear_interpolation(imgA, scale)
                gB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)
                kpB, desB = orb.detectAndCompute(gB, None)
                if desB is None:
                    continue
                m = sorted(bf.match(desA, desB), key=lambda x: x.distance)[:100]
                imgM = cv2.drawMatches(
                    imgA, kpA, imgB, kpB, m, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )
                self.save_image(imgM, f"Question6/6_matches_scale_{scale}.jpg")

            return scales, acc

def main():
    core = ComputerVisionAssignment()
    try:
        core.color_thresholding_greenscreen("greenscreen.jpg", "background.jpg")
        core.median_filter_denoising("semper1.jpg")
        core.unsharp_masking("fox.jpg")
        core.image_resizing_comparison("semper1.jpg")
        
        # Test ORB parameters before running feature detection
        print("Running ORB parameter grid search...")
        core.test_orb_parameters("semper1.jpg", "semper2.jpg")
        
        core.detect_and_match_features("semper1.jpg", "semper2.jpg")
        core.evaluate_scale_robustness("semper1.jpg")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()
