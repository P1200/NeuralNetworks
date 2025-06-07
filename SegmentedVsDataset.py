import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import PolLettDS.PolLettDS as pld
from skimage.metrics import structural_similarity as ssim


def preprocess_letter(letter_image, target_size=64, max_letter_size=30, thickness=2):
    # Dilation
    kernel = np.ones((thickness, thickness), np.uint8)
    dilated_image = cv2.dilate(letter_image, kernel, iterations=1)

    coords = cv2.findNonZero(dilated_image)
    x, y, w, h = cv2.boundingRect(coords)
    cropped = dilated_image[y:y + h, x:x + w]

    scale = max_letter_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    final_image = np.zeros((target_size, target_size), dtype=np.uint8)
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    final_image[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return final_image


loaded_data, loaded_labels, labels_count = pld.load_pol_lett_db_from_files(
    'PolLettDS/pol_lett_db.bin',
    'PolLettDS/pol_lett_db_labels.bin')

d = 1200
block_size = 64 * 64
a = 64 * 64 * d

classes = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'a', 'ą', 'b', 'c', 'ć', 'd', 'e', 'ę', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'ł', 'm', 'n', 'ń', 'o', 'ó', 'p', 'q',
    'r', 's', 'ś', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ź', 'ż',
    'A', 'Ą', 'B', 'C', 'Ć', 'D', 'E', 'Ę', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'Ł', 'M', 'N', 'Ń', 'O', 'Ó', 'P', 'Q',
    'R', 'S', 'Ś', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Ź', 'Ż'
]

letterFromDataset = loaded_data[a + 23 * block_size:a + 23 * block_size + block_size].reshape(64, 64)
label = loaded_labels[d + 23]

plt.figure(figsize=(2, 1))
plt.subplot(1, 2, 1)
plt.imshow(letterFromDataset, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
# plt.text(32, 70, str(classes[label]), fontsize=8, ha='center', va='top')

image = Image.open('../char_segmentation/letters/letter_0020.png')
image = np.array(image)
image = preprocess_letter(image)

plt.subplot(1, 2, 2)
plt.imshow(image, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
plt.show()

# Euclidean distance
euclidean_distance = np.linalg.norm(letterFromDataset.astype(np.float32) - image.astype(np.float32))

# SSIM
ssim_value, _ = ssim(letterFromDataset, image, full=True)

print(f"Odległość euklidesowa: {euclidean_distance:.2f}")
print(f"SSIM: {ssim_value:.4f}")
