import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from skimage.metrics import structural_similarity as ssim
import PolLettDS.PolLettDS as pld
from itertools import combinations

loaded_data, loaded_labels, labels_count = pld.load_pol_lett_db_from_files(
    '../PolLettDS/pol_lett_ds.bin',
    '../PolLettDS/pol_lett_ds_labels.bin')

unique_labels = np.unique(loaded_labels)
label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
mapped_labels = np.array([label_mapping[label] for label in loaded_labels])

classes = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'a', 'ą', 'b', 'c', 'ć', 'd', 'e', 'ę', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'ł', 'm', 'n', 'ń', 'o', 'ó', 'p', 'q',
    'r', 's', 'ś', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ź', 'ż',
    'A', 'Ą', 'B', 'C', 'Ć', 'D', 'E', 'Ę', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'Ł', 'M', 'N', 'Ń', 'O', 'Ó', 'P', 'Q',
    'R', 'S', 'Ś', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Ź', 'Ż'
]

# Parameters
target_char = '0'
target_class_index = classes.index(target_char)
block_size = 64 * 64

# Get all '0'
images_0 = []
for i in range(len(mapped_labels)):
    if mapped_labels[i] == target_class_index:
        start = i * block_size
        img = loaded_data[start:start + block_size].reshape(64, 64)
        images_0.append(img)

print(f"Znaleziono {len(images_0)} wystąpień znaku '{target_char}'.")

# Count SSIM
ssim_values = []
euclidean_values = []
for img1, img2 in combinations(images_0, 2):
    val, _ = ssim(img1, img2, full=True)
    ssim_values.append(val)

    # Euclidean distance
    dist = euclidean(img1.flatten(), img2.flatten())
    euclidean_values.append(dist)

plt.figure(figsize=(10, 6))
plt.hist(ssim_values, bins=50, color='skyblue', edgecolor='black')
plt.title(f'Histogram SSIM dla wszystkich wystąpień znaku "{target_char}"', fontsize=16)
plt.xlabel('SSIM', fontsize=16)
plt.ylabel('Liczba par', fontsize=16)
plt.tick_params(axis='both', labelsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"ssim_histogram_{target_char}.eps", format='eps')
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(euclidean_values, bins=30, color='steelblue', edgecolor='black')
plt.title(f'Histogram odległości euklidesowych dla wszystkich wystąpień znaku "{target_char}"', fontsize=16)
plt.xlabel("Odległość euklidesowa", fontsize=16)
plt.ylabel("Liczba par", fontsize=16)
plt.tick_params(axis='both', labelsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"euclidean_histogram_{target_char}.eps", format='eps')
plt.show()
