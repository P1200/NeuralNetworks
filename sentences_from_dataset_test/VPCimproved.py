import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread('pan_tadeusz.png', cv2.IMREAD_GRAYSCALE)

# _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
plt.imshow(image, cmap='gray')

horizontal_sum = np.sum(image, axis=1)

threshold = np.max(horizontal_sum) * 0.06
min_line_gap = 2

potential_splits = []
start = 0
in_text = False
lowest_start = None

for i in range(len(horizontal_sum)):

    if lowest_start is None:
        lowest_start = i

    if horizontal_sum[i] <= horizontal_sum[lowest_start]:
        lowest_start = i

    if horizontal_sum[i] > threshold and not in_text:
        start = lowest_start
        lowest_start = None
        in_text = True
    elif horizontal_sum[i] <= threshold and in_text:
        end = i - 1
        for j in range(i, min(i + 18, len(horizontal_sum))):
            if horizontal_sum[j] < horizontal_sum[end]:
                end = j
        in_text = False
        potential_splits.append((start, end))

potential_splits.append((start, len(horizontal_sum) - 1))

color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

for y1, y2 in potential_splits:
    cv2.line(color_image, (0, y1), (color_image.shape[1]-1, y1), (255, 0, 0), 2)
    cv2.line(color_image, (0, y2), (color_image.shape[1]-1, y2), (0, 0, 255), 2)

plt.figure(figsize=(12, 8))
plt.imshow(color_image)
plt.title('Podział na linie')
plt.axis('off')
plt.show()

line_positions = []
i = 0

while i < len(potential_splits):
    start, end = potential_splits[i]

    j = i + 1
    while j < len(potential_splits) and potential_splits[j][0] - end < min_line_gap:
        end = potential_splits[j][1]
        j += 1

    if j > i + 1:
        search_range = range(potential_splits[i][0], potential_splits[j-1][1])
        min_index = np.argmin(horizontal_sum[search_range])
        best_split = potential_splits[i][0] + min_index

        line_positions.append((potential_splits[i][0], potential_splits[j-1][1]))
    else:
        line_positions.append((start, end))

    i = j

color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

for start, end in line_positions:
    cv2.line(color_image, (0, start), (color_image.shape[1]-1, start), (255, 0, 0), 2)
    cv2.line(color_image, (0, end), (color_image.shape[1]-1, end), (0, 0, 255), 2)

plt.figure(figsize=(12, 8))
plt.imshow(color_image)
plt.title('Podział na linie')
plt.axis('off')
plt.show()

hist_threshold_ratio = 0.1
min_distance = 15

os.makedirs('literki', exist_ok=True)
letter_counter = 0

for idx, (y1, y2) in enumerate(line_positions):
    line_img = image[y1:y2, :]
    h, w = line_img.shape
    white_pixels = cv2.countNonZero(line_img)

    if h < 10 or (w / h > 20 > h) or white_pixels < 50:
        continue

    vertical_sum = np.sum(line_img, axis=0)

    hist_threshold = np.max(vertical_sum) * hist_threshold_ratio
    binary_histogram = (vertical_sum > hist_threshold).astype(np.uint8)

    start = None
    blocks = []
    for i, val in enumerate(binary_histogram):
        if val == 1 and start is None:
            start = i
        elif val == 0 and start is not None:
            end = i
            blocks.append((start, end))
            start = None

    if start is not None:
        blocks.append((start, len(binary_histogram)))

    for x1, x2 in blocks:
        letter_crop = line_img[:, x1:x2]
        if letter_crop.shape[1] > 2 and letter_crop.shape[0] > 2:  # filtracja śmieci
            letter_counter += 1
            filename = f'literki/letter_{letter_counter:04d}.png'
            cv2.imwrite(filename, letter_crop)  # odwrócenie kolorów

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    color_line = cv2.cvtColor(line_img, cv2.COLOR_GRAY2BGR)
    for x1, x2 in blocks:
        cv2.line(color_line, (x1, 0), (x1, h-1), (0, 255, 0), 1)
        cv2.line(color_line, (x2, 0), (x2, h-1), (0, 255, 0), 1)

    ax1.imshow(color_line)
    ax1.set_title(f'Linia {idx+1}: podział na literki (zielone)')
    ax1.axis('off')

    ax2.plot(vertical_sum)
    ax2.axhline(hist_threshold, color='red', linestyle='--', label=f'Threshold {hist_threshold:.0f}')
    ax2.set_title(f'Histogram pionowy dla linii {idx+1}')
    ax2.legend()

    plt.tight_layout()
    plt.show()

print(f"Zapisano {letter_counter} literek w folderze 'literki/'.")
