from matplotlib import pyplot as plt
import PolLettDS.PolLettDS as pld

loaded_data, loaded_labels, labels_count = pld.load_pol_lett_db_from_files(
    'pol_lett_db.bin',
    'pol_lett_db_labels.bin')

d = 1200
block_size = 64 * 64
a = 64 * 64 * d

chars = []
labels = []
for i in range(a, a + 80 * block_size, block_size):
    chars.append(loaded_data[i:i + block_size].reshape(64, 64))
    labels.append(loaded_labels[i // block_size])

plt.figure(figsize=(4, 10))
for i, (img, label) in enumerate(zip(chars[:80], labels[:80])):
    plt.subplot(17, 5, i + 1)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.text(32, 70, str(label), fontsize=8, ha='center', va='top')

plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, wspace=0.1, hspace=0.1)
plt.tight_layout(pad=0.2)
plt.show()
