from matplotlib import pyplot as plt
import PolLettDB as pld

loaded_data, loaded_labels, labels_count = pld.load_pol_lett_db_from_files(
                                                'pol_lett_db.bin',
                                                'pol_lett_db_labels.bin')

d=1200
block_size=64*64
a=64*64 * d

chars = []
for i in range(a, a + 80 * block_size, block_size):
    chars.append(loaded_data[i:i+block_size].reshape(64,64))

plt.figure(figsize=(2, 5))
for i, img in enumerate(chars[:80]):
    plt.subplot(17, 5, i + 1)
    plt.imshow(img,  cmap='gray', vmin=0, vmax=255)
    plt.axis('off')

plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, wspace=0.1, hspace=0.1)
plt.tight_layout(pad=0.2)
plt.show()
