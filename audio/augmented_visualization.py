import numpy as np
import matplotlib.pyplot as plt
import librosa.display

X = np.load('X_augmented.npy', mmap_mode='r')
y = np.load('y_augmented.npy')


target_label = 3
idx = np.where(y == target_label)[0][0]

specs = X[idx:idx+4]

titles = ['A. Original (Angry)', 'B. Noise Injection', 'C. Pitch Shift', 'D. Time Stretch']

plt.figure(figsize=(15, 8))
for i in range(4):
    plt.subplot(2, 2, i+1)
    librosa.display.specshow(specs[i].reshape(128, 128), sr=22050, x_axis='time', y_axis='mel', cmap='magma')
    plt.title(titles[i], fontsize=12, fontweight='bold')
    plt.colorbar(format='%+2.0f dB')

plt.tight_layout()
plt.show()