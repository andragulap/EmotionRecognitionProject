import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


X = np.load('X_augmented.npy')
y = np.load('y_augmented.npy')
actors = np.load('actors_augmented.npy')
is_aug = np.load('is_aug.npy')



actors_clean = np.array([int(str(a).strip()) for a in actors])
unique_actors = np.unique(actors_clean)
ravdess_actors = [a for a in unique_actors if a <= 24]
crema_actors = [a for a in unique_actors if a >= 1001]


_, temp_rav = train_test_split(ravdess_actors, test_size=0.30, random_state=42)
_, test_rav = train_test_split(temp_rav, test_size=0.50, random_state=42)

_, temp_crema = train_test_split(crema_actors, test_size=0.30, random_state=42)
_, test_crema = train_test_split(temp_crema, test_size=0.50, random_state=42)

test_actors_list = list(test_rav) + list(test_crema)

test_mask = np.isin(actors_clean, test_actors_list) & (is_aug == 0)

X_test, y_test = X[test_mask], y[test_mask]

print(f"Test set: {len(X_test)} samples")



model = tf.keras.models.load_model('audio_model.keras')


loss, acc = model.evaluate(X_test, y_test)

print(f"\nTest acc: {acc*100:.2f}%")


y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)


emotions = ['Neutral', 'Happy', 'Sad', 'Angry', 'Fear', 'Disgust', 'Surprise']


print("\n" + classification_report(y_test, y_pred, target_names=emotions))

history = np.load('training_history.npy', allow_pickle=True).item()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history['accuracy'], label='Train', color='steelblue')
axes[0].plot(history['val_accuracy'], label='Validare', color='orange')
axes[0].axhline(y=acc, color='green', linestyle='--', label=f'Test: {acc*100:.1f}%')
axes[0].set_title('Accuracy')
axes[0].set_xlabel('Epoci')
axes[0].set_ylabel('Accuracy')
axes[0].legend()

axes[1].plot(history['loss'], label='Train', color='red')
axes[1].plot(history['val_loss'], label='Validare', color='darkred')
axes[1].set_title('Loss')
axes[1].set_xlabel('Epoci')
axes[1].set_ylabel('Loss')
axes[1].legend()

plt.suptitle(f'Antrenament model audio  |  Test accuracy: {acc*100:.2f}%', fontsize=13)
plt.tight_layout()
plt.savefig('../results/training_curves.png', dpi=150)
plt.show()


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=emotions, yticklabels=emotions, cmap='Blues')
plt.xlabel('Prezis')
plt.ylabel('Real')
plt.title(f'Confusion Matrix  |  Test accuracy: {acc*100:.2f}%')
plt.tight_layout()
plt.savefig('../results/confusion_matrix.png', dpi=150)
plt.show()