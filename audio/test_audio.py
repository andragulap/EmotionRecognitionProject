import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


X = np.load('X_audio_spectrograms.npy')
y = np.load('y_audio_labels.npy')
actors = np.load('actors.npy')


X = (X - X.min()) / (X.max() - X.min())
if len(X.shape) == 3:
    X = np.expand_dims(X, axis=-1)

actors_clean = np.array([int(str(a).strip()) for a in actors])
unique_actors = np.unique(actors_clean)
ravdess_actors = [a for a in unique_actors if a <= 24]
crema_actors = [a for a in unique_actors if a >= 1001]


_, temp_rav = train_test_split(ravdess_actors, test_size=9, random_state=42)
_, test_rav = train_test_split(temp_rav, test_size=3, random_state=42)

_, temp_crema = train_test_split(crema_actors, test_size=18, random_state=42)
_, test_crema = train_test_split(temp_crema, test_size=9, random_state=42)

test_actors_list = list(test_rav) + list(test_crema)
test_mask = np.isin(actors_clean, test_actors_list)

X_test, y_test = X[test_mask], y[test_mask]



model = tf.keras.models.load_model('audio_model.keras')


loss, acc = model.evaluate(X_test, y_test)

print(f"\nTest acc: {acc*100:.2f}%")


y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)


emotions = ['Neutral', 'Happy', 'Sad', 'Angry', 'Fear', 'Disgust', 'Surprise']


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=emotions, yticklabels=emotions, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion matrix')
plt.show()