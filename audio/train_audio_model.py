import numpy as np
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils import class_weight
import tensorflow as tf

X = np.load('X_augmented.npy')
y = np.load('y_augmented.npy')
actors = np.load('actors_augmented.npy')


X = (X - X.min()) / (X.max() - X.min())

actors_clean = np.array([int(str(a).strip()) for a in actors])
unique_actors = np.unique(actors_clean)

ravdess_actors = [a for a in unique_actors if a <= 24]
crema_actors = [a for a in unique_actors if a >= 1001]

print(f" RAVDESS actors: {len(ravdess_actors)}")
print(f" CREMA-D actors: {len(crema_actors)}")


train_rav, temp_rav = train_test_split(ravdess_actors, test_size=9, random_state=42)
validation_rav, test_rav = train_test_split(temp_rav, test_size=3, random_state=42)


train_crema, temp_crema = train_test_split(crema_actors, test_size=18, random_state=42)
validation_crema, test_crema = train_test_split(temp_crema, test_size=9, random_state=42)


train_actors_list = list(train_rav) + list(train_crema)
validation_actors_list = list(validation_rav) + list(validation_crema)
test_actors_list = list(test_rav) + list(test_crema)


train_mask = np.isin(actors_clean, train_actors_list)
validation_mask = np.isin(actors_clean, validation_actors_list)
test_mask = np.isin(actors_clean, test_actors_list)


X_train, y_train = X[train_mask], y[train_mask]
X_valid, y_valid = X[validation_mask], y[validation_mask]
X_test, y_test = X[test_mask], y[test_mask]


if len(X_train.shape) == 3:
    X_train = np.expand_dims(X_train, axis=-1)
    X_valid = np.expand_dims(X_valid, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)


# np.random.seed(42)
# np.random.shuffle(unique_actors)
#
# train_actors = unique_actors[:19]
# test_actors  = unique_actors[19:]
#
# train_idx = np.isin(actors, train_actors)
# test_idx  = np.isin(actors, test_actors)
#
# X_train, X_test = X[train_idx], X[test_idx]
# y_train, y_test = y[train_idx], y[test_idx]


weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y = y_train
)
class_weights = dict(enumerate(weights))


def build_model(input_shape, num_classes):
    model = models.Sequential([

        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),


        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),


        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),


        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


model = build_model(input_shape=(128, 128, 1), num_classes=7)


early_stop = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
checkpoint = ModelCheckpoint('audio_model.keras', monitor='val_accuracy', save_best_only=True)
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=0.00001, verbose=1)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_valid, y_valid),
    class_weight=class_weights,
    callbacks=[early_stop, checkpoint, lr_reducer]
)


#model.save('audio_spectrogram_model.h5')

plt.figure(figsize=(12, 5))


plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation_accuracy')
plt.title('Audio Model')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training', color='red')
plt.plot(history.history['val_loss'], label='Validation', color='darkred')
plt.title('Loss function')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()