import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt

X = np.load('X_video.npy')            # (N, 10, 64, 64)
y = np.load('y_video.npy')
actors = np.load('actors_video.npy')

X = np.expand_dims(X, axis=-1)

actors_clean = np.array([int(str(a).strip()) for a in actors])
unique_actors = np.unique(actors_clean)

ravdess_actors = [a for a in unique_actors if a <= 24]
crema_actors = [a for a in unique_actors if a >= 1001]

print(f'RAVDESS actors: {len(ravdess_actors)}')
print(f'CREMA-D actors: {len(crema_actors)}')


train_rav, temp_rav = train_test_split(ravdess_actors, test_size=0.30, random_state=42)
validation_rav, test_rav = train_test_split(temp_rav, test_size=0.50, random_state=42)

train_crema, temp_crema = train_test_split(crema_actors, test_size=0.30, random_state=42)
validation_crema, test_crema = train_test_split(temp_crema, test_size=0.50, random_state=42)

train_actors_list = list(train_rav) + list(train_crema)
validation_actors_list = list(validation_rav) + list(validation_crema)
test_actors_list = list(test_rav) + list(test_crema)

train_mask = np.isin(actors_clean, train_actors_list)
validation_mask = np.isin(actors_clean, validation_actors_list)
test_mask = np.isin(actors_clean, test_actors_list)

X_train, y_train = X[train_mask], y[train_mask]
X_valid, y_valid = X[validation_mask], y[validation_mask]
X_test, y_test = X[test_mask], y[test_mask]

print(f'Train: {len(X_train)} | Val: {len(X_valid)} | Test: {len(X_test)}')

weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(weights))


def build_model(input_shape=(10, 64, 64, 1), num_classes=7):
    inputs = layers.Input(shape=input_shape)

    x = layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))(inputs)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)


    x = layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)


    x = layers.TimeDistributed(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
    x = layers.TimeDistributed(layers.Dropout(0.3))(x)


    x = layers.TimeDistributed(layers.Flatten())(x)


    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.4, recurrent_dropout=0.3))(x)
    x = layers.Bidirectional(layers.LSTM(64, dropout=0.4, recurrent_dropout=0.3))(x)


    x = layers.Dense(64, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


model = build_model(input_shape=(10, 64, 64, 1), num_classes=7)
model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
checkpoint = ModelCheckpoint('video_model.keras', monitor='val_accuracy', save_best_only=True)
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5, verbose=1)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_valid, y_valid),
    class_weight=class_weights,
    callbacks=[early_stop, checkpoint, lr_reducer]
)

np.save('video_history.npy', history.history)


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train', color='steelblue')
plt.plot(history.history['val_accuracy'], label='Validare', color='orange')
plt.title('Video Model — Accuracy')
plt.xlabel('Epoci')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train', color='red')
plt.plot(history.history['val_loss'], label='Validare', color='darkred')
plt.title('Video Model — Loss')
plt.xlabel('Epoci')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('../results/video_training_curves.png', dpi=150)
plt.show()
