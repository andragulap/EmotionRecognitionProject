import os
import numpy as np
from audio_preprocessing import extract_spectrogram


def load_data(ravdess_path, crema_path):
    x, y, actors = [], [], []

    emotions_map = {
        '01': 0,  # neutral
        '03': 1,  # happy
        '04': 2,  # sad
        '05': 3,  # angry
        '06': 4,  # fearful
        '07': 5,  # disgust
        '08': 6  # surprised
    }

    crema_emotions_map = {
        'NEU': 0,  # neutral
        'HAP': 1,  # happy
        'SAD': 2,  # sad
        'ANG': 3,  # angry
        'FEA': 4,  # fear
        'DIS': 5   # disgust
    }

    count_ravdess = 0;
    count_crema = 0;

    for root, dirs, files in os.walk(ravdess_path):
        for file in files:
            if file.endswith(".wav"):

                part = file.split('-')
                emotion_code = part[2]
                actor_id = int(part[-1].split('.')[0])


                if emotion_code not in emotions_map:
                    continue

                file_path = os.path.join(root, file)


                try:
                    features = extract_spectrogram(file_path)
                    x.append(features)
                    y.append(emotions_map[emotion_code])
                    actors.append(actor_id)
                    count_ravdess += 1
                except Exception as e:
                    print(f"Error {file}: {e}")

    print(f" {count_ravdess} from RAVDESS.")

    for file in os.listdir(crema_path):
        if file.endswith(".wav"):
            parts = file.split('_')
            actor_id = parts[0]
            emotion_code = parts[2]

            if emotion_code in crema_emotions_map:
                try:
                    features = extract_spectrogram(os.path.join(crema_path, file))
                    x.append(features)
                    y.append(crema_emotions_map[emotion_code])
                    actors.append(actor_id)
                    count_crema += 1
                except Exception as e: print(f"Error {file}: {e}")

    print(f" {count_crema} from CREMA-D.")

    return np.array(x), np.array(y), np.array(actors)


RAVDESS_PATH = "C:/Users/Andra/Desktop/Licenta1/data_ravdess/data_ravdess_audio"
CREMA_PATH = "C:/Users/Andra/Desktop/Licenta1/data_cremad/data_CREMA_D_audio/AudioWAV"

X, y, actors = load_data(RAVDESS_PATH, CREMA_PATH)
if len(X.shape) == 3:
    X = np.expand_dims(X, axis=-1)

np.save('X_audio_spectrograms.npy', X)
np.save('y_audio_labels.npy', y)
np.save('actors.npy', actors)

actors = np.load('actors.npy')
print(len(actors))
print(actors[0])
print(actors[1])
print(actors[-1])
print(np.unique(actors))


print(f"Shape: {X.shape}") #shape[0] esantioane->imagini de sunet