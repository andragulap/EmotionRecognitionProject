import librosa
import numpy as np
import os
import sys


def extract_augmented_features(file_path):
    try:

        y, sr = librosa.load(file_path, duration=3, offset=0.5)

        def to_spec(audio_signal):
            S = librosa.feature.melspectrogram(y=audio_signal, sr=sr, n_mels=128, hop_length=512)
            S_dB = librosa.power_to_db(S, ref=np.max)

            delta = librosa.feature.delta(S_dB)
            delta2 = librosa.feature.delta(S_dB, order=2)


            combined = np.stack([S_dB, delta, delta2], axis=-1)


            for i in range(3):
                ch = combined[:, :, i]
                std = ch.std()
                combined[:, :, i] = (ch - ch.mean()) / (std if std > 1e-8 else 1.0)

            T = 128
            if combined.shape[1] > T:
                combined = combined[:, :T, :]
            else:
                combined = np.pad(combined, ((0, 0), (0, T - combined.shape[1]), (0, 0)), mode='constant')

            return combined  # (128, 128, 3)


        spec_orig = to_spec(y)
        noise = np.random.randn(len(y))
        spec_noise = to_spec(y + 0.005 * noise)
        spec_pitch = to_spec(librosa.effects.pitch_shift(y=y, sr=sr, n_steps=2))


        try:
            spec_time = to_spec(librosa.effects.time_stretch(y=y, rate=1.1))
        except:

            spec_time = spec_orig

        return [spec_orig, spec_noise, spec_pitch, spec_time]

    except Exception as e:
        print(f"\nError at {file_path}: {e}")
        return None


def load_augmented_data(ravdess_path, crema_path):
    x_aug, y_aug, actors_aug, is_aug = [], [], [], []

    emotions_map = {'01': 0, '03': 1, '04': 2, '05': 3, '06': 4, '07': 5, '08': 6}
    crema_map = {'NEU': 0, 'HAP': 1, 'SAD': 2, 'ANG': 3, 'FEA': 4, 'DIS': 5}

    files_found = 0
    for root, _, files in os.walk(ravdess_path):
        for file in files:
            if file.endswith(".wav"):
                parts = file.split('-')
                emotion_code = parts[2]
                actor_id = int(parts[-1].split('.')[0])

                if emotion_code in emotions_map:
                    specs = extract_augmented_features(os.path.join(root, file))
                    if specs:
                        for i, s in enumerate(specs):
                            x_aug.append(s)
                            y_aug.append(emotions_map[emotion_code])
                            actors_aug.append(actor_id)
                            is_aug.append(0 if i == 0 else 1)  # 0=original, 1=augmentat
                        files_found += 1

                        sys.stdout.write(f"\rRavdess files found: {files_found}")
                        sys.stdout.flush()

    print(f"\nRAVDESS total images: {len(x_aug)}")

    crema_files = [f for f in os.listdir(crema_path) if f.endswith(".wav")]
    files_found_crema = 0
    for file in crema_files:
        parts = file.split('_')
        emotion_code = parts[2]
        actor_id = int(parts[0])

        if emotion_code in crema_map:
            specs = extract_augmented_features(os.path.join(crema_path, file))
            if specs:
                for i, s in enumerate(specs):
                    x_aug.append(s)
                    y_aug.append(crema_map[emotion_code])
                    actors_aug.append(actor_id)
                    is_aug.append(0 if i == 0 else 1)  # 0=original, 1=augmentat
                files_found_crema += 1
                sys.stdout.write(f"\rCrema files found: {files_found_crema + files_found}")
                sys.stdout.flush()

    print(f"\nCREMA total images: {len(x_aug)}")
    return np.array(x_aug), np.array(y_aug), np.array(actors_aug), np.array(is_aug)


RAVDESS = "C:/Users/Andra/Desktop/Licenta1/data_ravdess/data_ravdess_audio"
CREMA = "C:/Users/Andra/Desktop/Licenta1/data_cremad/data_CREMA_D_audio/AudioWAV"

X_final, y_final, act_final, is_aug_final = load_augmented_data(RAVDESS, CREMA)


np.save('X_augmented.npy', X_final)
np.save('y_augmented.npy', y_final)
np.save('actors_augmented.npy', act_final)
np.save('is_aug.npy', is_aug_final)

print(f"Total samples: {len(X_final)}")
print(f"  originale:   {(is_aug_final == 0).sum()}")
print(f"  augmentate:  {(is_aug_final == 1).sum()}")