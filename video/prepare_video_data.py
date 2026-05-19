import cv2
import numpy as np
import os
import sys

N_FRAMES = 10
IMG_SIZE = 64

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < N_FRAMES:
        cap.release()
        return None

    indices = np.linspace(0, total_frames - 2, N_FRAMES, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        if len(faces) > 0:

            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            face = gray[y:y + h, x:x + w]
        else:

            h, w = gray.shape
            size = min(h, w)
            y0 = (h - size) // 2
            x0 = (w - size) // 2
            face = gray[y0:y0 + size, x0:x0 + size]

        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = face.astype(np.float32) / 255.0  # normalizare [0, 1]
        frames.append(face)

    cap.release()

    if len(frames) != N_FRAMES:
        return None

    return np.stack(frames, axis=0)  # (10, 64, 64)


def load_ravdess_video(ravdess_path):
    X, y, actors = [], [], []

    emotions_map = {'01': 0, '03': 1, '04': 2, '05': 3, '06': 4, '07': 5, '08': 6}

    files_found = 0
    skipped = 0
    errors = 0
    for root, _, files in os.walk(ravdess_path):
        for file in files:
            if file.endswith('.mp4'):
                parts = file.split('-')
                if len(parts) < 7:
                    continue

                vocal_channel = parts[1]
                emotion_code = parts[2]
                actor_id = int(parts[-1].split('.')[0])


                if vocal_channel != '01':
                    skipped += 1
                    continue

                if emotion_code not in emotions_map:
                    skipped += 1
                    continue

                frames = extract_frames(os.path.join(root, file))
                if frames is not None:
                    X.append(frames)
                    y.append(emotions_map[emotion_code])
                    actors.append(actor_id)
                    files_found += 1
                else:
                    errors += 1

                sys.stdout.write(f'\rRAVDESS: {files_found} ok, {skipped} sarite, {errors} erori')
                sys.stdout.flush()

    print(f'\nRAVDESS total: {files_found} clipuri procesate, {skipped} sarite, {errors} erori')
    return X, y, actors


def load_crema_video(crema_path):
    X, y, actors = [], [], []

    crema_map = {'NEU': 0, 'HAP': 1, 'SAD': 2, 'ANG': 3, 'FEA': 4, 'DIS': 5}

    files = [f for f in os.listdir(crema_path) if f.endswith('.flv')]
    files_found = 0
    errors = 0

    for file in files:
        parts = file.split('_')
        if len(parts) < 3:
            continue

        emotion_code = parts[2]
        actor_id = int(parts[0])

        if emotion_code not in crema_map:
            continue

        frames = extract_frames(os.path.join(crema_path, file))
        if frames is not None:
            X.append(frames)
            y.append(crema_map[emotion_code])
            actors.append(actor_id)
            files_found += 1
        else:
            errors += 1

        sys.stdout.write(f'\rCREMA-D: {files_found} ok, {errors} erori')
        sys.stdout.flush()

    print(f'\nCREMA-D total: {files_found} clipuri procesate')
    return X, y, actors


if __name__ == '__main__':
    RAVDESS_VIDEO = 'C:/Users/Andra/Desktop/Licenta1/data_ravdess/data_ravdess_video'
    CREMA_VIDEO = 'C:/Users/Andra/Desktop/dowCrema/crema-d-mirror/VideoFlash'

    print('Procesare RAVDESS...')
    X_rav, y_rav, act_rav = load_ravdess_video(RAVDESS_VIDEO)

    print('Procesare CREMA-D...')
    X_cre, y_cre, act_cre = load_crema_video(CREMA_VIDEO)

    X_all = np.array(X_rav + X_cre, dtype=np.float32)   # (N, 10, 64, 64)
    y_all = np.array(y_rav + y_cre, dtype=np.int32)
    actors_all = np.array(act_rav + act_cre, dtype=np.int32)

    print(f'\nTotal: {len(X_all)} clipuri')
    print(f'Shape X: {X_all.shape}')
    print(f'Distributie clase: {dict(zip(*np.unique(y_all, return_counts=True)))}')

    np.save('X_video.npy', X_all)
    np.save('y_video.npy', y_all)
    np.save('actors_video.npy', actors_all)

    print('Salvat: X_video.npy, y_video.npy, actors_video.npy')
