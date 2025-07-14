from django.apps import AppConfig
import os
import cv2
import numpy as np
import tqdm
import mediapipe as mp
import shutil
from scipy.spatial.distance import euclidean
from glob import glob
import tensorflow as tf

class ApplipsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "applips"


class VideoProcessor:
    def __init__(self, input_videos_path, output_frames_path, output_landmarks_path, frame_extension='.jpg'):
        self.input_videos_path = input_videos_path
        self.output_frames_path = output_frames_path
        self.output_landmarks_path = output_landmarks_path
        self.frame_extension = frame_extension
        
        self.mp_face_mesh = mp.solutions.face_mesh

        # MediaPipe mouth landmark indices
        self.MOUTH_INDICES = [
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
            95, 88, 178, 87, 14, 317, 402, 318, 324
        ]
        
    def create_directories(self):
        os.makedirs(self.output_frames_path, exist_ok=True)
        os.makedirs(self.output_landmarks_path, exist_ok=True)

    def extract_all_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    def process_frame(self, frame):
        with self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5) as face_mesh:
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                mouth_landmarks = []

                
                for idx in self.MOUTH_INDICES:
                    landmark = landmarks.landmark[idx]
                    mouth_landmarks.append([landmark.x, landmark.y, landmark.z])  # In [0, 1]

                
                height, width = frame.shape[:2]

                
                mouth_points = np.array([
                    (int(lm[0] * width), int(lm[1] * height)) for lm in mouth_landmarks
                ])

                
                x_min = np.min(mouth_points[:, 0])
                x_max = np.max(mouth_points[:, 0])
                y_min = np.min(mouth_points[:, 1])
                y_max = np.max(mouth_points[:, 1])

                
                margin_x = int((x_max - x_min) * 0.1)
                margin_y = int((y_max - y_min) * 0.1)

                
                x_min = max(0, x_min - margin_x)
                x_max = min(width, x_max + margin_x)
                y_min = max(0, y_min - margin_y)
                y_max = min(height, y_max + margin_y)

               
                mouth_region = frame[y_min:y_max, x_min:x_max]

                return np.array(mouth_landmarks), mouth_region

            return None, None
        
    def process_videos(self):
        # Créer/réinitialiser le dossier de frames
        if os.path.exists(self.output_frames_path):
            import shutil
            shutil.rmtree(self.output_frames_path)
        os.makedirs(self.output_frames_path, exist_ok=True)

        print(f"\nProcessing video file: {self.input_videos_path}")

        video_path = self.input_videos_path  
        if not os.path.isfile(video_path) or not video_path.lower().endswith(('.mp4', '.avi', '.mov')):
            print(f"[ERREUR] {video_path} n'est pas un fichier vidéo valide.")
            return

        video_name = os.path.basename(video_path)
        base_name = os.path.splitext(video_name)[0]

        frames = self.extract_all_frames(video_path)
        sequence_landmarks = []

        for i, frame in enumerate(frames):
            landmarks, mouth_region = self.process_frame(frame)
            if landmarks is not None and mouth_region is not None:
                sequence_landmarks.append(landmarks)

                frame_num = i + 1
                frame_filename = f"{base_name}_{frame_num:03d}{self.frame_extension}"
                frame_path = os.path.join(self.output_frames_path, frame_filename)  # DIRECT dans le dossier frames
                cv2.imwrite(frame_path, mouth_region)

        if sequence_landmarks:
            landmarks_filename = f"{base_name}_landmarks.npy"
            landmarks_path = os.path.join(self.output_landmarks_path, landmarks_filename)
            np.save(landmarks_path, np.array(sequence_landmarks))

    

                
                
                
## class comparaison
class ComparaisonClass:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True)
        self.LIPS_IDX = list(set([
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
            95, 88, 178, 87, 14, 317, 402, 318, 324, 308
        ]))

    def extract_lip_landmarks(self, image):
        results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None
        landmarks = results.multi_face_landmarks[0]
        lips = [(lm.x, lm.y) for i, lm in enumerate(landmarks.landmark) if i in self.LIPS_IDX]
        return np.array(lips).flatten()

    def compute_differences(self, landmarks_list):
        diffs = [0.0]
        for i in range(1, len(landmarks_list)):
            if landmarks_list[i] is not None and landmarks_list[i - 1] is not None:
                diff = euclidean(landmarks_list[i], landmarks_list[i - 1])
            else:
                diff = 0.0
            diffs.append(diff)
        return diffs

    def imread_unicode(self, path):
        try:
            stream = np.fromfile(path, dtype=np.uint8)
            image = cv2.imdecode(stream, cv2.IMREAD_COLOR)
            return image
        except Exception as e:
            print(f"[ERREUR imread_unicode] {path} : {e}")
            return None

    def lip_openness(self, landmarks):
        if landmarks is None:
            return np.inf
        y_coords = landmarks[1::2]
        return np.ptp(y_coords)

    def process_video_folder(self, video_folder, output_folder, total_frames=15):
       

        frames = sorted([f for f in os.listdir(video_folder) if f.endswith('.png') or f.endswith('.jpg')])
        if len(frames) < total_frames:
            print(f"[AVERTISSEMENT] {len(frames)} Pas assez de frames dans {video_folder}")
            return

        if len(frames) == total_frames:
            print(f"[INFO] {len(frames)} frames trouvées, aucune réduction nécessaire dans {video_folder}")
            return

        # 1. Extraction des landmarks
        landmarks = []
        for fname in frames:
            img_path = os.path.join(video_folder, fname)
            image = self.imread_unicode(img_path)
            lips = self.extract_lip_landmarks(image)
            landmarks.append(lips)

        openness = [self.lip_openness(lm) for lm in landmarks]

        start_idx = np.argmin(openness[:5])
        end_idx = len(frames) - 5 + np.argmin(openness[-5:])

        required_mid_frames = total_frames - 2

        diffs = self.compute_differences(landmarks)

        candidate_indices = list(range(start_idx + 1, end_idx))
        candidates = [(i, diffs[i]) for i in candidate_indices if landmarks[i] is not None]
        candidates_sorted = sorted(candidates, key=lambda x: x[1], reverse=True)
        mid_indices = [i for i, _ in candidates_sorted[:required_mid_frames]]

        if len(mid_indices) < required_mid_frames:
            remaining = [i for i in candidate_indices if i not in mid_indices]
            mid_indices += remaining[:required_mid_frames - len(mid_indices)]

        final_indices = sorted([start_idx] + mid_indices + [end_idx])

        if len(final_indices) != total_frames:
            print(f"[AVERTISSEMENT] Repli équidistant pour {video_folder}")
            final_indices = np.linspace(0, len(frames)-1, num=total_frames, dtype=int).tolist()

        os.makedirs(output_folder, exist_ok=True)
        for i, idx in enumerate(final_indices, 1):
            src = os.path.join(video_folder, frames[idx])
            dst = os.path.join(output_folder, f"{i:04d}.jpg")
            shutil.copyfile(src, dst)
            print("frames suvegardées :", dst)




def load_and_prepare_frames(frames_dir, target_size=(192, 96)):
    print(f"Chargement des frames depuis : {frames_dir}")
    FRAMES_PER_SAMPLE = 15    
    # Récupération et tri des images
    frame_paths = sorted(
        glob(os.path.join(frames_dir, '*.[jJ][pP][gG]')) +
        glob(os.path.join(frames_dir, '*.[jJ][pP][eE][gG]')) +
        glob(os.path.join(frames_dir, '*.[pP][nN][gG]'))
    )

    if len(frame_paths) != FRAMES_PER_SAMPLE:
        print(f"[ERREUR] {len(frame_paths)} frames trouvées au lieu de {FRAMES_PER_SAMPLE}")
        return None  # ou raise, selon ta stratégie

    frames = []
    for path in frame_paths:
        try:
            img = tf.keras.preprocessing.image.load_img(path, target_size=target_size)
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            frames.append(img_array)
        except Exception as e:
            print(f"[ERREUR] Échec de lecture de {path} : {e}")
            continue

    if len(frames) != FRAMES_PER_SAMPLE:
        print(f"[ERREUR] Seulement {len(frames)} frames valides chargées.")
        return None

    sequence = np.stack(frames, axis=0)  # (15, H, W, 3)
    sequence = np.expand_dims(sequence, axis=0)  # (1, 15, H, W, 3)
    
    print(f"[INFO] Frames chargées avec succès, shape finale : {sequence.shape}")
    return sequence