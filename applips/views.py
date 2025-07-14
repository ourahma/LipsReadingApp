from django.shortcuts import render
from .apps import *
import os
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from tensorflow.keras.models import load_model
import tensorflow as tf



MODEL_PATH = os.path.join("applips", "model", "best_model2.keras")
FRAMES_INPUT_DIR = os.path.join("applips", "model", "frames") 
FRAMES_SELECTED_DIR = os.path.join("applips", "model", "selected_frames")  

CLASS_NAMES = ['akala', 'bayt', 'darasa', 'kataba', 'kitab', 'madrasa', 'qalam', 'talib', 'usra', 'bab']
FRAMES_PER_SAMPLE = 15
IMG_SIZE = (192, 96)
NUM_CLASSES = 10


def index(request):
    video_url = None
    if request.method == 'POST' and request.FILES.get('video'):
        video = request.FILES['video']
        fs = FileSystemStorage(location='media/uploads')  
        filename = fs.save(video.name, video)
        video_url = fs.url(f"uploads/{filename}")

    return render(request, 'index.html', {'video_url': video_url})

from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import uuid

### uploader la video 

def predict_word(request):
    if request.method == "POST":
        video_file = request.FILES.get("video")

        if not video_file:
            return render(request, "index.html", {"error": "Aucune vidéo sélectionnée."})

        # Générer un nom unique
        unique_filename = f"{uuid.uuid4().hex}_{video_file.name}"
        save_path = os.path.join("uploads", unique_filename)
        full_path = os.path.join(settings.MEDIA_ROOT, save_path)

        # Sauvegarder la vidéo
        path = default_storage.save(save_path, ContentFile(video_file.read()))
        video_path = os.path.join(settings.MEDIA_ROOT, path)
        video_url = f"/media/uploads/{os.path.basename(path)}"

        # Traitement
        try:
            prediction, accuracy = process_video(video_path)
        except Exception as e:
            return render(request, "index.html", {
                "error": f"Erreur pendant le traitement : {str(e)}"
            })

        return render(request, "index.html", {
            "video_url": video_url,
            "prediction": prediction,
            "accuracy": np.round(accuracy * 100, 2)
        })

    return render(request, "index.html")






def process_video(video_path):
    print("Chemin de la vidéo :", video_path)
    
    # 1. Découpage de la vidéo brute en frames
    processor = VideoProcessor(
        input_videos_path=video_path,
        output_frames_path=FRAMES_INPUT_DIR,
        output_landmarks_path=os.path.join("applips", "model", "landmarks"),
        frame_extension='.jpg'
    )
    processor.process_videos()
    os.makedirs(FRAMES_SELECTED_DIR, exist_ok=True)
    # 2. Sélection intelligente des 15 meilleures frames
    comp = ComparaisonClass()
    comp.process_video_folder(
        video_folder=FRAMES_INPUT_DIR,
        output_folder=FRAMES_SELECTED_DIR,
        total_frames=FRAMES_PER_SAMPLE
    )

    # 3. Chargement du modèle et prédiction
    model = tf.keras.models.load_model(MODEL_PATH)
    sequence = load_and_prepare_frames(FRAMES_SELECTED_DIR, target_size=IMG_SIZE)
    prediction = model.predict(sequence) 

    predicted_idx = np.argmax(prediction)
    confidence = float(np.max(prediction))
    predicted_word = CLASS_NAMES[predicted_idx]

    return predicted_word, confidence