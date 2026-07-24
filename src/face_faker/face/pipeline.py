"""
Face generation pipeline
"""

import os
import time
import json
import random
from io import BytesIO
from typing import Optional, List, Dict, Any

from PIL import Image, ImageOps
import requests
from tqdm import tqdm

from ..config import (
    DEFAULT_FACE_OUTPUT_DIR,
    DEFAULT_NUM_IMAGES,
    DEFAULT_FRONTAL_THRESHOLD,
    DEFAULT_DLIB_PREDICTOR,
)


def get_person_image() -> Optional[Image.Image]:
    """
    Fetch face image from thispersondoesnotexist.com

    Returns:
        PIL Image or None on failure
    """
    url = "https://thispersondoesnotexist.com/"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
    except Exception:
        pass
    return None


def remove_background_transparent(image_pil: Image.Image, grayscale: bool = True) -> Image.Image:
    """
    Remove background and make it transparent using rembg.

    Args:
        image_pil: Input PIL Image
        grayscale: If True, convert to grayscale while preserving transparency

    Returns:
        PIL Image with transparent background
    """
    from rembg import remove

    result_rgba = remove(image_pil)

    if grayscale:
        rgb = result_rgba.convert('RGB')
        gray = rgb.convert('L')
        result_gray = Image.new('LA', gray.size)
        result_gray.putdata(list(zip(gray.getdata(), result_rgba.split()[3].getdata())))
        return result_gray

    return result_rgba


def detect_head_pose(image_pil: Image.Image, ratio_threshold: float = 0.1) -> tuple:
    """
    Detect frontal face using eye visibility.

    Args:
        image_pil: Input PIL Image
        ratio_threshold: Threshold for EAR difference

    Returns:
        tuple: (pose_data, is_frontal)
    """
    import cv2
    import numpy as np
    import dlib

    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    detector = dlib.get_frontal_face_detector()
    predictor_path = str(DEFAULT_DLIB_PREDICTOR)

    if not os.path.exists(predictor_path):
        return None, False

    predictor = dlib.shape_predictor(predictor_path)

    faces = detector(gray, 1)
    if len(faces) == 0:
        return None, False

    landmarks = predictor(gray, faces[0])

    left_eye = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(36, 42)])
    right_eye = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(42, 48)])

    left_eye_width = np.linalg.norm(left_eye[3] - left_eye[0])
    right_eye_width = np.linalg.norm(right_eye[3] - right_eye[0])

    left_eye_height = np.linalg.norm(left_eye[1] - left_eye[5])
    right_eye_height = np.linalg.norm(right_eye[1] - right_eye[5])

    left_ear = left_eye_height / left_eye_width if left_eye_width > 0 else 0
    right_ear = right_eye_height / right_eye_width if right_eye_width > 0 else 0

    ear_diff = abs(left_ear - right_ear)
    is_frontal = ear_diff < ratio_threshold

    return (left_ear, right_ear, ear_diff), is_frontal


def detect_gender(image_pil: Image.Image) -> str:
    """
    Detect gender from image using DeepFace.

    Args:
        image_pil: Input PIL Image

    Returns:
        'Male' or 'Female'
    """
    import numpy as np
    from deepface import DeepFace

    img_array = np.array(image_pil)
    analysis = DeepFace.analyze(
        img_path=img_array,
        actions=['gender'],
        enforce_detection=False
    )

    if analysis:
        return analysis[0]['dominant_gender']
    return 'Unknown'


def save_image_metadata(metadata_entry: Dict[str, Any], output_dir: str) -> None:
    """Save individual image metadata as JSON file."""
    filename_without_ext = metadata_entry['filename'].replace('.png', '')
    metadata_file = os.path.join(output_dir, f"{filename_without_ext}_metadata.json")

    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata_entry, f, ensure_ascii=False, indent=2)


def generate_id_faces(
    output_dir: str = DEFAULT_FACE_OUTPUT_DIR,
    num_images: int = DEFAULT_NUM_IMAGES,
    save_metadata: bool = True,
    remove_bg: bool = True,
    frontal_only: bool = False,
    frontal_threshold: int = DEFAULT_FRONTAL_THRESHOLD,
) -> List[Dict[str, Any]]:
    """
    Generate face images with optional filtering for frontal faces.

    Args:
        output_dir: Directory to save images
        num_images: Number of images to generate
        save_metadata: Save JSON and CSV metadata
        remove_bg: Remove background and make transparent
        frontal_only: Only save frontal faces
        frontal_threshold: Threshold for frontal detection (degrees)

    Returns:
        List of metadata entries
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    successful = 0
    failed = 0
    filtered_out = 0
    male_count = 0
    female_count = 0
    unknown_count = 0
    start_time = time.time()

    metadata_list = []

    pbar = tqdm(total=num_images, desc="Generating images", unit="image")
    attempts = 0
    max_attempts = num_images * 2

    while successful < num_images and attempts < max_attempts:
        attempts += 1

        try:
            picture = get_person_image()
            if picture is None:
                failed += 1
                pbar.update(0)
                continue

            if not isinstance(picture, Image.Image):
                picture = Image.open(BytesIO(picture))

            if picture.mode in ('RGBA', 'LA', 'P'):
                picture = picture.convert('RGB')

            if frontal_only:
                pose_angles, is_frontal = detect_head_pose(picture, frontal_threshold)
                if not is_frontal:
                    filtered_out += 1
                    pbar.update(0)
                    continue

            sleep = random.randint(1, 3)
            time.sleep(sleep)
            gender = detect_gender(picture)

            if remove_bg:
                processed_img = remove_background_transparent(picture, grayscale=True)
            else:
                processed_img = ImageOps.grayscale(picture)

            output_path = os.path.join(output_dir, f"face_{successful+1:04d}.png")
            processed_img.save(output_path)

            metadata_entry = {
                "filename": f"face_{successful+1:04d}.png",
                "gender": gender,
                "index": successful + 1,
                "background_removed": remove_bg,
                "frontal_filtered": frontal_only
            }
            if frontal_only and pose_angles:
                metadata_entry["pose"] = {
                    "yaw": round(pose_angles[0], 2),
                    "pitch": round(pose_angles[1], 2),
                    "roll": round(pose_angles[2], 2)
                }

            metadata_list.append(metadata_entry)

            if save_metadata:
                save_image_metadata(metadata_entry, output_dir)

            successful += 1
            if gender == 'Male':
                male_count += 1
            elif gender == 'Female':
                female_count += 1
            else:
                unknown_count += 1

            pbar.update(1)
            pbar.set_postfix({
                'Male': male_count,
                'Female': female_count,
                'Unknown': unknown_count,
                'Filtered': filtered_out
            })

        except Exception:
            failed += 1
            pbar.update(0)
            continue

    pbar.close()

    if save_metadata:
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_list, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(output_dir, "gender_statistics.csv")
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("filename,gender,background_removed,frontal_filtered\n")
        for entry in metadata_list:
            f.write(f"{entry['filename']},{entry['gender']},{entry['background_removed']},{entry['frontal_filtered']}\n")

    return metadata_list
