# Face Faker

**AI-powered face image generator for creating fake identities**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)

Generate realistic AI-generated face images with gender detection, background removal, and frontal face filtering.

## Features

- **AI-generated faces** - Fetch realistic faces from thispersondoesnotexist.com
- **Gender detection** - Automatic male/female classification using DeepFace
- **Background removal** - Transparent PNG output with rembg
- **Frontal face filtering** - dlib landmark-based pose detection
- **Batch processing** - Generate thousands of face images
- **Metadata export** - JSON and CSV metadata for each generated face

## Installation

```bash
pip install face-faker
```

For full features (gender detection, background removal, frontal filtering):
```bash
pip install face-faker[full]
```

## Quick Start

### Generate Face Images
```python
from face_faker import generate_id_faces

# Generate 100 face images
metadata = generate_id_faces(
    output_dir="output/faces",
    num_images=100
)

print(f"Generated {len(metadata)} faces")
```

### Generate with Background Removal
```python
from face_faker import generate_id_faces

# Generate faces with transparent background
metadata = generate_id_faces(
    output_dir="output/faces",
    num_images=50,
    remove_bg=True
)
```

### Generate Frontal Faces Only
```python
from face_faker import generate_id_faces

# Only save faces within 15 degrees of frontal
metadata = generate_id_faces(
    output_dir="output/frontal_faces",
    num_images=200,
    frontal_only=True,
    frontal_threshold=15
)
```

## CLI Commands

```bash
# Generate 100 face images
face-faker generate --count 100 --output-dir ./faces

# Generate with background removal
face-faker generate --count 50 --remove-bg --output-dir ./transparent

# Generate frontal faces only
face-faker generate --count 200 --frontal-only --threshold 15

# Show information
face-faker info
```

## API Reference

### `generate_id_faces()`

```python
generate_id_faces(
    output_dir="id_faces",      # Output directory
    num_images=100,             # Number of images to generate
    save_metadata=True,         # Save JSON/CSV metadata
    remove_bg=True,             # Remove background (transparent PNG)
    frontal_only=False,         # Only save frontal faces
    frontal_threshold=15,       # Frontal threshold in degrees
) -> List[Dict]                 # Returns list of metadata entries
```

### Metadata Format

Each generated face includes metadata:

```json
{
    "filename": "face_0001.png",
    "gender": "Male",
    "index": 1,
    "background_removed": true,
    "frontal_filtered": false
}
```

## Requirements

### Core
- Python 3.8+
- Pillow
- requests
- tqdm

### Full Features (optional)
- deepface - Gender detection
- opencv-python - Image processing
- numpy - Array operations
- rembg - Background removal
- dlib - Frontal face detection

## Models

For frontal face filtering, download the dlib shape predictor:

```bash
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
mv shape_predictor_68_face_landmarks.dat models/
```

## License

Proprietary License - See [LICENSE](LICENSE) for details.

## Author

Ali Sadeghi
