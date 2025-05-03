#!/usr/bin/env python3
"""
count_faces_mtcnn.py

A standalone script that uses facenet-pytorch's MTCNN to detect faces in an image,
count them, save individual face crops with timestamped filenames, and optionally
draw bounding boxes on the original image.

Usage:
    python count_faces_mtcnn.py --image PATH/TO/IMAGE.jpg \
        [--output PATH/TO/ANNOTATED.jpg] [--crop-dir PATH/TO/CROPS]

Dependencies:
    pip install facenet-pytorch torch torchvision pillow
"""
import argparse
from pathlib import Path
import sys
from datetime import datetime

from PIL import Image, ImageDraw
import torch
from facenet_pytorch import MTCNN

def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect, count, and crop faces in an image using MTCNN"
    )
    parser.add_argument(
        '-i', '--image',
        required=True,
        type=Path,
        help='Path to the input image file'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=None,
        help='Path to save the annotated output image (optional)'
    )
    parser.add_argument(
        '-c', '--crop-dir',
        type=Path,
        default=None,
        help='Directory to save cropped face images (optional)'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate input
    if not args.image.is_file():
        print(f"ERROR: Input image not found: {args.image}")
        sys.exit(1)

    # Select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialize MTCNN
    mtcnn = MTCNN(keep_all=True, device=device)

    # Load image with PIL
    img = Image.open(args.image).convert('RGB')

    # Detect faces
    boxes, probs = mtcnn.detect(img)
    num_faces = 0 if boxes is None else len(boxes)
    print(f"Detected {num_faces} face(s) in '{args.image.name}'")

    # Save crops if requested
    if num_faces > 0:
        # Determine crop directory
        if args.crop_dir:
            crop_dir = args.crop_dir
        else:
            crop_dir = args.image.parent / f"{args.image.stem}_crops"
        crop_dir.mkdir(parents=True, exist_ok=True)

        # Timestamp base for filenames
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        for idx, box in enumerate(boxes, start=1):
            # box = [x1, y1, x2, y2]
            x1, y1, x2, y2 = [int(coord) for coord in box]
            face_crop = img.crop((x1, y1, x2, y2))
            crop_filename = f"{args.image.stem}_{timestamp}_{idx}.jpg"
            crop_path = crop_dir / crop_filename
            face_crop.save(crop_path)
            print(f"Saved crop {idx} to: {crop_path}")

    # Draw and save annotated image if requested
    if num_faces > 0 and args.output:
        draw = ImageDraw.Draw(img)
        for box in boxes:
            x1, y1, x2, y2 = [int(coord) for coord in box]
            draw.rectangle([x1, y1, x2, y2], outline='lime', width=3)
        out_path = args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)
        print(f"Annotated image saved to: {out_path}")
    elif num_faces > 0 and not args.output:
        print("Tip: rerun with --output OUTPUT_PATH to save an annotated image.")

if __name__ == '__main__':
    main()
