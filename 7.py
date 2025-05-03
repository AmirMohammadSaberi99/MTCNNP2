#!/usr/bin/env python3
"""
count_faces_mtcnn.py

A standalone script that uses facenet-pytorch's MTCNN to detect and count faces in a given image.
Optionally draws bounding boxes around the detected faces and saves an annotated output image.

Usage:
    python count_faces_mtcnn.py --image PATH/TO/IMAGE.jpg [--output PATH/TO/OUTPUT.jpg]

Dependencies:
    pip install facenet-pytorch torch torchvision pillow
"""
import argparse
from pathlib import Path
import sys

from PIL import Image, ImageDraw
import torch
from facenet_pytorch import MTCNN

def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect and count faces in an image using MTCNN"
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
    return parser.parse_args()


def main():
    args = parse_args()

    # Check input image
    if not args.image.is_file():
        print(f"ERROR: Input image not found: {args.image}")
        sys.exit(1)

    # Select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialize MTCNN
    mtcnn = MTCNN(keep_all=True, device=device)

    # Load image
    img = Image.open(args.image).convert('RGB')

    # Detect faces
    boxes, probs = mtcnn.detect(img)
    num_faces = 0 if boxes is None else len(boxes)
    print(f"Detected {num_faces} face(s) in '{args.image.name}'")

    # If output requested, draw boxes and save
    if num_faces > 0 and args.output is not None:
        draw = ImageDraw.Draw(img)
        for box in boxes:
            draw.rectangle([box[0], box[1], box[2], box[3]], outline='lime', width=3)
        out_path = args.output
        # Ensure parent directory exists
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)
        print(f"Annotated image saved to: {out_path}")

    # If output not provided but faces detected, suggest using --output
    elif num_faces > 0 and args.output is None:
        print("Tip: rerun with --output OUTPUT_PATH to save an annotated image.")

if __name__ == '__main__':
    main()