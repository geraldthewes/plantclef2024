import argparse
from PIL import Image
from transformers import pipeline

# See https://huggingface.co/docs/transformers/main/en/tasks/image_classification

# Setup the argument parser
parser = argparse.ArgumentParser(description='Classify an image using a pre-trained model.')
parser.add_argument('image_path', type=str, help='Path to the image file')
parser.add_argument('--model', type=str, default="gerald29/plantclef2024", help='Path to the remote or local model directory (default: gerald29/plantclef2024)')



args = parser.parse_args()


# Load and classify the image
classifier = pipeline("image-classification", args.model, device=0)


image = Image.open(args.image_path)
result = classifier(image)

# Print the result
print(result)
