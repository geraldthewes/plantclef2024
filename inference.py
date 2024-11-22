from PIL import Image
import requests
import torch
from transformers import pipeline


# See https://huggingface.co/docs/transformers/main/en/tasks/image_classification



classifier = pipeline("image-classification", model="gerald29/plantclef2024", device=0)



image = Image.open('/mnt/data6/AI/data/plants/PlantCLEF2024/PlantCLEF2024singleplanttrainingdata/PlantCLEF2024/test/1392299/9f5004af16e14a935f0d0616df1457d95b946242.jpg')

classifier(image)
