# Related third party imports
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image


def load_labels(labels_file_path):
    with open(labels_file_path, 'r') as f:
        labels = f.read().splitlines()
    return [f"a photo of {label.lower().replace(' ', '_')}" for label in labels]


def predict_image_text(image_path, model_path, processor_path, labels_file_path):
    # Load the trained model and processor
    model = CLIPModel.from_pretrained(model_path)
    processor = CLIPProcessor.from_pretrained(processor_path)

    # Set the model to evaluation mode
    model.eval()

    # Load and process the image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    # Load text candidates from the labels file
    text_candidates = load_labels(labels_file_path)
    text_inputs = processor(text=text_candidates, return_tensors="pt", padding=True)

    # Get the features from the model
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        text_features = model.get_text_features(**text_inputs)

    # Calculate the similarity between image and text features
    logits_per_image = (image_features @ text_features.T).softmax(dim=-1)

    # Get the highest similarity text
    max_similarity_index = logits_per_image.argmax().item()
    predicted_text = text_candidates[max_similarity_index]

    return predicted_text


# Example usage
image_path = ".data/food-101-test/images/baklava/Koskeroglu-Dry-Baklava-with-Pistachio.jpg"
model_path = "./FoodCLIP101"
processor_path = "./FoodCLIP101"
labels_file_path = ".data/food-101/meta/labels.txt"

predicted_text = predict_image_text(image_path, model_path, processor_path, labels_file_path)
print(f"Predicted text: {predicted_text}")
