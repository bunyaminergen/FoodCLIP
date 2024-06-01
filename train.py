# Standard library imports
import os
import logging

# Related third party imports
import torch
from tqdm import tqdm
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from transformers import CLIPProcessor, CLIPModel

# Local imports
from src.utils.utils import Download
from src.preprocess.preprocess import get_data_loaders

os.makedirs('.logs', exist_ok=True)

log_file = '.logs/logs.txt'
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')

with open(log_file, 'w') as log_header_file:
    log_header_file.write("Training Logs\n")


def log_message(message):
    print(message)
    logging.info(message)


def process_images(processor, images, device):
    inputs = processor(images=images, return_tensors="pt", padding=True, do_rescale=False)
    return {k: v.to(device) for k, v in inputs.items()}


def get_texts_for_labels(labels, label_to_text):
    return [f"a photo of {label_to_text[label]}" for label in labels]


def train_clip_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    model.to(device)
    model.train()

    optimizer = Adam(model.parameters(), lr=5e-5)
    criterion = CrossEntropyLoss()

    data_root = '.data/food-101'
    train_loader, test_loader = get_data_loaders(data_root, batch_size=32)

    with open(os.path.join(data_root, 'meta', 'labels.txt'), 'r') as f:
        labels = f.read().splitlines()

    label_to_text = {label.lower().replace(' ', '_'): text for label, text in zip(labels, labels)}

    for epoch in range(3):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()

            inputs = process_images(processor, images, device)

            texts = get_texts_for_labels(labels, label_to_text)
            text_inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)

            # Print image and text pairs
            # print(f"Batch {i + 1}")
            # for img, text in zip(images, texts):
            #     print(f"Image: {img}, Text: {text}")

            image_features = model.get_image_features(**inputs)
            text_features = model.get_text_features(**text_inputs)

            logits_per_image = image_features @ text_features.T
            logits_per_text = text_features @ image_features.T

            ground_truth = torch.arange(len(images), dtype=torch.long, device=device)

            image_loss = criterion(logits_per_image, ground_truth)
            text_loss = criterion(logits_per_text, ground_truth)
            loss = (image_loss + text_loss) / 2
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:  # Log every 100 mini-batches
                log_message(f"[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0

        avg_train_loss = running_loss / len(train_loader)
        log_message(f"Epoch {epoch + 1} completed with average train loss: {avg_train_loss:.3f}")

        # Evaluation
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for i, (images, labels) in enumerate(tqdm(test_loader)):
                inputs = process_images(processor, images, device)
                texts = get_texts_for_labels(labels, label_to_text)
                text_inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)

                # # Print image and text pairs
                # print(f"Test Batch {i + 1}")
                # for img, text in zip(images, texts):
                #     print(f"Image: {img}, Text: {text}")

                image_features = model.get_image_features(**inputs)
                text_features = model.get_text_features(**text_inputs)

                logits_per_image = image_features @ text_features.T
                logits_per_text = text_features @ image_features.T

                ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
                image_loss = criterion(logits_per_image, ground_truth)
                text_loss = criterion(logits_per_text, ground_truth)
                loss = (image_loss + text_loss) / 2
                test_loss += loss.item()

                predicted = torch.argmax(logits_per_image, dim=1)
                total += ground_truth.size(0)
                correct += (predicted == ground_truth).sum().item()

        avg_test_loss = test_loss / len(test_loader)
        accuracy = correct / total
        accuracy_message = f"Epoch {epoch + 1} - Test Loss: {avg_test_loss:.3f}, Accuracy: {accuracy * 100:.2f}%"
        log_message(accuracy_message)

        with open(log_file, 'a') as log_test_file:
            log_test_file.write("\nTest Logs\n")
            log_test_file.write(
                f"Epoch {epoch + 1} - Test Loss: {avg_test_loss:.3f}, Accuracy: {accuracy * 100:.2f}%\n")

    # Save the model
    model.save_pretrained("./FoodCLIP101")
    processor.save_pretrained("./FoodCLIP101")


def main():
    downloader = Download()
    dataset_name = 'food-101'
    downloader.download_and_extract(dataset_name)
    log_message("Starting model training...")
    train_clip_model()


if __name__ == "__main__":
    main()
