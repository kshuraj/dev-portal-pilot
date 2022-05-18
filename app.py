"""
Hello World:
A small program to demonstrate how the app package is supposed to be organised.
Here we've used 'gpt2' model from HuggingFace to generate text for user's
input.
"""
from transformers import pipeline
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image

# from PIL import Image

def run_model(image):
    """
    Example taken from: https://huggingface.co/google/vit-base-patch16-224
    """
    image = Image.open(image)
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    # model predicts one of the 1000 ImageNet classes
    predicted_class_idx = logits.argmax(-1).item()
    
    return ("Predicted class:", model.config.id2label[predicted_class_idx])

def main(input_image):
    """
    Argument(s) name in the function signature are matching with the id in the
    app.config
    """
    output = run_model(input_image)
    return output