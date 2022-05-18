"""
Hello World:
A small program to demonstrate how the app package is supposed to be organised.
The example uses image classification model that takes image as an input and
return a dictionary of scores.
"""
import requests

from PIL import Image
from transformers import pipeline, ViTFeatureExtractor, ViTForImageClassification

def run_model(image):
    """
    Example taken from: https://huggingface.co/google/vit-base-patch16-224#how-to-use
    """
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    
    generator = pipeline('image-classification', model=model, feature_extractor=feature_extractor)
    return generator(image)


def main(image):
    """
    Argument(s) name in the function signature are matching with the id in the
    app.config
    """
    # NOTE
    # We do this if we run this file on its own.
    # image = Image.open(image)
    # BUT!
    # When you are uploading, make sure to pass the argument directly. The app 
    # that will integrate your code will pass file object everytime.
    output = run_model(image)
    return output