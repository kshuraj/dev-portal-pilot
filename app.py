import os
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image


def run_model(image):

    image = Image.open(image)

    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    # model predicts one of the 1000 ImageNet classes
    predicted_class_idx = logits.argmax(-1).item()

    return model.config.id2label[predicted_class_idx]

def main(input_image):
    """
    Argument(s) name in the function signature are matching with the id in the
    app.config
    """
    output = run_model(input_image)
    print(output)
    return output

if __name__=="__main__":
    main('download.jpg')
