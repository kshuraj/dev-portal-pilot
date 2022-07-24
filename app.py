from transformers import pipeline,DetrFeatureExtractor, DetrForSegmentation

def run_model(image):
    feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-50-panoptic')
    model = DetrForSegmentation.from_pretrained('facebook/detr-resnet-50-panoptic')

    pipe = pipeline('image-segmentation',model=model,feature_extractor=feature_extractor)
    output = pipe(image)
    my_dict = []
    for out in output:
        adds = {'score':out["score"],'label':out["label"]}
        my_dict.append(adds)
    return my_dict

def main(input_image):
    """
    Argument(s) name in the function signature are matching with the id in the
    app.config
    """
    output = run_model(input_image)
    print(output)
    return output

if __name__=="__main__":
    main('dev-portal-pilot/download.jpg')

