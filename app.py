import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from PIL import Image
import keras
from huggingface_hub import from_pretrained_keras
# from io import BytesIO
import imageio
import pathlib

model = from_pretrained_keras("keras-io/lowlight-enhance-mirnet", compile=False)

def run_model(image):
    
    output_path = os.path.join('static/images')
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True) 
    image = Image.open(image)
    image = image.resize((960,640))
    image = keras.preprocessing.image.img_to_array(image)
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    output = model.predict(image)
    output_image = output[0] * 255.0
    output_image = output_image.clip(0, 255)
    output_image = output_image.reshape(
        (np.shape(output_image)[0], np.shape(output_image)[1], 3)
    )
    output_image = np.uint32(output_image)
    output_image = output_image.astype(np.uint8)
    output_file = os.path.join(output_path, 'output.jpeg')
    imageio.imwrite(output_file, output_image)
    return output_file

def main(input_image):
    """
    Argument(s) name in the function signature are matching with the id in the
    app.config
    """
    output = run_model(input_image)
    return output

if __name__=="__main__":
    main('download.png')
