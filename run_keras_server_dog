from flask import Flask, jsonify, request
from keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import ResNet50
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import io

app = Flask(__name__)
model = None


#demo: 使用内置模型识别狗的图片,部署 在flask上,以api的形式调用
#
#



# load model,ResNet50内置模型,自带的训练好的模型
def load_model():
    global model
    model = ResNet50(weights="imagenet")


# preprocess input image
def preprocessiamge(image, target):

    if image.mode != 'RGB':
        image = image.covert('RGB')
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)
    return image


# predict image,and return to request client
@app.route('/predict', methods=['POST'])
def predict():
    #输出数组
    data = {}
    data['success'] = False
    data["predictions"] = []
    if request.method == 'POST':
        if request.files.get('image'):
            image = request.files['image'].read()
            image = Image.open(io.BytesIO(image))
            image = preprocessiamge(image,target=(224, 224))
            predicts = model.predict(image)
            results = imagenet_utils.decode_predictions(predicts)
            for (imageId,label,probability) in results[0]:
                 r = {"label":label,"pro":float(probability)}
                 data["predictions"].append(r)
            data['success'] = True
    return jsonify(data)


@app.route('/test_server')
def test_server():
    return jsonify({"data": "ok"})


def main():
    load_model()
    app.run(host='0.0.0.0', port='9998')


if __name__ == '__main__':
    main()
