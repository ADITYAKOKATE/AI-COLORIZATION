import numpy as np
import cv2
import os

class Colorizer:
    def __init__(self, height=480, width=600):
        self.height, self.width = height, width
        model_dir = "model"
        prototxt_path = os.path.join(model_dir, "colorization_deploy_v2.prototxt")
        caffe_model_path = os.path.join(model_dir, "colorization_release_v2.caffemodel")

        if not os.path.isfile(prototxt_path) or not os.path.isfile(caffe_model_path):
            raise FileNotFoundError("Prototxt or Caffe model file not found.")

        self.colorModel = cv2.dnn.readNetFromCaffe(prototxt_path, caffeModel=caffe_model_path)

        clusterCenters = np.load("model/pts_in_hull.npy")
        clusterCenters = clusterCenters.transpose().reshape(2, 313, 1, 1)

        self.colorModel.getLayer(self.colorModel.getLayerId('class8_ab')).blobs = [clusterCenters.astype(np.float32)]
        self.colorModel.getLayer(self.colorModel.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, dtype=np.float32)]

    def processImage(self, imgName):
        self.img = cv2.imread(imgName)

        if self.img is None or self.img.size == 0:
            print("Error: Image not loaded or empty.")
            return

        self.img = cv2.resize(self.img, (self.width, self.height))
        self.processFrame()
        output_path = os.path.join("output", os.path.basename(imgName))
        cv2.imwrite(output_path, self.imgFinal)

        cv2.imshow("Output", self.imgFinal)
        cv2.waitKey(0)  # Wait indefinitely for a key press
        cv2.destroyAllWindows()  # Close windows

    def processFrame(self):
        imgNormalized = (self.img[:, :, [2, 1, 0]] * 1.0 / 255).astype(np.float32)

        imgLab = cv2.cvtColor(imgNormalized, cv2.COLOR_RGB2LAB)
        channelL = imgLab[:, :, 0]

        imgLabResized = cv2.cvtColor(cv2.resize(imgNormalized, (224, 224)), cv2.COLOR_RGB2Lab)
        channelLResized = imgLabResized[:, :, 0]
        channelLResized -= 50

        self.colorModel.setInput(cv2.dnn.blobFromImage(channelLResized))
        result = self.colorModel.forward()[0, :, :, :].transpose((1, 2, 0))

        resultResized = cv2.resize(result, (self.width, self.height))

        self.imgOut = np.concatenate((channelL[:, :, np.newaxis], resultResized), axis=2)
        self.imgOut = np.clip(cv2.cvtColor(self.imgOut, cv2.COLOR_LAB2BGR), 0, 1)
        self.imgOut = (self.imgOut * 255).astype(np.uint8)

        self.imgFinal = np.hstack((self.img, self.imgOut))


