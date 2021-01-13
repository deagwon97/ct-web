import numpy as np
import pydicom
import cv2
import torch
import segmentation_models_pytorch as smp
import sys
import matplotlib.pyplot as plt

margin = 70
device = torch.device('cpu')

def load_image(img_path):
    image = pydicom.dcmread(f"{img_path}").pixel_array
    image = np.array(image)
    return image
    
def preprocessing(image, margin = margin):
    # set margin
    margin = margin
    image = image[margin:-margin,margin:-margin]
    # upper, lower threshold
    under_val = 700
    over_val = 1200
    over_idx  = (image > over_val)
    under_idx = (image < under_val)
    image[over_idx]  = over_val
    image[under_idx] = under_val
    image = (image - under_val) / (over_val - under_val) * 255
    image = image.astype(np.uint8)[..., np.newaxis]
    # resize image to (512,512,1)
    image = cv2.resize(image, (512, 512))[...,np.newaxis]
    return image
    # to tensor
    
    return image_tensor

def predict(image, model):
    image = image.transpose(2, 0, 1).astype('float32')[np.newaxis,...]
    image_tensor = torch.FloatTensor(image).to(device)
    predict_mask = model.predict(image_tensor).numpy()[0,:,:,:]
    predict_mask = predict_mask.transpose([1,2,0]).argmax(axis = 2)
    return predict_mask


if __name__ == "__main__":
    # input path
    if sys.argv[1] == '-img_path':
        img_path = sys.argv[2]
    if sys.argv[3] == '-model_path':
        model_path = sys.argv[4]
    if sys.argv[5] == '-save_path':
        save_path = sys.argv[6]
    with torch.no_grad():
        #set path, model
        model = torch.load(model_path).to(device)
        # load image
        image = load_image(img_path)
        # preprocessing
        image_tensor = preprocessing(image)
        # predict
        predict_mask = predict(image_tensor, model)
        predict_mask = (predict_mask * 255 / 3).astype(np.uint8)
        cv2.imwrite(save_path + 'input.png', image_tensor)
        cv2.imwrite(save_path + 'output.png', predict_mask)
    
        print((predict_mask == 0).sum() / len(predict_mask.reshape(-1)) * 100)
        print((predict_mask == 1* 255/ 3).sum() / len(predict_mask.reshape(-1)) * 100)
        print((predict_mask == 2* 255/ 3).sum() / len(predict_mask.reshape(-1)) * 100)
        print((predict_mask == 3* 255/ 3).sum() / len(predict_mask.reshape(-1)) * 100)

