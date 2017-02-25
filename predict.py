import numpy as np
import matplotlib.pyplot as plt
import cv2

from utils import palette

from deeplabV2 import DeeplabV2


# predicts an image, with the cropping policy of deeplab (single scale for simplicity)
def predict(img, model, crop_size):

    img = img.astype(np.float32)
    h, w, c = img.shape
    c_h, c_w = crop_size

    assert (c_h >= 500 and c_w >= 500), 'Crop size should be greater than 500 for VOC12.'

    pad_height = max(c_h - h, 0)
    pad_width = max(c_w - w, 0)

    x = cv2.copyMakeBorder(src=img, top=0, bottom=pad_height, left=0, right=pad_width,
                           borderType=cv2.BORDER_CONSTANT, value=np.array([104.008, 116.669, 122.675]))

    x[:, :, 0] -= 104.008
    x[:, :, 1] -= 116.669
    x[:, :, 2] -= 122.675

    x_batch = np.expand_dims(x.transpose(2, 0, 1), axis=0)

    prob = model.predict(x_batch)[0]  # remove batch dimension

    prob = prob[:, 0:h, 0:w]
    pred = np.argmax(prob, axis=0)

    return pred


if __name__ == '__main__':
    model = DeeplabV2(input_shape=(3, 512, 512), apply_softmax=False)
    model.summary()

    # predict image
    img = cv2.imread('imgs_deeplabv2/2007_000129.jpg')
    pred = predict(img=img, model=model, crop_size=(512, 512))

    # convert prediction to color
    pred_image = palette[pred.ravel()].reshape(img.shape)

    # visualize results
    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    imgplot = plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    a.set_title('Image')
    a = fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(pred_image)
    a.set_title('Segmentation')
    plt.show(fig)
