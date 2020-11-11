import cv2
import numpy as np
from skimage import transform as trans


def preprocess_112(img,landmark=None):
  M = None
  image_size = [112,112]
  if landmark is not None:
    assert len(image_size)==2
    src = np.array([
      [30.2946, 51.6963],
      [65.5318, 51.5014],
      [48.0252, 71.7366],
      [33.5493, 92.3655],
      [62.7299, 92.2041] ], dtype=np.float32 )
    if image_size[1]==112:
      src[:,0] += 8.0
    dst = landmark.astype(np.float32)

    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2,:]

    warped = cv2.warpAffine(img,M,(image_size[1],image_size[0]), borderValue = 0.0)

    return warped

def preprocess_400(img, landmark=None, **kwargs):
  if isinstance(img, str):
    img = read_image(img, **kwargs)
  M = None
  image_size = [400, 400]
  if landmark is not None:
    assert len(image_size)==2
    src = np.array([
      [157.5944, 180],
      [242.4064, 180],
      [199.76024, 221.28209],
      [162.78812, 274.4534],
      [237.37758, 274.14816] ], dtype=np.float32 )
    dst = landmark.astype(np.float32)
    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2,:]
    warped = cv2.warpAffine(img,M,(image_size[1],image_size[0]), borderValue = 0.0)
    return warped



def preprocess_178(img,landmark=None):
  M = None
  image_size = [218,178]
  if landmark is not None:
    assert len(image_size)==2
    src = np.array([
      [70.808, 112.620],  # 'left_eye'
      [108.847, 112.430],  # 'right_eye
      [89.594, 133.812],  # 'nose'
      [73.842, 153.59],  # 'mouth_left'
      [105.3475, 153.7383] ], dtype=np.float32 )  # 'mouth_right'

    dst = landmark.astype(np.float32)

    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2,:]

    warped = cv2.warpAffine(img,M,(image_size[1],image_size[0]), borderValue = 0.0)

    print(warped.shape)


    return warped[20:-20,:]



def preprocess_256(img,landmark=None):
  M = None
  image_size = [256,256]
  if landmark is not None:
    assert len(image_size)==2
    src = np.array([
      [ 85.12117  , 83.337265],
      [172.01433  , 87.34686 ],
      [125.431725, 134.44124 ],
      [ 93.82528 , 178.23882 ],
      [153.64217 , 181.26077 ]], dtype=np.float32 )  # 'mouth_right'

    dst = landmark.astype(np.float32)

    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2,:]

    warped = cv2.warpAffine(img,M,(image_size[1],image_size[0]), borderValue = 0.0)

    return warped
