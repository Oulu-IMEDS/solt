import cv2

allowed_paddings = ['z', 'r']
allowed_interpolations = {'bilinear': cv2.INTER_LINEAR_EXACT, 'bicubic': cv2.INTER_CUBIC, 'nearest': cv2.INTER_NEAREST}