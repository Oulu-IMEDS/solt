import cv2

allowed_paddings = {'z': cv2.BORDER_CONSTANT, 'r': cv2.BORDER_REFLECT}
allowed_interpolations = {'bilinear': cv2.INTER_LINEAR_EXACT, 'bicubic': cv2.INTER_CUBIC, 'nearest': cv2.INTER_NEAREST}
allowed_crops = {'c', 'r'}
allowed_noise_types = {'gaussian', 's&p'}
