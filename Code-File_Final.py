# """
# It detects anomalies using an autoencoder model.

# It includes functionality to convert a video into individual image frames,
# preprocess those frames by resizing and normalizing, and train an autoencoder
# to identify anomalies based on reconstruction errors. The script also organizes
# project directories, visualizes reconstruction losses, and saves the trained
# model for future use.
# """
# import matplotlib.pyplot as plt
# import os
# import cv2
# from PIL import Image
# from glob import glob
# import numpy as np
# from tensorflow.keras.models import load_model
# from keras.models import Sequential
# from keras.layers import Dense, Input, Reshape


# def create_project_dirs():
#     """
#     Create essential project directories.

#     Specifically, it ensures the existence of the following directories:
#     'data/videos', 'data/images', 'models', and 'src'. If any of these
#     directories already exist, they are left unchanged. This setup helps
#     maintain a structured workflow for managing data and models during
#     the project.
#     """
#     directories = [
#         'data/videos',
#         'data/images',
#         'models',
#         'src'
#     ]
#     for directory in directories:
#         os.makedirs(directory, exist_ok=True)
#     print("Directories created successfully.")


# create_project_dirs()


# def convert_video_to_images(img_folder, filename):
#     """
#     Convert the video file (assignment3_video.avi) to JPEG images.

#     Once the video has been converted to images, then this function doesn't
#     need to be run again.
#     """
#     if not os.path.exists(img_folder):
#         os.makedirs(img_folder)
#     img_folder = f'{img_folder.rstrip(os.path.sep)}{os.path.sep}'

#     # Open the video file
#     video = cv2.VideoCapture(filename)
#     if not video.isOpened():
#         print("Error opening video file")
#         return

#     i = 0
#     # Process each frame
#     while video.isOpened():
#         ret, frame = video.read()
#         if ret:
#             im_fname = f'{img_folder}frame{i:0>4}.jpg'
#             cv2.imwrite(im_fname, frame)
#             print(f'Captured {im_fname}')
#             i += 1
#         else:
#             break
#     video.release()
#     cv2.destroyAllWindows()

#     if i > 0:
#         print(f'Video converted: {i} images written to {img_folder}')


# video_path = "assignment3_video.avi"  # Adjust path as necessary
# image_path = 'data/images'
# convert_video_to_images(image_path, video_path)


# def load_images(img_dir, im_width=60, im_height=44):
#     """
#     Read, resize, and normalize the extracted image frames from a folder.

#     The images are returned both as a Numpy array of flattened images
#     (i.e. the images with the 3-d shape (im_width, im_height, num_channels)
#     are reshaped into the 1-d shape (im_width x im_height x num_channels))
#     and a list with the images with their original number of dimensions
#     suitable for display.
#     """
#     img_dir = f'{img_dir.rstrip(os.path.sep)}{os.path.sep}'
#     images = []
#     fnames = glob(f'{img_dir}frame*.jpg')
#     fnames.sort()
#     for fname in fnames:
#         im = Image.open(fname)
#         im_array = np.array(im.resize((im_width, im_height)))
#         images.append(im_array.astype(np.float32) / 255.)
#         im.close()
#     X = np.array(images).reshape(-1, np.prod(images[0].shape))
#     return X, images


# image_path = 'data/images'


# X, images = load_images(image_path)


# def build_autoencoder(input_shape):
#     """
#     Build and compile an autoencoder model for anomaly detection tasks.

#     The model is constructed using the Keras Sequential API, with an
#     architecture that includes an input layer matching the specified
#     input shape, three fully connected dense layers with ReLU activation
#     to learn compressed representations, and an output layer with a sigmoid
#     activation to reconstruct the input data. A reshape layer is used to
#     restore the original input dimensions. The model is compiled with the
#     Adam optimizer and mean squared error (MSE) loss function, making it
#     suitable for reconstruction-based learning tasks. The function returns
#     a compiled autoencoder model ready for training.
#     """
#     model = Sequential([
#         Input(shape=input_shape),
#         Dense(128, activation='relu'),
#         Dense(64, activation='relu'),
#         Dense(128, activation='relu'),
#         Dense(np.prod(input_shape), activation='sigmoid'),
#         Reshape(input_shape)  # Using flattened data for training
#     ])
#     model.compile(optimizer='adam', loss='mse')
#     return model


# # Assuming X is already flattened
# input_shape = (7920,)  # This should match the flattened shape
# autoencoder = build_autoencoder(input_shape)
# autoencoder.fit(X, X, epochs=50, batch_size=32)


# def calculate_losses(autoencoder, X):
#     """
#     Calculate reconstruction losses for input data using a trained autoencoder.

#     This function computes the mean squared error (MSE) between the original
#     input data and its reconstructed version produced by the autoencoder.
#     The losses represent how well the autoencoder is able to reconstruct the
#     input, where higher losses may indicate anomalies.
#     """
#     reconstructed = autoencoder.predict(X)
#     losses = np.mean(np.square(X - reconstructed), axis=1)
#     return losses


# plt.plot(calculate_losses(autoencoder, X))


# plt.show()


# threshold = 0.001


# autoencoder.save('autoencoder_model.keras', save_format='keras')


autoencoder = load_model("autoencoder_model.keras")


def predict(frame):
    """
    Argument.

    frame : Video frame with shape == (44, 60, 3) and dtype == float.
    Return
    anomaly : A boolean indicating whether the frame is an anomaly or not.
    ------
    """
    threshold = 0.001
    frame = frame.reshape(-1, np.prod(frame.shape))
    loss = autoencoder.evaluate(frame, frame, verbose=0)
    # Your fancy computations here!!
    return loss > threshold
