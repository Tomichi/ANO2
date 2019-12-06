import cv2
import numpy as np
import re
from pathlib import Path
from polygon import Polygon
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, AveragePooling2D, BatchNormalization
from keras import backend as K
from keras.models import model_from_json
from keras.losses import binary_crossentropy
from cnn_model import CNNModel
from sobel_model import SobelModel

def modify_images_for_CNN(image, size):
    if K.image_data_format() == 'channels_first':
        output_images = image.reshape(image.shape[0], 1, size[0], size[1])
        output_shape = (1, size[0], size[1])
        return output_images, output_shape
    else:
        output_images = image.reshape(image.shape[0], size[0], size[1], 1)
        output_shape = (size[0], size[1], 1)
        return output_images, output_shape


def load_parking_geometry(file):
    polygons = []
    with open(file, 'rt') as f:
        for line in f:
            x1, y1, x2, y2, x3, y3, x4, y4 = map(int, re.sub('[,;]', ' ', line).strip(). split())
            p = Polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])
            polygons.append(p)
    return polygons


def extract_space(src_image, parking_geometry, crop_image_size, for_neural):
    output_images = []
    output_size = (int(crop_image_size[0]/2), int(crop_image_size[1]/2))
    for polygon in parking_geometry:
        point1, point2, point3, point4 = map(list, polygon.points)
        source = np.array([list(point1), list(point2), list(point3), list(point4)], np.float32)
        destination = np.array([[0, 0], [crop_image_size[0], 0], list(crop_image_size), [0, crop_image_size[1]]], np.float32)
        M, _ = cv2.findHomography(source, destination, 0)
        out_image = cv2.warpPerspective(src_image, M, crop_image_size)
        if for_neural:
            out_image = cv2.GaussianBlur(out_image, (3,3), 1, 0)
        out = cv2.resize(out_image, (output_size), interpolation=cv2.INTER_AREA)
        # Sobel filter
        Gx = cv2.Sobel(out,cv2.CV_32F, 1, 0, 3)
        Gx = cv2.convertScaleAbs(Gx)
        Gy = cv2.Sobel(out,cv2.CV_32F, 0, 1, 3)
        Gy = cv2.convertScaleAbs(Gy)
        G = cv2.addWeighted(Gx, 0.5, Gy, 0.5, 0)
        # Gaussian Blur
        if for_neural:
            G = cv2.GaussianBlur(G, (5, 5), 1, 0)
        output_images.append(G)
    return output_images, output_size


def train_parking(parking_geometry, image_size):
    images = []
    labels = []
    with open('train_images.txt', 'rt') as f:
        for line in f:
            full_image_path = str(Path(line.strip()).absolute())
            occupied_place = 1 if full_image_path.find('/full/') != -1 else 0
            parking_image = cv2.imread(full_image_path, 0)
            labels.extend([occupied_place] * len(parking_geometry))
            extract_images, _ = extract_space(parking_image, parking_geometry, image_size, True)
            images.extend(extract_images)

    print(f'Train images: {len(images)}\n'
          f'Train labels: {len(labels)}')

    output_labels = np.array(labels)
    output_images = np.vstack([images])
    # normalization
    output_images = output_images * 1/255
    return output_images, output_labels


def test_parking_for_neural(parking_geometry, image_size):
    images = []
    labels = []
    with open('groundtruth.txt', 'rt') as f:
        for line in f:
            clear_line = line.strip()
            labels.append(int(clear_line))
    with open('test_images.txt', 'rt') as f:
        for line in f:
            full_image_path = str(Path(line.strip()).absolute())
            parking_image = cv2.imread(full_image_path, 0)
            parking_chunk_images, _ = extract_space(parking_image, parking_geometry, image_size, True)
            images.extend(parking_chunk_images)
    out_images = np.vstack([images])
    out_images = out_images * 1/255
    out_labels = np.array(labels)
    return out_images, out_labels


def test_parking(parking_geometry, image_size, model):
    test_labels = []
    labels = []
    with open('groundtruth.txt', 'rt') as f:
        for line in f:
            clear_line = line.strip()
            labels.append(int(clear_line))
    with open('test_images.txt', 'rt') as f:
        for path in f:
            full_image_path = str(Path(path.strip()).absolute())
            parking_image = cv2.imread(full_image_path, 0)
            parking_color_image = cv2.imread(full_image_path, 1)
            for_neural = str(model) == "CNN Model"
            parking_chunk_images, new_image_size = extract_space(parking_image, parking_geometry, image_size, for_neural)
            out_images = np.vstack([parking_chunk_images])
            out_images = out_images * 1 / 255
            out_images, _ = modify_images_for_CNN(out_images, new_image_size)
            predict_labels = []
            for image in out_images:
                new_image = image.reshape(1, new_image_size[0], new_image_size[1], 1)
                value = model.predict(new_image)
                predict_labels.append(value)
            parking = draw_circles(parking_color_image, parking_geometry, predict_labels)
            cv2.imshow("Img", parking)
            test_labels.extend(predict_labels)
            cv2.waitKey(0)
    evaluation(labels, test_labels)


def draw_circles(image, geometry, predict):
    size = len(geometry)
    colors = [(0,255, 0), (0, 0, 255)]
    for i in range(size):
        point1, point2, point3, point4 = map(list, geometry[i].points)
        x_center = int((point1[0] + point3[0]) / 2)
        y_center = int((point1[1] + point3[1]) / 2)
        image = cv2.circle(image, (x_center, y_center), 5, colors[predict[i]], -1)
    return image


def evaluation(detect_output, ground_output):
    samples_len = len(detect_output)
    result = [0] * 4
    for i in range(samples_len):
        result[ground_output[i] * 2 + detect_output[i]] += 1
    accuracy = (result[0] + result[3]) / samples_len
    print(f"True negative: {result[0]}\n"
          f"True positive: {result[3]}\n"
          f"False positive: {result[1]}\n"
          f"False negative: {result[2]}\n"
          f"Accuracy: {accuracy}"
          )


def init_CNN(train_images, train_labels, test_images, test_labels, image_size, dense1, dense2):
    train_images, input_shape = modify_images_for_CNN(train_images, image_size)
    test_images, _ = modify_images_for_CNN(test_images, image_size)
    model = Sequential()
    model.add(Conv2D(5, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(12, kernel_size=(3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3, 3)))
    model.add(Flatten())
    model.add(Dense(dense1, activation='relu'))
    model.add(Dense(109, activation='relu'))
    model.add(Dropout(0.05))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss=binary_crossentropy, optimizer="Adam", metrics=['accuracy'])
    model.fit(train_images, train_labels, batch_size=600, epochs=20, verbose=0,
              validation_data=(test_images, test_labels))
    score = model.evaluate(test_images, test_labels, verbose=0)
    #model.summary()

    return model, score


def save_model(model, accuracy):
    model_json_file = 'models/model' + str(accuracy) + '.json'
    model_hdf5_file = 'models/model' + str(accuracy) + '.h5'
    print(f" file saved json {model_json_file} file hdf5 {model_hdf5_file}")
    with open(model_json_file, 'w') as json_file:
        json_file.write(model.to_json())
    model.save_weights(model_hdf5_file)
    print("saved")


def load_model(file_model_path, file_weights_path):
    model_file  = open(file_model_path, 'r')
    model_json = model_file.read()
    model_file.close()
    model =  model_from_json(model_json)
    model.load_weights(file_weights_path)
    model.compile(loss=binary_crossentropy, optimizer="Adam", metrics=['accuracy'])
    return model


def main():
    small_image_size = (80, 80)
    half_image_size = (40, 40)
    parking_geometry = load_parking_geometry("parking_map.txt")
    print("Preprocessing train images - OpenCv start")
    train_images, train_labels = train_parking(parking_geometry, small_image_size)
    print("Preprocessing test images - OpenCv start")
    test_images, test_labels = test_parking_for_neural(parking_geometry, small_image_size)
    # training
    #for i in range (20):
    #    for j in range(4):
    #        print(f" dense-1 {40+i*10}, dense-2 {j*5+5}")
    #        model, score = init_CNN(train_images, train_labels, test_images, test_labels, half_image_size, 50+i*5, j*5+5)
    #        save_model(model, score[1])
    #        print(score[1])

    print("CNN")
    model = load_model('models/model0.9895833134651184.json', 'models/model0.9895833134651184.h5')
    cnn_model = CNNModel('CNN Model', model)
    test_parking(parking_geometry, small_image_size, cnn_model)

    print("SOBEL")
    sobel_mode = SobelModel('Sobel Model', 216)
    test_parking(parking_geometry, small_image_size, sobel_mode)

if __name__=="__main__":
    main()
