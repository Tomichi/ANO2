import cv2
import numpy as np
import re
from pathlib import Path
from polygon import Polygon
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, AveragePooling2D
from keras import backend as K
from keras.losses import binary_crossentropy


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
        for line in f.readlines():
            x1, y1, x2, y2, x3, y3, x4, y4 = map(int, re.sub('[,;]', ' ', line).strip(). split())
            p = Polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])
            polygons.append(p)
    return polygons


def extract_space(src_image, parking_geometry, output_size):
    output_images = []
    #cv2.equalizeHist(src_image, src_image)
    for polygon in parking_geometry:
        point1, point2, point3, point4 = map(list, polygon.points)
        source = np.array([list(point1), list(point2), list(point3), list(point4)], np.float32)
        destination = np.array([[0, 0], [output_size[0], 0], list(output_size), [0, output_size[1]]], np.float32)
        M, _ = cv2.findHomography(source, destination, 0)
        out_image = cv2.warpPerspective(src_image, M, output_size)
        blur = cv2.medianBlur(out_image, 7)
        #sobel
        Gx = cv2.Sobel(blur,cv2.CV_32F, 1, 0, 3)
        Gx = cv2.convertScaleAbs(Gx)
        Gy = cv2.Sobel(blur,cv2.CV_32F, 0, 1, 3)
        Gy = cv2.convertScaleAbs(Gy)
        G = cv2.addWeighted(Gx, 0.5, Gy, 0.5, 0)
        output_images.append(G)
    return output_images


def train_parking(parking_geometry, image_size):
    images = []
    labels = []
    with open('train_images.txt', 'rt') as f:
        for path in f.readlines():
            full_image_path = str(Path(path.strip()).absolute())
            occupied_place = 1 if full_image_path.find('/full/') != -1 else 0
            parking_image = cv2.imread(full_image_path, 0)
            labels.extend([occupied_place] * len(parking_geometry))
            images.extend(extract_space(parking_image, parking_geometry, image_size))

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
    with open('out_prediction.txt', 'rt') as f:
        for line in f.readlines():
            clear_line = line.strip()
            labels.append(int(clear_line))
    with open('test_images.txt', 'rt') as f:
        for path in f.readlines():
            full_image_path = str(Path(path.strip()).absolute())
            parking_image = cv2.imread(full_image_path, 0)
            parking_chunk_images = extract_space(parking_image, parking_geometry, image_size)
            images.extend(parking_chunk_images)
    out_images = np.vstack([images])
    out_images = out_images * 1/255
    out_labels = np.array(labels)
    return out_images, out_labels


def test_parking(parking_geometry, image_size, model):
    test_labels = []
    labels = []
    geometry_size = len(parking_geometry)
    with open('out_prediction.txt', 'rt') as f:
        for line in f.readlines():
            clear_line = line.strip()
            labels.append(int(clear_line))
    with open('test_images.txt', 'rt') as f:
        for path in f.readlines():
            full_image_path = str(Path(path.strip()).absolute())
            parking_image = cv2.imread(full_image_path, 0)
            parking_color_image = cv2.imread(full_image_path, 1)
            parking_chunk_images = extract_space(parking_image, parking_geometry, image_size)
            out_images = np.vstack([parking_chunk_images])
            out_images = out_images * 1 / 255
            out_images, _ = modify_images_for_CNN(out_images, image_size)
            predicts = model.predict(out_images)
            predict_labels = []
            for iteration in range(geometry_size):
                value = 1 if predicts[iteration][0] > 0.7 else 0
                predict_labels.append(value)
            parking = draw_circles(parking_color_image, parking_geometry, predict_labels)
            cv2.imshow("Img", parking)
            cv2.waitKey(0)
        #   images.extend(parking_chunk_images)
            test_labels.extend(predict_labels)
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


def init_CNN(train_images, train_labels, test_images, test_labels, image_size):
    train_images, input_shape = modify_images_for_CNN(train_images, image_size)
    test_images, _ = modify_images_for_CNN(test_images, image_size)

    model = Sequential()
    model.add(Conv2D(6, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.15))
    model.add(Flatten())
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss=binary_crossentropy, optimizer="adadelta", metrics=['accuracy'])
    model.fit(train_images, train_labels, batch_size=300, epochs=6, verbose=1,
              validation_data=(test_images, test_labels))
    score = model.evaluate(test_images, test_labels, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model.summary()

    return model

def main():
    small_image_size = (80, 80)
    parking_geometry = load_parking_geometry("parking_map.txt")
    print("Preprocessing train images - OpenCv start")
    train_images, train_labels = train_parking(parking_geometry, small_image_size)
    print("Preprocessing test images - OpenCv start")
    test_images, test_labels = test_parking_for_neural(parking_geometry, small_image_size)

    #CNN BEGIN
    model = init_CNN(train_images, train_labels, test_images, test_labels, small_image_size)
    print("Test OpenCv start")
    test_parking(parking_geometry, small_image_size, model)
    #test_parking(parking_geometry, small_image_size)

main()