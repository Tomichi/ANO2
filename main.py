import cv2
import numpy as np
import re
from pathlib import Path
from polygon import Polygon


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
    for polygon in parking_geometry:
        point1, point2, point3, point4 = map(list, polygon.points)
        source = np.array([list(point1), list(point2), list(point3), list(point4)], np.float32)
        destination = np.array([[0, 0], [output_size[0], 0], list(output_size), [0, output_size[1]]], np.float32)
        M, _ = cv2.findHomography(source, destination, 0)
        out_images = cv2.warpPerspective(src_image, M, output_size)
        output_images.append(out_images)

    return output_images


def train_parking(parking_geometry, image_size):
    output_images = []
    output_labels = []
    with open('train_images.txt', 'rt') as f:
        for path in f.readlines():
            full_image_path = str(Path(path.strip()).absolute())
            occupied_place = 1 if full_image_path.find('/full/') != -1 else 0
            parking_image = cv2.imread(full_image_path, 0)
            output_labels.extend([occupied_place] * len(parking_geometry))
            output_images.extend(extract_space(parking_image, parking_geometry, image_size))

    print(f'Train images: {len(output_images)}\n'
          f'Train labels: {len(output_labels)}')
    # todo train CNN

def test_parking(parking_geometry, image_size):
    test_labels = []
    test_images = []
    ground_labels = []
    geometry_size = len(parking_geometry)
    with open('out_prediction.txt', 'rt') as f:
        for line in f.readlines():
            clear_line = line.strip()
            ground_labels.append(int(clear_line))
    with open('test_images.txt', 'rt') as f:
        for path in f.readlines():
            full_image_path = str(Path(path.strip()).absolute())
            parking_image = cv2.imread(full_image_path, 0)
            parking_color_image = cv2.imread(full_image_path, 1)
            parking_chunk_images = extract_space(parking_image, parking_geometry, image_size)
            predict_labels = []
            for iteration in range(geometry_size):
                # todo call CNN
                predict = 0
                predict_labels.append(predict)
            parking = draw_circles(parking_color_image, parking_geometry, predict_labels)
            cv2.imshow("Img", parking)
            cv2.waitKey(0)
            test_images.extend(parking_chunk_images)
            test_labels.extend(predict_labels)
        evaluation(ground_labels, test_labels)


def draw_circles(image, geometry, predict):
    size = len(geometry)
    colors = [(0, 0, 255), (0,255, 0)]
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


def main():
    small_image_size = (80, 80)
    parking_geometry = load_parking_geometry("parking_map.txt")
    print("Train OpenCv start")
    train_parking(parking_geometry, small_image_size)
    print("Test OpenCv start")
    test_parking(parking_geometry, small_image_size)

main()