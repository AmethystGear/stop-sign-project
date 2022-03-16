import cv2
import os
import numpy as np
import math
import glob
import keras
import random

x_train = []
y_train = []
for _, dir, _ in os.walk("data/train"):
    for dir in dir:
        for _, _, files in os.walk("data/train/" + dir):
            for file in files:
                img = cv2.imread("data/train/" + dir + "/" + file)
                classification = int(dir)
                x_train.append(img)
                y_train.append(classification)

temp = list(zip(x_train, y_train))
random.shuffle(temp)
x_train, y_train = zip(*temp)

x_train = np.stack(list(x_train), axis=0)
y_train = np.array(list(y_train))

x_train = x_train.astype('float32')
x_train /= 255

THRESHOLD = 0.8
def get_model():
    while(True):
        model = keras.models.Sequential([
            keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(100, 100, 3)),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Conv2D(32, (3,3), activation='relu'),
            keras.layers.MaxPooling2D(2,2),
            keras.layers.Conv2D(64, (3,3), activation='relu'),
            keras.layers.MaxPooling2D(2,2),
            keras.layers.Conv2D(64, (3,3), activation='relu'),
            keras.layers.MaxPooling2D(2,2),
            keras.layers.Conv2D(64, (3,3), activation='relu'),
            keras.layers.MaxPooling2D(2,2),
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(loss = "binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        hist = model.fit(x_train, y_train, batch_size = 4, epochs=10)
        if hist.history['accuracy'][-1] >= THRESHOLD:
            return model

if os.path.exists('model'):
    model = keras.models.load_model('model')
else:   
    model = get_model()
    model.save('model')

MIN_SCALE = 0.25
CONFIDENCE_THRESHOLD = 0.9

class Sign:
    def __init__(self, reference_img, reference_pts):
        self.reference_img = reference_img
        self.reference_pts = reference_pts

def find_highest_point(points, ignore_set):
    highest = None
    maximum = -math.inf
    for point in points:
        if (point[0][0], point[0][1]) in ignore_set:
            continue

        if point[0][1] > maximum:
            highest = point[0]
            maximum = point[0][1]
    
    if highest is not None:
        ignore_set.add((highest[0], highest[1]))
        return (ignore_set, highest)
    else:
        return (ignore_set, next(iter(ignore_set)))

def find_lowest_point(points, ignore_set):
    lowest = None
    minimum = math.inf
    for point in points:
        if (point[0][0], point[0][1]) in ignore_set:
            continue

        if point[0][1] < minimum:
            lowest = point[0]
            minimum = point[0][1]

    if lowest is not None:
        ignore_set.add((lowest[0], lowest[1]))
        return (ignore_set, lowest)
    else:
        return (ignore_set, next(iter(ignore_set)))

def find_polygon_hi_low_points(polygon):
    points = []
    ignore_set, p1 = find_lowest_point(polygon, set())
    ignore_set, p2 = find_lowest_point(polygon, ignore_set)
    if p1[0] < p2[0]:
        points += [p1, p2]
    else:
        points += [p2, p1]

    ignore_set, p3 = find_highest_point(polygon, ignore_set)
    ignore_set, p4 = find_highest_point(polygon, ignore_set)
    if p3[0] < p4[0]:
        points += [p3, p4]
    else:
        points += [p4, p3]

    return points

def run_canny(im, blur, canny_thresh_1, canny_thresh_2):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if blur != 0.0:
        im = cv2.GaussianBlur(im, (0, 0), blur)
    return cv2.Canny(im, canny_thresh_1, canny_thresh_2)

def get_potential_polygons(image_file, blur, canny_thresh_1, canny_thresh_2, proportion_arclen):
    img = cv2.imread(image_file)
    fname = os.path.splitext(image_file)[0]
    fname = os.path.basename(fname)

    im = run_canny(img, blur, canny_thresh_1, canny_thresh_2)
    cv2.imwrite(fname + '_canny.png', im)

    contours, _ = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    img_cpy = np.zeros_like(img)
    polygons = []
    for con in contours:
        approx = cv2.approxPolyDP(con, proportion_arclen * cv2.arcLength(con, True), True)
        cv2.drawContours(img_cpy, [approx],  -1, (255 , 255 , 255), thickness=-1)
        # close every contour
        np.append(approx, approx[0])
        polygons.append(approx)

    return (img, fname, polygons)

def get_sign(image):
    reference_img, _, polygons = get_potential_polygons(image, 0.0, 100, 200, 0.01)
    reference_pts = np.asarray(find_polygon_hi_low_points(polygons[0]), dtype=np.float32)
    return Sign(reference_img, reference_pts)

def detect_signs(image, sign_reference):
    reference_img = sign_reference.reference_img
    reference_pts = sign_reference.reference_pts

    cv2.imwrite('ref_img.png', reference_img)
    img, fname, polygons = get_potential_polygons(image, 0.5, 100, 200, 0.01)
    fname = os.path.basename(fname)
    img_copy = img.copy()

    for i in range(len(polygons)):
        polygon = polygons[i]
        mask = np.zeros_like(img)

        if cv2.contourArea(polygon) < reference_img.shape[0] * reference_img.shape[1] * MIN_SCALE:
            continue

        cv2.drawContours(mask, [polygon],  -1, (255 , 255 , 255), thickness=-1)
        cv2.drawContours(mask, [polygon],  -1, (0 , 0 , 0), thickness=3)

        img2gray = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        bg_removed = cv2.bitwise_and(img,img,mask=mask)

        cv2.imwrite(fname + '_bg_removed_' + str(i) + '.png', bg_removed)

        pts = np.asarray(find_polygon_hi_low_points(polygon), dtype=np.float32)
        
        warped = cv2.warpPerspective(bg_removed, cv2.getPerspectiveTransform(pts, reference_pts), (reference_img.shape[1], reference_img.shape[0]))

        if model.predict(np.stack([warped], axis=0)) > CONFIDENCE_THRESHOLD:
            cv2.imwrite(fname + '_warped_' + str(i) + '.png', warped)
            cv2.drawContours(img_copy, [polygon], 0, (0 , 255 , 0), thickness=2)

    return img_copy


STOP_SIGN = get_sign("stop_sign_ref.png")
for img in glob.glob('in/*'):
    print(img)
    stop_signs_highlighted = detect_signs(img, STOP_SIGN)
    cv2.imwrite('out/' + os.path.basename(img), stop_signs_highlighted)



    




