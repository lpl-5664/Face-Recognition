import os
import sys
import cv2
import argparse
import numpy as np
from scipy import spatial
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow_addons.losses import TripletSemiHardLoss

def get_mean_features(person_folder):
    facevecs = None
    for filename in os.listdir(person_folder):
        image = person_folder+filename
        #load image from the path
        image = cv2.imread(image)
        image = np.expand_dims(image, axis=0)
        facevec = facenet.predict(image)
        #facevec = np.expand_dims(facevec, axis=0)
        if facevecs is None:
            facevecs = facevec
        else:
            facevecs = np.append(facevecs, facevec, axis=0)
    return np.array(facevecs).sum(axis=0)/len(facevecs)

def build_database(data_path):
    database = []
    for subdir in os.listdir(data_path):
        person_folder = data_path+subdir+'/'
        #skip any possbile files
        if not os.path.isdir(person_folder):
            continue
        mean_feat = get_mean_features(person_folder)
        database.append({'name': subdir, 'features': mean_feat})
    return database

def get_identity(features, database, threshold=7.2103):
    distances = []
    for person in database:
        distance = spatial.distance.euclidean(person.get('features'), features)
        distances.append(distance)
    min_dist = min(distances)
    min_dist_index = distances.index(min_dist)
    print(min_dist, distances)
    if min_dist < threshold:
        return database[min_dist_index].get('name')
    else:
        return 'unidentified'

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-tr", "--train", type=str, help='a path to the train images data')
    parser.add_argument("-ts", "--test", type=str, help='a path to the test image data')
    parser.add_argument("-ti", "--testimage", type=str, help='a path to a test image')
    parser.add_argument("-m", "--model", type=str, help='a path to the txt data file')

    args = parser.parse_args()

    train_path = args.train
    test_path = args.test
    test_image = args.testimage
    model_path = args.model

    # Check if the input for command lines is correct
    if not os.path.isdir(train_path):
        print("Directory for train data does not exist, please check again.")
        sys.exit(1)
    if not os.path.isfile(model_path):
        print("File containing model does not exist, please check again.")
        sys.exit(1)

    # Prepare data
    gen = ImageDataGenerator(rescale=1./255)
    data = gen.flow_from_directory(train_path, batch_size=16, class_mode='sparse')

    # Prepare pretrained model
    facenet = load_model(model_path)
    facenet.compile(optimizer='sgd', loss=TripletSemiHardLoss(margin = 5.0), metrics=['accuracy'])
    facenet.fit(data, epochs=5)         # Re-train the model with new data

    # Evaluate the model
    test_data = gen.flow_from_directory(test_path, class_mode='sparse')
    facenet.evaluate(test_data)

    # Test model on individual image
    database = build_database(train_path)
    image = cv2.imread(test_image)
    image = np.expand_dims(image, axis=0)
    feature = facenet.predict(image)
    name = get_identity(feature, database)
    print("Person predicted: ", name)