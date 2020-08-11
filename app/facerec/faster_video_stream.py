import cv2
import pickle
import face_recognition
from mtcnn import MTCNN
from .Resnet import Resnet34Triplet
import numpy as np
import torch
import json

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def predict_orig(rgb_frame, knn_clf=None, model_path=None, distance_threshold=0.5):
    """
    Recognizes faces in given image using a trained KNN classifier

    :param X_img_path: path to image to be recognized
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
    """
    # if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
    #     raise Exception("Invalid image path: {}".format(X_img_path))

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    # X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(rgb_frame, number_of_times_to_upsample=2)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(rgb_frame, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
    # print(closest_distances)
    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]


def predict(rgb_frame, model_path=None, data_path=None):  # distance_threshold=0.5):

    if model_path is None:
        raise Exception("Must supply model_path")
    if data_path is None:
        raise Exception("must supply data_path")

    checkpoint = torch.load(model_path)
    model = Resnet34Triplet(embedding_dimension=checkpoint['embedding_dimension']).cuda()
    model.load_state_dict(checkpoint['model_state_dict'])
    distance_threshold = checkpoint['best_distance_threshold']
    device = torch.device('cpu')
    mtcnn = MTCNN()

    x_img = np.array(rgb_frame)
    out = mtcnn.detect_faces(x_img)

    faces_encodings = []
    dict = {}
    for i in out:
        prob = i['confidence']
        box = i['box']  # [x,y,w,h]
        x_aligned = x_img[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
        box_array = np.array([box[1] + box[3], box[0] + box[2], box[1], box[0]])
        if x_aligned is not None:
            # print('Face detected with probability: {:8f}'.format(prob))
            x_aligned = cv2.resize(x_aligned, (100, 100), interpolation=cv2.INTER_CUBIC)
            model.to(device)

            faces_encoding = model(torch.tensor(x_aligned, dtype=torch.float).permute(2, 1, 0).unsqueeze(0)).detach()
            faces_encodings.append(faces_encoding)
            dict[faces_encoding] = box_array

    with open(data_path, "r") as read_file:
        embeddings = json.load(read_file)
    are_matches = []

    for k, e1 in embeddings.items():
        for e2 in faces_encodings:
            dict2 = {}
            dist = (torch.tensor(e1) - e2).norm().item()
            if dist <= distance_threshold:
                dict2[k] = dict[e2]
                are_matches.append([key, value] for key, value in dict2.items())
            else:
                dict2["unknown"] = dict[e2]
                are_matches.append([key, value] for key, value in dict2.items())

    return are_matches


# if __name__ == "__main__":
def stream():
    video_capture = cv2.VideoCapture(0)

    process_this_frame = True

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = small_frame[:, :, ::-1]

        if process_this_frame:
            predictions = predict(rgb_frame, model_path="madels/facenet/model_resnet34_triplet.pt", data_path="embeddings.json")
            # print(predictions)

        process_this_frame = not process_this_frame

        for j in predictions:
            x = list(j)
            if x == []:
                x_ = ["", [0, 0, 0, 0]]
            else:
                x_ = x[0]
            name, arr = x_[0], x_[1]
            top, right, bottom, left = arr[0], arr[1], arr[2], arr[3]

            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)


        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows() 

