from django.shortcuts import render, get_object_or_404
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.utils import timezone
from .facerec.faster_video_stream import stream
from .facerec.click_photos import click
from .facerec.train_faces import trainer
from .models import Employee, Detected
from .forms import EmployeeForm
import cv2
import pickle
import face_recognition
import datetime
from cachetools import TTLCache
from mtcnn import MTCNN
from .facerec.Resnet import Resnet34Triplet
import numpy as np
import torch
import json

cache = TTLCache(maxsize=20, ttl=60)


def identify1(frame, name, buf, buf_length, known_conf):
    if name in cache:
        return
    count = 0
    for ele in buf:
        count += ele.count(name)

    if count >= known_conf:
        timestamp = datetime.datetime.now(tz=timezone.utc)
        print(name, timestamp)
        cache[name] = 'detected'
        path = 'detected/{}_{}.jpg'.format(name, timestamp)
        write_path = 'media/' + path
        cv2.imwrite(write_path, frame)
        try:
            emp = Employee.objects.get(name=name)
            emp.detected_set.create(time_stamp=timestamp, photo=path)
        except:
            pass


def predict(rgb_frame, model_path=None, data_path=None):  # distance_threshold=0.5):

    if model_path is None:
        raise Exception("Must supply model_path")
    if data_path is None:
        raise Exception("must supply data_path")

    checkpoint = torch.load(model_path)
    model = Resnet34Triplet(embedding_dimension=checkpoint['embedding_dimension'])
    model.load_state_dict(checkpoint['model_state_dict'])
    distance_threshold = checkpoint['best_distance_threshold'] * 10**(-3)
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


def identify_faces(video_capture):
    buf_length = 10
    known_conf = 6
    buf = [[]] * buf_length
    i = 0

    process_this_frame = True

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = small_frame[:, :, ::-1]

        if process_this_frame:
            predictions = predict(rgb_frame, model_path="app/facerec/models/facenet/model_resnet34_triplet.pt", data_path="app/facerec/embeddings.json")
            # print(predictions)

        process_this_frame = not process_this_frame

        face_names = []

        #for name, (top, right, bottom, left) in predictions:
        for j in predictions:
            x = list(j)
            if x == []:
                pass
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

            identify1(frame, name, buf, buf_length, known_conf)

            face_names.append(name)

        buf[i] = face_names
        i = (i + 1) % buf_length

        # print(buf)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()


def index(request):
    return render(request, 'app/index.html')


def video_stream(request):
    stream()
    return HttpResponseRedirect(reverse('index'))


def add_photos(request):
    emp_list = Employee.objects.all()
    return render(request, 'app/add_photos.html', {'emp_list': emp_list})


def click_photos(request, emp_id):
    cam = cv2.VideoCapture(0)
    emp = get_object_or_404(Employee, id=emp_id)
    click(emp.name, emp.id, cam)
    return HttpResponseRedirect(reverse('add_photos'))


def train_model(request):
    trainer()
    return HttpResponseRedirect(reverse('index'))


def detected(request):
    if request.method == 'GET':
        date_formatted = datetime.datetime.today().date()
        date = request.GET.get('search_box', None)
        if date is not None:
            date_formatted = datetime.datetime.strptime(date, "%Y-%m-%d").date()
        det_list = Detected.objects.filter(time_stamp__date=date_formatted).order_by('time_stamp').reverse()

    # det_list = Detected.objects.all().order_by('time_stamp').reverse()
    return render(request, 'app/detected.html', {'det_list': det_list, 'date': date_formatted})


def identify(request):
    video_capture = cv2.VideoCapture(0)
    identify_faces(video_capture)
    return HttpResponseRedirect(reverse('index'))


def add_emp(request):
    if request.method == "POST":
        form = EmployeeForm(request.POST)
        if form.is_valid():
            emp = form.save()
            # post.author = request.user
            # post.published_date = timezone.now()
            # post.save()
            return HttpResponseRedirect(reverse('index'))
    else:
        form = EmployeeForm()
    return render(request, 'app/add_emp.html', {'form': form})


def temperature():
    temperature = models.CharField(max_length=3)


def signin_signout():
    pass
