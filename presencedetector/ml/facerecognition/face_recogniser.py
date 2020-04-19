#  Copyright 2020 CJWW Development
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from presencedetector.config.config_loader import ConfigLoader
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os


class FaceRecogniser:
    def __init__(self):
        self.conf = ConfigLoader().conf['face-recognition']
        self.image_paths = list(paths.list_images(self.conf['dataset']))

    def is_model_generated(self):
        try:
            f = open('model.lock')
            print('model.lock detected, skipping model generation')
            f.close()
            return True
        except FileNotFoundError:
            print('No model.lock detected, generating model based on face rec dataset')
            f = open('model.lock', 'x')
            return False


    def load_dataset(self):
        known_encodings = []
        known_names = []
        for(i, img_path) in enumerate(self.image_paths):
            print("[INFO] processing image {}/{}".format(i + 1, len(self.image_paths)))
            name = img_path.split(os.path.sep)[-2]
            image = cv2.imread(img_path)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb, model=self.conf['mode'])
            encodings = face_recognition.face_encodings(rgb, boxes)
            for encoding in encodings:
                known_encodings.append(encoding)
                known_names.append(name)
                print("[INFO] serializing encodings...")
                data = {"encodings": known_encodings, "names": known_names}
                f = open('encodings.pickle', "wb")
                f.write(pickle.dumps(data))
                f.close()

    def run_inference_on(self, img):
        print('Loading encodings')
        data = pickle.loads(open(self.conf['encodings-file'], 'rb').read())

        image = cv2.imread(img)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        print('Recognising faces...')
        boxes = face_recognition.face_locations(rgb, model=self.conf['mode'])
        encodings = face_recognition.face_encodings(rgb, boxes)

        names = []

        for encoding in encodings:
            matches = face_recognition.compare_faces(data['encodings'], encoding)
            name = 'Unknown'
            if True in matches:
                matched_idxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                for i in matched_idxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                name = max(counts, key=counts.get)

            names.append(name)

            for ((top, right, bottom, left), name) in zip(boxes, names):
                cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 255, 0), 2)
            # show the output image
            # cv2.imshow("Image", image)
            # cv2.waitKey(0)

            print(names)
