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
#
# from presencedetector.ml.objectdetection.static_object_detector import StaticObjectDetector
# from presencedetector.ml.objectdetection.video_object_detector import VideoObjectDetector
from presencedetector.ml.facerecognition.face_recogniser import FaceRecogniser
from presencedetector.config.config_loader import ConfigLoader
import os
import cv2


class App:
    def __init__(self):
        self.config = ConfigLoader().conf
        self.exec_path = os.getcwd()
        self.face_rec = FaceRecogniser()

    def run(self):
        if not self.face_rec.is_model_generated():
            self.face_rec.load_dataset()

        test_img = self.config['face-recognition.test-img']
        self.face_rec.run_inference_on(test_img)

        # if self.config['mode.static']:
        #     obj_detector = StaticObjectDetector(self.exec_path)
        #     obj_detector.run_inference_on(
        #         input_path=os.path.join(self.exec_path, self.config['obj-detector.input-path']),
        #         output_path=os.path.join(self.exec_path, self.config['obj-detector.output-path.static'])
        #     )
        # else:
        #     video_obj_detector = VideoObjectDetector(self.exec_path)
        #     video_input = cv2.VideoCapture(0)
        #     video_obj_detector.run_inference_on(camera=video_input)
