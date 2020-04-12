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

from imageai.Detection import ObjectDetection
from presencedetector.config.config_loader import ConfigLoader
import os


class ObjectDetector:
    def __init__(self, exec_path):
        self.detector = ObjectDetection()
        self.config = ConfigLoader().conf
        self.detector.setModelTypeAsRetinaNet()
        self.detector.setModelPath(os.path.join(exec_path, self.config['obj-detector.model-path']))
        self.detector.loadModel()

    def detect_objects(self, input_path, output_path):
        detections = self.detector.detectObjectsFromImage(input_image=input_path, output_image_path=output_path )
        for obj in detections:
            print(obj["name"], " : ", obj["percentage_probability"])
