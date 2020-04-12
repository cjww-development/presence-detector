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

from presencedetector.ml.object_detector import ObjectDetector
from presencedetector.config.config_loader import ConfigLoader
import os


class App:
    def __init__(self):
        self.config = ConfigLoader().conf
        self.exec_path = os.getcwd()
        self.objDetector = ObjectDetector(self.exec_path)

    def run(self):
        self.objDetector.detect_objects(
            input_path=os.path.join(self.exec_path, self.config['obj-detector.input-path']),
            output_path=os.path.join(self.exec_path, self.config['obj-detector.output-path'])
        )
