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


from imageai.Detection import VideoObjectDetection
from presencedetector.config.config_loader import ConfigLoader
import os


class VideoObjectDetector:
    def __init__(self, exec_path):
        self.exec_path = exec_path
        self.detector = VideoObjectDetection()
        self.config = ConfigLoader().conf
        self.detector.setModelTypeAsRetinaNet()
        self.detector.setModelPath(os.path.join(self.exec_path, self.config['obj-detector.model-path']))
        self.detector.loadModel()

    def run_inference_on(self, camera):
        self.detector.detectObjectsFromVideo(
            camera_input=camera,
            output_file_path=os.path.join(self.exec_path, self.config['obj-detector.output-path.video']),
            frames_per_second=self.config['obj-detector.video.fps'],
            log_progress=self.config['obj-detector.video.log-progress'],
            minimum_percentage_probability=self.config['obj-detector.video.confidence']
        )