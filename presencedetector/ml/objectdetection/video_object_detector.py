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
from matplotlib import pyplot as plt
import os




class VideoObjectDetector:
    def __init__(self, exec_path):
        self.exec_path = exec_path
        self.detector = VideoObjectDetection()
        self.config = ConfigLoader().conf
        self.detector.setModelTypeAsRetinaNet()
        self.detector.setModelPath(os.path.join(self.exec_path, self.config['obj-detector.model-path']))
        self.detector.loadModel()

    def for_frame(self, frame_number, output_array, output_count, returned_frame, resized=False):
        color_index = {'bus': 'red', 'handbag': 'steelblue', 'giraffe': 'orange', 'spoon': 'gray', 'cup': 'yellow',
                       'chair': 'green', 'elephant': 'pink', 'truck': 'indigo', 'motorcycle': 'azure',
                       'refrigerator': 'gold', 'keyboard': 'violet', 'cow': 'magenta', 'mouse': 'crimson',
                       'sports ball': 'raspberry', 'horse': 'maroon', 'cat': 'orchid', 'boat': 'slateblue',
                       'hot dog': 'navy', 'apple': 'cobalt', 'parking meter': 'aliceblue', 'sandwich': 'skyblue',
                       'skis': 'deepskyblue', 'microwave': 'peacock', 'knife': 'cadetblue', 'baseball bat': 'cyan',
                       'oven': 'lightcyan', 'carrot': 'coldgrey', 'scissors': 'seagreen', 'sheep': 'deepgreen',
                       'toothbrush': 'cobaltgreen', 'fire hydrant': 'limegreen', 'remote': 'forestgreen',
                       'bicycle': 'olivedrab', 'toilet': 'ivory', 'tv': 'khaki', 'skateboard': 'palegoldenrod',
                       'train': 'cornsilk', 'zebra': 'wheat', 'tie': 'burlywood', 'orange': 'melon', 'bird': 'bisque',
                       'dining table': 'chocolate', 'hair drier': 'sandybrown', 'cell phone': 'sienna', 'sink': 'coral',
                       'bench': 'salmon', 'bottle': 'brown', 'car': 'silver', 'bowl': 'maroon',
                       'tennis racket': 'palevilotered', 'airplane': 'lavenderblush', 'pizza': 'hotpink',
                       'umbrella': 'deeppink', 'bear': 'plum', 'fork': 'purple', 'laptop': 'indigo',
                       'vase': 'mediumpurple', 'baseball glove': 'slateblue', 'traffic light': 'mediumblue',
                       'bed': 'navy', 'broccoli': 'royalblue', 'backpack': 'slategray', 'snowboard': 'skyblue',
                       'kite': 'cadetblue', 'teddy bear': 'peacock', 'clock': 'lightcyan', 'wine glass': 'teal',
                       'frisbee': 'aquamarine', 'donut': 'mincream', 'suitcase': 'seagreen', 'dog': 'springgreen',
                       'banana': 'emeraldgreen', 'person': 'honeydew', 'surfboard': 'palegreen', 'cake': 'sapgreen',
                       'book': 'lawngreen', 'potted plant': 'greenyellow', 'toaster': 'ivory', 'stop sign': 'beige',
                       'couch': 'khaki'}

        plt.clf()

        this_colors = []
        labels = []
        sizes = []

        counter = 0

        for eachItem in output_count:
            counter += 1
            labels.append(eachItem + " = " + str(output_count[eachItem]))
            sizes.append(output_count[eachItem])
            this_colors.append(color_index[eachItem])

        resized

        if not resized:
            manager = plt.get_current_fig_manager()
            manager.resize(width=1000, height=500)
            resized = True

        plt.subplot(1, 2, 1)
        plt.title("Frame : " + str(frame_number))
        plt.axis("off")
        plt.imshow(returned_frame, interpolation="none")

        plt.subplot(1, 2, 2)
        plt.title("Analysis: " + str(frame_number))
        plt.pie(sizes, labels=labels, colors=this_colors, shadow=True, startangle=140, autopct="%1.1f%%")

        plt.pause(0.01)

    def run_inference_on(self, camera):
        self.detector.detectCustomObjectsFromVideo(
            custom_objects=self.detector.CustomObjects(person=True),
            camera_input=camera,
            frames_per_second=2,
            log_progress=True,
            save_detected_video=False,
            per_frame_function=self.for_frame,
            return_detected_frame=True
        )
