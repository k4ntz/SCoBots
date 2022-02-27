import os
import cv2

# class for creating video and saving it

class VideoLogger:
    def __init__(self, vfolder="/xrl/video/", size=(128, 128)):
        self.PATH_TO_VIDEO = os.getcwd() + vfolder
        self.video_buffer = []
        self.size = size

    def fill_video_buffer(self, image, fps=30):
        self.video_buffer.append(image)

    def save_video(self, model_name, fps=25.0):
        if not os.path.exists(self.PATH_TO_VIDEO):
            os.makedirs(self.PATH_TO_VIDEO)
        if len(self.video_buffer) > 15:
            file_path = self.PATH_TO_VIDEO + model_name + ".avi"
            # do video saving stuff
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            writer = cv2.VideoWriter(file_path, fourcc, fps, self.size)
            # fill buffer of video writer
            for frame in self.video_buffer:
                writer.write(frame)
            writer.release() 
            self.video_buffer = []
        else:
            print("Warning: Trying to write log video without enough frames!")