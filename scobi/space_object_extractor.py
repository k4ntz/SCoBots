
from typing import OrderedDict
import numpy as np
from torchvision import transforms
import time
import torch

from scobi.object_extractor import ObjectExtractor
from scobi.utils.SPACEGameObject import KFandSPACEGameObject


#from motrackers import load_space_detector
#from motrackers import CentroidKF_Tracker, CentroidTracker

# from space repository
import sys
import os
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_PATH, "spaceandmoc", "src"))
from motrackers import load_space_detector
from motrackers import CentroidKF_Tracker, CentroidTracker


def get_game_name(env_name):
    game_name = env_name.split("/")[-1].split("-")[0]
    game_name = game_name.lower()
    return game_name

def scobi_image2space_image(img):
    #img =  Image.fromarray(img[:, :, ::-1], 'RGB')
    #img = img.resize((128, 128), Image.LANCZOS)
    #img = transforms.ToTensor()(img).unsqueeze(0).to("cuda")
    #return img

    # store image before resizing
    #original_img = Image.fromarray(img[:, :, ::-1], 'RGB')
    ## save original_img
    #original_img.save("original_img.png")
    #original_img = original_img.resize((128, 128), Image.LANCZOS)
    #original_img = transforms.ToTensor()(original_img).unsqueeze(0).to("cuda")

    img = img[:, :, ::-1].copy()
    img_tensor = torch.from_numpy(img).to("cuda")
    img_tensor = img_tensor.permute(2, 0, 1).float() / 255.0
    # Resize the image using PyTorch
    # Note: Interpolation mode 'bilinear' is usually a good balance between speed and quality
    resize = transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
    img_tensor = resize(img_tensor)
    img_tensor = img_tensor.unsqueeze(0)

    # save img_tensor for debugging
    #img_tensor = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #img_tensor_for_save = Image.fromarray((img_tensor * 255).astype(np.uint8)[:,:,::-1], 'RGB')
    #img_tensor_for_save.save("img_tensor.png")

    return img_tensor

def space_bboxes2scobi_bboxes(bboxes, game_name):
    if len(bboxes) == 0:
        return bboxes

    x_shift = 0
    y_shift = 0
    if game_name == "pong":
        x_shift = 6
        y_shift = 9
    elif game_name == "boxing":
        x_shift = 10
        y_shift = 25
    else:
        raise ValueError("game_name not recognized")
    
    bboxes[:, 0] = bboxes[:, 0] * (160 / 128) + x_shift
    bboxes[:, 1] = bboxes[:, 1] * (210 / 128) + y_shift
    bboxes[:, 2] = bboxes[:, 2] * (160 / 128)
    bboxes[:, 3] = bboxes[:, 3] * (210 / 128)
    bboxes = np.array(bboxes, dtype=np.int32)
    return bboxes

#class SPOCObjectExtractor(ObjectExtractor):
#    def __init__(self, env_name):
#        self.game_name = get_game_name(env_name)
#        self.spoc = load_spoc_detector(self.game_name)
#        self.previous_game_objects_dict = {} #key: class_id, value: KFandSPOCGameObject
#
#    def get_objects(self, img):
#        img = scobi_image2spoc_image(img)
#        bboxes, confidences, class_ids, _ = self.spoc.detect(img)
#        bboxes = spoc_bboxes2scobi_bboxes(bboxes)
#
#        objects = self.spocoutput2objects(bboxes, confidences, class_ids)
#        return objects
#
#    def spocoutput2objects(self, bboxes, confidences, class_ids):
#        # assumes that only one object per class can be detected
#        #pong_dict = {0: 4, 1: 1, 2: 2}
#        #class_ids = [pong_dict[class_id] for class_id in class_ids]
#        for bbox, confidence, class_id in zip(bboxes, confidences, class_ids):
#            if class_id not in self.previous_game_objects_dict:
#                self.previous_game_objects_dict[class_id ] = KFandSPOCGameObject(bbox[0], bbox[1], bbox[2], bbox[3], class_id, confidence, self.game_name)
#            else:
#                self.previous_game_objects_dict[class_id].prev_xy = self.previous_game_objects_dict[class_id].xy
#                self.previous_game_objects_dict[class_id].xy = (bbox[0], bbox[1])
#                self.previous_game_objects_dict[class_id].class_id = class_id
#                self.previous_game_objects_dict[class_id].w = bbox[2]
#                self.previous_game_objects_dict[class_id].h = bbox[3]
#
#        class_ids_to_remove = []
#        for class_id in self.previous_game_objects_dict:
#            if class_id not in class_ids:
#                class_ids_to_remove.append(class_id)
#        for class_id in class_ids_to_remove:
#            del self.previous_game_objects_dict[class_id]
#            
#        return list(self.previous_game_objects_dict.values())

class TrackerAndDetectorObjectExtractor(ObjectExtractor):
    """
    Abstract class for object extractors that use a tracker and a detector
    """

    def __init__(self, env_name, tracker, detector):
        self.game_name = get_game_name(env_name)
        self.detector = detector
        self.tracker = tracker
        self.previous_game_objects_dict = {}

    def get_objects(self, img):
        img_trans_time_start = time.time()
        img = scobi_image2space_image(img)
        img_trans_time_end = time.time()

        detector_time_start = time.time()
        bboxes, confidences, class_ids, _ = self.detector.detect(img)
        #import ipdb; ipdb.set_trace()
        detector_time_end = time.time()
        bboxes = space_bboxes2scobi_bboxes(bboxes, self.game_name)
        tracker_time_start = time.time()
        self.tracker.update(bboxes, confidences, class_ids)
        tracker_time_end = time.time()
        tracks2objects_time_start = time.time()
        objects = self._tracks2objects()
        tracks2objects_time_end = time.time()
        #print("img_trans_time", img_trans_time_end-img_trans_time_start)
        #print("detector_time", detector_time_end-detector_time_start)
        #print("tracker_time", tracker_time_end-tracker_time_start)
        #print("tracks2objects_time", tracks2objects_time_end-tracks2objects_time_start)
        return objects
    
    def _tracks2objects(self,):
        tracks: OrderedDict = self.tracker.tracks
        for track in tracks.values():
            track_id = track.id
            bbox = np.array(track.bbox, dtype=np.int32)
            if track_id not in self.previous_game_objects_dict:
                self.previous_game_objects_dict[track_id] = KFandSPACEGameObject(bbox[0], bbox[1], bbox[2], bbox[3], track.class_id, track.detection_confidence, self.game_name)
            else:
                self.previous_game_objects_dict[track_id].prev_xy = self.previous_game_objects_dict[track_id].xy
                self.previous_game_objects_dict[track_id].xy = (bbox[0], bbox[1])
                self.previous_game_objects_dict[track_id].class_id = track.class_id
                #self.previous_game_objects_dict[track_id].w = bbox[2]
                #self.previous_game_objects_dict[track_id].h = bbox[3]
        
        track_ids_to_remove = []
        for track_id in self.previous_game_objects_dict:
            if track_id not in tracks:
                track_ids_to_remove.append(track_id)
        for track_id in track_ids_to_remove:
            del self.previous_game_objects_dict[track_id]
        return list(self.previous_game_objects_dict.values())
    
class CentroidTrackerAndSPACEObjectExtractor(TrackerAndDetectorObjectExtractor):
    def __init__(self, env_name):
        detector = load_space_detector(get_game_name(env_name))
        tracker = CentroidTracker(max_lost=0)
        super().__init__(env_name, tracker, detector)

class CentroidKFTrackerAndSPACEObjectExtractor(TrackerAndDetectorObjectExtractor):
    def __init__(self, env_name):
        detector = load_space_detector(get_game_name(env_name))
        tracker = CentroidKF_Tracker(max_lost=0)
        super().__init__(env_name, tracker, detector)
    





        