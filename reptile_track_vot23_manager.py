import vot
import sys
import time
import cv2
import numpy as np

from PIL import Image
# from Tracker_Reptile import ReptileTracker
# from Tracker_ReptileR1 import ReptileTracker
from Tracker_ReptileR3 import ReptileTracker

def NCCTracker(image, region):
        
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    window = max(region.width, region.height) * 2

    left = max(region.x, 0)
    top = max(region.y, 0)

    right = min(region.x + region.width, image.shape[1] - 1)
    bottom = min(region.y + region.height, image.shape[0] - 1)

    template = image[int(top):int(bottom), int(left):int(right)]
    position = [region.x + region.width / 2, region.y + region.height / 2]
    size = (region.width, region.height)

    def track(image):
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

        left = max(round(position[0] - float(window) / 2), 0)
        top = max(round(position[1] - float(window) / 2), 0)

        right = min(round(position[0] + float(window) / 2), image.shape[1] - 1)
        bottom = min(round(position[1] + float(window) / 2), image.shape[0] - 1)

        if right - left < template.shape[1] or bottom - top < template.shape[0]:
            return vot.Rectangle(position[0] + size[0] / 2, position[1] + size[1] / 2, size[0], size[1])

        cut = image[int(top):int(bottom), int(left):int(right)]

        matches = cv2.matchTemplate(cut, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matches)

        position[0] = left + max_loc[0] + float(size[0]) / 2
        position[1] = top + max_loc[1] + float(size[1]) / 2
        return vot.Rectangle(left + max_loc[0], top + max_loc[1], size[0], size[1])

    return track

def VotReptileTracker(imagefile, region):
    image = Image.open(imagefile).convert('RGB')
    # tracker = ReptileTracker(image, region, False) # disable displaying results
    tracker = ReptileTracker(image, region, True)  # enable displaying results
    def track(imagefile):
        image = Image.open(imagefile).convert('RGB')
        region, confidence = tracker.track(image, None )
        return vot.Rectangle(region[0], region[1], region[2], region[3])

    return track

if __name__ == "__main__":
    print(vot.__file__)
    # manager = vot.VOTManager(NCCTracker, "rectangle")
    manager = vot.VOTManager(VotReptileTracker, "rectangle")
    manager.run()