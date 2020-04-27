import cv2
import numpy as np
import os
import sys

class BoundingBox(object):
	def __init__(self):
		self.bboxs = []

	def run_bbox(self):
		for i in sorted(os.listdir('./test_images/')):
			if '.jpg' in i:
				print('Finding faces in image',i)
				os.system("python yoloface/yoloface.py --image test_images/"+i+" --output-dir output_bbox/")
		print('All bounding boxes extracted successfully')

if __name__ == '__main__':
    bbox = BoundingBox()
    bbox.run_bbox()
