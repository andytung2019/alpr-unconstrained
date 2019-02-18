import sys
import cv2
import numpy as np
import traceback

import darknet.python.darknet as dn

from src.label 				import Label, lwrite
from os.path 				import splitext, basename, isdir
from os 					import makedirs
from src.utils 				import crop_region, image_files_from_folder
from darknet.python.darknet import detect

class MyCheck(object):
	def __init__(self):
		self.list_out = []
		self.list_pic = []

		try:
	
			self.input_dir  = sys.argv[1]
			self.output_dir = sys.argv[2]

			self.vehicle_threshold = .5


			self.vehicle_weights = 'data/vehicle-detector/yolo-voc.weights'
			self.vehicle_netcfg  = 'data/vehicle-detector/yolo-voc.cfg'
			self.vehicle_dataset = 'data/vehicle-detector/voc.data'

			self.vehicle_net  = dn.load_net(self.vehicle_netcfg, self.vehicle_weights, 0)
			self.vehicle_meta = dn.load_meta(self.vehicle_dataset)

			self.imgs_paths = image_files_from_folder(self.input_dir)
			self.imgs_paths.sort()

			#if not isdir(self.output_dir):
			#	makedirs(self.output_dir)
		except:
			traceback.print_exc()
			sys.exit(1)


	def get_pic_list(self):
		pic_subfix = ['jpg', 'JPG', 'JPEG', 'jpeg', 'png']
		path = '/content/images'
		path = os.path.expanduser(path)
		for (dirname, subdir, subfile) in os.walk(path):
			for f in subfile:
				sufix = os.path.splitext(f)[1][1:]
				if sufix in pic_subfix:
					path = os.path.join(dirname, f)
					dit= (f, path)
					self.list_pic.append(dit)
		print 'get pic list:%d' % len(self.list_pic)

	def detect_pic_list(self, list_path):
		for f_path in list_path:
			pic_name = f_path[0]
			image = skimage.io.imread(f_path[1])
			R,_ = detect(self.vehicle_net, self.vehicle_meta, f_path[1] ,thresh=self.vehicle_threshold)
			R = [r for r in R if r[0] in ['car','bus','pickup truck','truck']]
			if len(R):
				self.list_out.append({'image_name':pic_name, 'class':r[0],'percent':r[1],'rois': r[2]})


	def write_to_csv(self):
		with open(r'out.csv', 'a') as out_csv:
			fields = ['image_name', 'class','percent','rois_a', 'rois_b','rois_c', 'rois_d']
			for i in range(len(self.list_out)):
				r = self.list_out[i]
				writer = csv.DictWriter(out_csv, fieldnames=fields)
				writer.writerow({'image_name':r['image_name'], 'class':r['class'],'percent':r['percent'], 'rois_a': int(r['rois'][0]), 'rois_b':int(r['rois'][1]), 'rois_c':int(r['rois'][2]), 'rois_d':int(r['rois'][3])})


	def detect_all_pic(self):
		print "img list len:%d" % len(self.list_pic)

		start = 0
		end = 0
		while start < len(self.list_pic):
			tm_start = time.clock()
			list_path = []
			
			start = end
			end = start + 10
			if end >= len(self.list_pic):
				end = len(self.list_pic)
			for idx in range(start, end):
				list_path.append(self.list_pic[idx])
			self.detect_pic_list(list_path)
			tm_end = time.clock()
			#print("image from %d->%d time:%d", start, end, tm_end-tm_start)

mycheck = MyCheck()

mycheck.get_pic_list()
mycheck.detect_all_pic(10)
mycheck.write_to_csv()

