# -*- coding = utf-8 -*-
# @Time     : 2022/6/10 0:50
# @Author   : 陈梓雄
# @ID       : 2009853G-II20-0083
# @File     : check_coco_tf_record.py
# @Software : PyCharm
import os
import tensorflow as tf
import sys
import shutil

ROOT_DIR = os.path.abspath("C:/MUST/Graduation_Project/DLML/Code/data/tf_example/coco/val")
sys.path.append(ROOT_DIR)
TF_PATH = os.path.join(ROOT_DIR, "val-?????-of-00032.tfrecord")
TXT_PATH = os.path.join(ROOT_DIR, "images_missing_bbox.txt")

IMG_DIR = "C:\\MUST\\Graduation_Project\\DLML\\Code\\data\\coco2017\\image\\"
IMG_PATH = IMG_DIR + "val2017\\"
TARGET_PATH = IMG_DIR + "val_no_box\\"

file_list = tf.gfile.Glob(TF_PATH)
i = 0
count = 0
image_filenames = []
for file in file_list:
    for string_record in tf.python_io.tf_record_iterator(file):
        example = tf.train.Example()
        example.ParseFromString(string_record)

        i += 1
        image_id = example.features.feature['image/source_id'].bytes_list.value
        image_filename = example.features.feature['image/filename'].bytes_list.value
        xmax = example.features.feature['image/object/bbox/xmax'].float_list.value
        # xmin = example.features.feature['image/object/bbox/xmin'].float_list.value
        # ymax = example.features.feature['image/object/bbox/ymax'].float_list.value
        # ymin = example.features.feature['image/object/bbox/ymin'].float_list.value

        if xmax == []:
            image_filenames.append(image_filename)
            print(image_id)
            print(image_filename)
            count += 1
    print(file, 'has been checked.')

print(i, 'images have been checked.')
print(count, 'annotations without bbox.')

BUCKET_DIR = "gs://mapan/coco2017/image/"
OLD_PATH = BUCKET_DIR + "val2017/"
NEW_PATH = OLD_PATH.replace('train2017', 'train2017_bbox').replace('val2017', 'val2017_bbox')

filenames = []
TXT_PATH = os.path.join(ROOT_DIR, "images_missing_bbox.sh")
f = open(TXT_PATH, 'w')
f.write("export NEW_PATH=${" + NEW_PATH + "}" + "\n")
NEW_PATH = "${NEW_PATH}"
for j in image_filenames:
    j = str(j)
    j = j.replace('[','').replace('b','').replace('\'','').replace('\'','').replace(']','').strip('\n')
    print(j)
    if len(j) < 12:
        j = j.zfill(12)
    filenames.append(j)
    name = j
    command = "gsutil mv " + OLD_PATH + name + " " + NEW_PATH
    f.write(command + "\n")
#     f.write(j + "\n")
f.close()
# gsutil -m cp ./official/vision/data/images_missing_bbox.sh gs://mapan/coco2017/image/

IMG_PATH = IMG_DIR + "val2017\\"
TARGET_PATH = IMG_DIR + "val_no_box\\"
for num, name in enumerate(filenames):
    shutil.move(IMG_PATH + name, TARGET_PATH + name)
    print(num+1, name, " from ", IMG_PATH, " to ", TARGET_PATH)
