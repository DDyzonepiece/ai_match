# -*- coding: utf-8 -*-

import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile



VALIDATION_PRECENTAGE=10
TEST_PRECENTAGE=10

def create_image_lists(sess,testing_percentage,validation_percentage):
    INPUT_DATA = r'D:\tianchi_dataset\IJCAI_2019_AAAC_train'
    current_label = 0


    sub_dirs=[x[0] for x in os.walk(INPUT_DATA)]

    #is_root_dir=True

    training_images=[]
    training_labels=[]
    testing_images=[]
    testing_labels=[]
    validation_images=[]
    validation_labels=[]


    file_and_label = []
    print(sub_dirs)
    for sub_dir in sub_dirs[1:]:
        # if is_root_dir:
        #     is_root_dir=False
        #     continue
        extensions=['jpg','jpeg','JPG','JPEG']

        #dir_name=os.path.basename(sub_dir)
        #for extension in extensions[0]:
        file_glob=os.path.join(sub_dir,'*.jpg')
        print(file_glob)
        file_list=glob.glob(file_glob)


        print(len(file_list))
        file_and_label_temp=[(x,current_label) for x in file_list]
        file_and_label.extend(file_and_label_temp)
        current_label += 1


    random.shuffle(file_and_label)
    random.shuffle(file_and_label)
    #print(file_and_label)

        #if not file_list:
            #continue

        # for file_name in file_list:
        #
        #     image_raw_data=gfile.FastGFile(file_name,'rb').read()
        #
        #     image=tf.image.decode_jpeg(image_raw_data)
        #
        #     if image.dtype !=tf.float32:
        #         image=tf.image.convert_image_dtype(
        #             image,dtype=tf.float32
        #         )
        #     image=tf.image.resize_images(image,[299,299])
        #     image_value=sess.run(image)

            # chance=np.random.randint(100)
            # if chance<validation_percentage:
            #
            #     validation_images.append(image_value)
            #     validation_labels.append(current_label)
            #
            # elif chance<(testing_percentage+validation_percentage):
            #     testing_images.append(image_value)
            #     testing_labels.append(current_label)
            # else:
            #     training_images.append(image_value)
            #     training_labels.append(current_label)

        # OUTPUT_FILE = r'flower_photos\flower_processed_data_%d.npy' % (current_label)


    # state=np.random.get_state()
    # np.random.shuffle(training_images)
    # np.random.set_state(state)
    # np.random.shuffle(training_labels)

        # processed_data=np.asarray([training_images,training_labels,
        #                validation_images,validation_labels,
        #                testing_images,testing_labels])
        #
        # np.save(OUTPUT_FILE, processed_data)

def main():
    with tf.Session() as sess:
        create_image_lists(
            sess,TEST_PRECENTAGE,VALIDATION_PRECENTAGE
        )
        #np.save(OUTPUT_FILE,processed_data)

if __name__=='__main__':
    main()