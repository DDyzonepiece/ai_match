# -*- coding: utf-8 -*-

import sqlite3
import glob
import io
import gc
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

def adapt_array(arr):

    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def creat_tensor_table(table_name,cursor):
    # 创建数据库表
    #cursor=conn.cursor()
    sql = 'create table  if not exists ' + table_name + \
          '(id int primary key  not null,\
        label    int    not null,\
        tensor   array   not null)'
    cursor.execute(sql)
    #conn.commit()

#表格中插入数据
def db_jpgtonp_in(sess,table_name,id,label,file_name_jpg,cursor):
    file_name=file_name_jpg
    print(file_name)
    image_raw_data=gfile.FastGFile(file_name,'rb').read()

    image=tf.image.decode_jpeg(image_raw_data)

    if image.dtype !=tf.float32:
        image=tf.image.convert_image_dtype(
            image,dtype=tf.float32
        )

    image=tf.image.resize_images(image,[299,299])
    image_value=sess.run(image)
        # 插入一行数据
    #cursor = conn.cursor()
    sql_insert = "insert into " + table_name + "(id,label,tensor) values (?,?,?)"
    cursor.execute(sql_insert, (id, label, image_value))
    #清除内存

    del image,image_value

    # 提交
    #conn.commit()



def create_image_lists(photo_file):
    current_label = 0

    sub_dirs=[x[0] for x in os.walk(photo_file)]#读取每个文件夹
    file_and_label = []
    #选取该文件夹下所有非根文件夹
    for sub_dir in sub_dirs:
    #for sub_dir in sub_dirs[1:]:

        file_glob=os.path.join(sub_dir,'*.jpg')#选取该文件夹下所有jpg图片
        file_list=glob.glob(file_glob)

        file_and_label_temp=[(x,current_label) for x in file_list]
        file_and_label.extend(file_and_label_temp)
        current_label += 1#给定标签
    #将列表随机打乱
    print(len(file_and_label))
    random.shuffle(file_and_label)
    random.shuffle(file_and_label)
    #print(file_and_label,len(file_and_label))测试用
    return file_and_label

photo_file = r'D:\tianchi_dataset\IJCAI_2019_AAAC_train\00003'
file_and_label=create_image_lists(photo_file)
conn = sqlite3.connect(r'D:\database\test.db',detect_types=sqlite3.PARSE_DECLTYPES)

sqlite3.register_adapter(np.ndarray, adapt_array)#检测类型自动转换

cursor = conn.cursor()
#创建表格
creat_tensor_table('ali_003',cursor)

#creat_tensor_table('flower_test',cursor)

#creat_tensor_table('flower_validation',cursor)

#开始插入表格
nn_file=len(file_and_label)

with tf.Session() as sess:

    for i in range(nn_file):
        db_jpgtonp_in(sess, 'ali_003', i,
            file_and_label[i][1],
            file_and_label[i][0], cursor)


# with tf.Session() as sess:
#     for i in range(nn_file):
#         if i<0.1*nn_file:
#             db_jpgtonp_in(sess,'flower_test', i,
#                           file_and_label[i][1],
#                           file_and_label[i][0], cursor)
#
#         elif i<0.2*nn_file:
#             db_jpgtonp_in(sess,'flower_validation', i,
#                           file_and_label[i][1],
#                           file_and_label[i][0], cursor)
#         else:
#             db_jpgtonp_in(sess,'flower_train', i,
#                           file_and_label[i][1],
#                           file_and_label[i][0],cursor)
#
#         print(i)
#         if i/500==0:
#             gc.collect()
#             conn.commit()

conn.commit()
conn.close()