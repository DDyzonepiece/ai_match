import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets

def reporthook(a,b,c):
    print('downloading: %5.1f%%'%(a*b*100/c),end='')
#从TF-slim图像分类库中加载预先训练的网络
tf.logging.set_verbosity(tf.logging.ERROR)
sess = tf.InteractiveSession()
#设置输入图像。使用tf.Variable而不是使用tf.placeholder，这是因为要确保它是可训练的。当我们需要时，仍然可以输入它。
image = tf.Variable(tf.zeros((299, 299, 3)))
#接下来，加载Inception v3模型
def inception(image, reuse):
    # multiply元素相乘，subtract元素级减法操作，expand_dims给图片增加一个维度
    preprocessed = tf.multiply(tf.subtract(tf.expand_dims(image, 0), 0.5), 2.0)#这步就是给图像做个预处理
    arg_scope = nets.inception.inception_v3_arg_scope(weight_decay=0.0)
    with slim.arg_scope(arg_scope):
        logits, _ = nets.inception.inception_v3(
            preprocessed, 1001, is_training=False, reuse=reuse)
        logits = logits[:,1:] # ignore background class
        probs = tf.nn.softmax(logits) # probabilities
    return logits, probs

logits, probs = inception(image, reuse=False)

#加载预训练的权重
import tempfile
from urllib.request import urlretrieve
import tarfile
import os
data_dir = tempfile.mkdtemp()
inception_tarball=r'download\inception_v3_2016_08_28.tar.gz'
tarfile.open(inception_tarball, 'r:gz').extractall(data_dir)
restore_vars = [
    var for var in tf.global_variables()
    if var.name.startswith('InceptionV3/')
]
saver = tf.train.Saver(restore_vars)
saver.restore(sess, os.path.join(data_dir, 'inception_v3.ckpt'))

#编写一些代码来显示图像，并对它进行分类及显示分类结果
import json
import matplotlib.pyplot as plt
#json里面就是标签列表
imagenet_json=r'download\imagenet.json'
with open(imagenet_json) as f:
    imagenet_labels = json.load(f)


def classify(img, correct_class=None, target_class=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
    fig.sca(ax1)
    p = sess.run(probs, feed_dict={image: img})[0]
    ax1.imshow(img)
    fig.sca(ax1)

    topk = list(p.argsort()[-10:][::-1])
    topprobs = p[topk]
    barlist = ax2.bar(range(10), topprobs)
    if target_class in topk:
        barlist[topk.index(target_class)].set_color('r')
    if correct_class in topk:
        barlist[topk.index(correct_class)].set_color('g')
    plt.sca(ax2)
    plt.ylim([0, 1.1])
    plt.xticks(range(10),
               [imagenet_labels[i][:15] for i in topk],
               rotation='vertical')
    fig.subplots_adjust(bottom=0.2)
    plt.show()

#加载示例图像，并确保它已被正确分类
import PIL
import numpy as np
img_path=r'download\cat.jpg'
img_class = 281
img = PIL.Image.open(img_path)
big_dim = max(img.width, img.height)
wide = img.width > img.height
new_w = 299 if not wide else int(img.width * 299 / img.height)
new_h = 299 if wide else int(img.height * 299 / img.width)
img = img.resize((new_w, new_h)).crop((0, 0, 299, 299))
img = (np.asarray(img) / 255.0).astype(np.float32)
classify(img, correct_class=img_class)



x = tf.placeholder(tf.float32, (299, 299, 3))

x_hat = image # our trainable adversarial input
assign_op = tf.assign(x_hat, x)

learning_rate = tf.placeholder(tf.float32, ())
y_hat = tf.placeholder(tf.int32, ())

labels = tf.one_hot(y_hat, 1000)
loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=[labels])
optim_step = tf.train.GradientDescentOptimizer(
    learning_rate).minimize(loss, var_list=[x_hat])

epsilon = tf.placeholder(tf.float32, ())

below = x - epsilon
above = x + epsilon
projected = tf.clip_by_value(tf.clip_by_value(x_hat, below, above), 0, 1)
with tf.control_dependencies([projected]):
    project_step = tf.assign(x_hat, projected)

demo_epsilon = 2.0 / 255.0  # a really small perturbation
demo_lr = 1e-1
demo_steps = 100
demo_target = 924  # "guacamole"

# initialization step
sess.run(assign_op, feed_dict={x: img})

# projected gradient descent
for i in range(demo_steps):
    # gradient descent step
    _, loss_value = sess.run(
        [optim_step, loss],
        feed_dict={learning_rate: demo_lr, y_hat: demo_target})
    # project step
    sess.run(project_step, feed_dict={x: img, epsilon: demo_epsilon})
    if (i + 1) % 10 == 0:
        print('step %d, loss=%g' % (i + 1, loss_value))

adv = x_hat.eval()  # retrieve the adversarial example

classify(adv, correct_class=img_class, target_class=demo_target)
