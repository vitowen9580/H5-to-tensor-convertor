
'''
======================================
                unet
=======================================
'''
''''''
#region
"""------------------------keras-hdf5------------------------"""
# from keras.models import load_model
from tensorflow.keras.models import load_model
import tensorflow as tf #
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = tf.keras.layers.Flatten()(y_true)
    y_pred_f = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

#=========================================
#paramter
#==========================================
# Load the model 'Unet_' AttUnet_ ResUnet++_ ResUnet_
model_name='ResUnet_'


# keras_model = load_model('../resources/'+model_name+'.h5',custom_objects={'dice_loss': dice_loss,'dice_coef':dice_coef})
# [<tf.Tensor 'dense_2/Softmax:0' shape=(?, 10) dtype=float32>]
# # data
img_path = "test_img/0.png"
# img = cv2.imread(img_path, 0)
img = cv2.imread(img_path)

# # model
# unet = load_model(keras_model_path)
unet = load_model('../resources/'+model_name+'.h5',custom_objects={'dice_loss': dice_loss,'dice_coef':dice_coef})

# # prepare_and_predict
img = img / 255
img = cv2.resize(img, (256, 256))
# X = np.reshape(img, (1, 256, 256, 1))
X = np.reshape(img, (1, 256, 256, 3))

start_time = time.time()
y = unet.predict(X)
time_elapsed = time.time()-start_time
FPS=np.around(1/time_elapsed,2)
print('FPS:',FPS)
out = np.reshape(y, (256, 256)) * 255
out = np.array(out, dtype="u1")
print("img.shape: %s, out.shape: %s" % (img.shape, out.shape))

# # show
# plt.subplot(1, 2, 1); plt.imshow(img, cmap="gray")
# plt.subplot(1, 2, 2); plt.imshow(out, cmap="gray")
# plt.suptitle("keras-hdf5")
# plt.show()

# # """------------------------tensorflow-pb------------------------"""
# import tensorflow as tf
import tensorflow._api.v2.compat.v1 as tf

# model
tf_model_path = "./pb_model/"+model_name+".pb"

# graph_and_predict
with tf.Graph().as_default():
    output_graph_def = tf.GraphDef()

    # open the *.pb model
    with open(tf_model_path, "rb") as fr:
        output_graph_def.ParseFromString(fr.read())
        tf.import_graph_def(output_graph_def, name="")

    # run the forward in the session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  # init
        current_graph = tf.compat.v1.get_default_graph()

        graph_def = current_graph.as_graph_def()
        all_names = [n.name for n in graph_def.node]

        # print(all_names)
        # for tensor in tf.get_default_graph().as_graph_def().node:
        #     print(tensor.name)
        inp_tensor = sess.graph.get_tensor_by_name("x:0")
        out_tensor = sess.graph.get_tensor_by_name("Identity:0")
        # inp_tensor = 'input_1:0'
        # out_tensor = 'activation_20/Sigmoid:0'
        npy_tensor = sess.run(out_tensor, feed_dict={inp_tensor: X})  # X in line 18.
        # # postprocessing
        npy = np.reshape(npy_tensor, (256, 256)) * 255
        npy = np.array(npy, dtype="u1")

plt.subplot(1, 3, 1); plt.imshow(img, cmap="gray"), plt.title("IMG")
plt.subplot(1, 3, 2); plt.imshow(out, cmap="gray"), plt.title("keras")
plt.subplot(1, 3, 3); plt.imshow(npy, cmap="gray"), plt.title("TF")
plt.suptitle(model_name)

plt.show()
#endregion
''''''
