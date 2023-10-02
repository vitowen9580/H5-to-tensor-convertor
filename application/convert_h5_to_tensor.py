from tensorflow.keras.models import load_model
import tensorflow as tf

smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = tf.keras.layers.Flatten()(y_true)
    y_pred_f = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)
# Load the model
#=========================================
#paramter
#==========================================
model_name='ResUnet_'
keras_model = load_model('../resources/'+model_name+'.h5',custom_objects={'dice_loss': dice_loss,'dice_coef':dice_coef})

keras_model.save('model_tf', save_format='tf')

# Load the model
loaded_model = tf.keras.models.load_model('model_tf',custom_objects={'dice_loss': dice_loss,'dice_coef':dice_coef})

# Check its architecture
loaded_model.summary()
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

full_model = tf.function(lambda x: loaded_model(x))
full_model = full_model.get_concrete_function(
    tf.TensorSpec(loaded_model.inputs[0].shape, loaded_model.inputs[0].dtype))

# Get frozen ConcreteFunction
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()

# Save the frozen graph from the frozen ConcreteFunction to disk
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir="./pb_model",
                  name=model_name+".pb",
                  as_text=False)