import tensorflow as tf

def build_Unet(K,stages,filters,input_size,type,last_layer_activation):

  #k_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02, seed=None)

  concats=[]
  
  if (type=='encoder'):
    inputs =tf.keras.Input(shape=(input_size,input_size,3))
  else:
    inputs =tf.keras.Input(shape=(input_size,input_size,K))

  x = inputs
  for i in stages:
    x= tf.keras.layers.Conv2D(filters=filters,kernel_size=(3,3),strides=(1,1),padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x= tf.keras.layers.ReLU()(x)

    x= tf.keras.layers.Conv2D(filters=filters,kernel_size=(3,3),strides=(1,1),padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x= tf.keras.layers.ReLU()(x)

    concats.append(x)



    x = tf.keras.layers.MaxPool2D(pool_size=(2,2),padding='same')(x)
    
    filters *=2
    
    


  x= tf.keras.layers.Conv2D(filters=filters,kernel_size=(3,3),strides=(1,1),padding='same')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x= tf.keras.layers.ReLU()(x)

  x= tf.keras.layers.Conv2D(filters=filters,kernel_size=(3,3),strides=(1,1),padding='same')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x= tf.keras.layers.ReLU()(x)

  for i in stages:

    filters /=2

    u = tf.keras.layers.Conv2DTranspose(filters=filters,kernel_size=(2,2),strides=(2,2),padding='valid')(x)
    y= concats[-i]
    
    x = tf.keras.layers.Concatenate(axis=3)([y, u])
    
    x= tf.keras.layers.Conv2D(filters=filters,kernel_size=(3,3),strides=(1,1),padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x= tf.keras.layers.ReLU()(x)

    x= tf.keras.layers.Conv2D(filters=filters,kernel_size=(3,3),strides=(1,1),padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x= tf.keras.layers.ReLU()(x)

  if(type == 'encoder'):
    x = tf.keras.layers.Conv2D(filters=K,kernel_size=(1,1),strides=(1,1),padding='same')(x)
   
  else:
    x = tf.keras.layers.Conv2D(filters=3,kernel_size=(1,1),strides=(1,1),padding='same')(x)
  x= tf.keras.layers.Activation(last_layer_activation)(x)

  model = tf.keras.Model(inputs=inputs, outputs=x)


  return model

  