import pandas as pd
import tensorflow as tf
import numpy as np
import datetime
from tensorflow.keras.optimizers import RMSprop
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import TensorBoard

xx=256
nEpochs=10
updatemodel=2
m_name='train'
model = tf.keras.models.Sequential([
	tf.keras.layers.Dense(128, activation='relu',input_shape=(xx,xx,3)),
	tf.keras.layers.Dropout(rate=1-0.7),
	tf.keras.layers.Conv2D(32,(128,128)),
	tf.keras.layers.Dense(128, activation='relu'),
	tf.keras.layers.Dropout(rate=1-0.7),
	tf.keras.layers.Conv2D(32,(64,64)),
	tf.keras.layers.GaussianNoise(0.1),
	tf.keras.layers.Dense(128, activation='relu'),
	tf.keras.layers.Conv2D(32,(32,32)),
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(128, activation='relu'),
	tf.keras.layers.Dense(43, activation='softmax')
	])

now = datetime.datetime.now()
newDirName = now.strftime("logs"+m_name+"_%d-%H_%M")
print ("Making directory: " + newDirName)
tensorboard=TensorBoard(write_images=True,
						write_graph=True,
#						histogram_freq=50,
						log_dir=newDirName+now.strftime('%S'))
model.compile(loss='categorical_crossentropy',
			  optimizer=RMSprop(lr=0.001),
			  metrics=['acc'])

#x_train=x_train1[:,:,:,np.newaxis]
def append_ext(fn):
    return fn+'.jpg'
#y_train = to_categorical(y_train,num_classes=2)
#validation_datagen =ImageDataGenerator(rescale=1/255)
train_datagen = ImageDataGenerator(rescale=1/255)
#								   validation_split=0.2,
#	rotation_range=360,
#	vertical_flip=True,
#	horizontal_flip=True,

dff=pd.read_excel(r"train1.xlsx", encoding = 'utf-8')
dff=dff.astype(str)
dff['image_id']=dff['image_id'].apply(append_ext)

train_generator=train_datagen.flow_from_dataframe(dataframe=dff,directory="./data/train1",
												  has_ext=False,
												  x_col='image_id',y_col='category',
												  classmode='categorical',
												  target_size=(256,256),
												  batch_size=32)

class myCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs={}):
		if(logs.get('acc')>0.97 or logs.get('val_loss')<0.050):
			print("\nReached 98% accuracy so cancelling training!")
			self.model.stop_training = True
if updatemodel==1:
	from keras.models import load_model
	from keras.utils import CustomObjectScope
	from keras.initializers import glorot_uniform
	with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
		model=load_model(' '+m_name+'.h5')
		print('model loaded successfully')

callback = myCallback()
history = model.fit_generator(
	  train_generator,
	  steps_per_epoch=20,
	  epochs=nEpochs,
      validation_steps=5,
	  verbose=1,
	  callbacks=[tensorboard,callback])
model.summary()
if updatemodel==1:
	model.save('models/'+m_name+'update.h5')
else:
	model.save('models/'+m_name+'.h5')

del m_name,nEpochs,x_train,x_train1,y_train,xx
print('------------------------model saved-----------------------------')


def updatemodel1(a,b,model1,init_epoch=50,end=70):
	import keras
	from keras.utils import to_categorical

	from keras.models import load_model
	from keras.utils import CustomObjectScope
	from keras.initializers import glorot_uniform
	with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
		model=load_model('models/'+model1+'.h5')

	a=a[:,:,:,np.newaxis]
	b = to_categorical(b,num_classes=2)
	history=model.fit(a,b,steps_per_epoch=(int(len(a)/15)),
	epochs=end,
	initial_epoch=init_epoch,
	validation_steps=5,
	verbose=1,
	shuffle=True,
	callbacks=[tensorboard])
	model.summary()
	return model

def predict(ipath):

	import keras
	from keras.models import load_model
	from keras.utils import CustomObjectScope
	from keras.initializers import glorot_uniform
	with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
		model=load_model('models/new_begining3.h5')
# =============================================================================
# 		model=load_model('models/model_dense_testtesting.h5')
# =============================================================================
	import numpy as np
	x = np.expand_dims(ipath, axis=3)
	# images = np.vstack([x])
	classes = model.predict(x, batch_size=40)
	print('prediction completed successfully')
	print(classes)
	#print(len(classes))
	return classes

def plot(history):
	import matplotlib.pyplot as plt
	acc = history.history['acc']
	#val_acc = history.history['val_acc']
	loss = history.history['loss']
	#val_loss = history.history['val_loss']
	epochs = range(len(acc))
	plt.plot(epochs, acc, 'r', label='Training accuracy')
	#plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
	plt.title('Training and validation accuracy')
	plt.legend(loc=0)
	plt.figure()
	plt.show()
