# Pokemon Classifier

This classifier is built by AndroidStudio with TFLite model which is trained in Python.

* [Dataset](https://github.com/tom1484/pokemon-classifier#dataset)
  * [Features](https://github.com/tom1484/pokemon-classifier#features)
  * [Download](https://github.com/tom1484/pokemon-classifier#download)

* [Model building](https://github.com/tom1484/pokemon-classifier#model-building)
  * [Prepare data](https://github.com/tom1484/pokemon-classifier#prepare-data)
  * [MobileNet](https://github.com/tom1484/pokemon-classifier#mobilenet)
  * [Callbacks](https://github.com/tom1484/pokemon-classifier#callbacks)
  * [Train & convert the model](https://github.com/tom1484/pokemon-classifier#train--convert-the-model)
  * [Visualize result](https://github.com/tom1484/pokemon-classifier#visualize-result)
* [Application](https://github.com/tom1484/pokemon-classifier#application)
  * [Features](https://github.com/tom1484/pokemon-classifier#features-1)
  * [Classification](https://github.com/tom1484/pokemon-classifier#classification)

## Dataset

The dataset we use to train the model is [Pokemon Classification](https://www.kaggle.com/lantian773030/pokemonclassification) from [Kaggle](https://www.kaggle.com). 

### Features

* Contains 150 classes of generation-one Pokemon
* Each class has 25 - 50 images of the Pokemon
* 6820 images in total

### Download

```bash
# since Kaggle has its own API to download dataset, I put its .zip file in google drive
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=17IWB7DLTFOR4_gRZoAzPTeoOQSbwSKIM' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=17IWB7DLTFOR4_gRZoAzPTeoOQSbwSKIM" -O PokemonData.zip && rm -rf /tmp/cookies.txt
```

## Model building

### Prepare data

Let's prepare our data by using **ImageDataGenerator** first.

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```

```python
# the generator will crop all images into this shape
image_shape = (224, 224)
# this means images are in RGB mode
channel = 3
full_image_shape = image_shape + (channel, )
batch_size = 32
num_classes = 150
```

```python
datagen = ImageDataGenerator(
    # makes all value in range between 0 and 1
    rescale=1. / 255,
    # we'll finally get 5511 images for training and 1309 for validation
    validation_split=0.2
)

# flow_from_directory reads only one batch of data each step, 
# this is helpful when dataset is too huge to be completely stored in memory
train_generator = datagen.flow_from_directory(
    directory='PokemonData',
    target_size=image_shape, 
    batch_size=batch_size, 
    subset='training'
)

val_generator = datagen.flow_from_directory(
    directory='PokemonData',
    target_size=image_shape, 
    batch_size=batch_size, 
    subset='validation'
)
```

```python
# the number of batches which is also the steps per epoch
num_train = len(train_generator)
num_val = len(val_generator)
```

### MobileNet

**MobileNet** is a architecture of large model that significantly reduces the parameters between layers, so that it is more suitable for mobile devices. You can see more details in its [paper](https://arxiv.org/pdf/1801.04381.pdf).

tf.keras provides various famous models, including **MobileNetV2**. It's very easy to use.

```python
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
```

```python
base_model = MobileNetV2(
    input_shape=full_image_shape, 
    alpha=1.0, 
    include_top=False, 
    # this is the pretrained weights used in images processing
    weights='imagenet', 
    input_tensor=Input(full_image_shape), 
    pooling=None, 
    classes=num_classes
)
```

```python
# snag the last layer of the imported model
x = base_model.layers[-1].output

x = GlobalMaxPooling2D()(x)
x = Dense(1024,activation='relu')(x)
x = Dense(num_classes, activation='softmax', name='predictions')(x)

# we can define the last few layers by ourselves
model = Model(inputs=base_model.input, outputs=x)

# let's train all the layers
for layer in model.layers:
    layer.training = True
```

```python
# compile the network
model.compile(
    optimizer=Adam(1e-4), 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)
```

### Callbacks

```python
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
```

```python
# these are utilities to maximize learning, while preventing over-fitting
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    patience=12, 
    cooldown=6, 
    rate=0.6, 
    min_lr=1e-8, 
    verbose=1
)
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=24, 
    verbose=1
)

# this save the best model which has the minimal validation loss
checkpoint = ModelCheckpoint(
    'best_model.h5', 
    monitor='val_loss', 
    mode='min', 
    save_best_only=True)
```

### Train & Convert the model

```python
# train the model for 200 epochs
history = model.fit_generator(
    train_generator, 
    validation_data=val_generator, 
    steps_per_epoch=num_train, 
    validation_steps=num_val, 
    epochs=200, 
    shuffle=True, 
    callbacks=[reduce_lr, early_stop, checkpoint]
)
```

```python
model.load_weights('best_model.h5')
# make a converter which converts our keras model into TFLite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# get the result
tflite_model = converter.convert()
# save the converted model
open('classfier.tflite', 'wb').write(tflite_model)
```

### Visualize result

```python
import matplotlib.pyplot as plt
```

```python
# plot training and validation iou_score values
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('accy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')

# Plot training and validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()
```

## Application

### Features

* Classifies up to 150 classes of Pokemon
* Supports camera preview with the newest [Camera2 API](https://developer.android.com/reference/android/hardware/camera2/package-summary)

### Classification

This application uses [Firebase](https://firebase.google.com/?hl=zh-tw) to process the model.

```java
private FirebaseModelInputOutputOptions inputOutputOptions;
private FirebaseModelInterpreter interpreter;
```

```java
// load model file
FirebaseCustomLocalModel localModel = new FirebaseCustomLocalModel.Builder()
    	.setAssetFilePath(modelFilename)
    	.build();

try {
    // create an intepreter for model
    FirebaseModelInterpreterOptions options =
        new FirebaseModelInterpreterOptions.Builder(localModel).build();
    interpreter = FirebaseModelInterpreter.getInstance(options);
	
    // create inputOutputOptions to set formats
    inputOutputOptions =
        new FirebaseModelInputOutputOptions.Builder()
        .setInputFormat(0, FirebaseModelDataType.FLOAT32, new int[]{1, inputSize, inputSize, 3})
        .setOutputFormat(0, FirebaseModelDataType.FLOAT32, new int[]{1, num_classes})
        .build();
} catch (FirebaseMLException e) {
    e.printStackTrace();
}
```

```java
// create input for intepreter
float[][][][] imgData = new float[1][224][224][3];
// put every bytes of image in imgData
for (int i = 0; i < inputSize; ++ i) {
    for (int j = 0; j < inputSize; ++j) {
        int pixelValue = intValues[i * inputSize + j];
        // 0xRRGGBB
        imgData[0][i][j][0] = (float) Color.red(pixelValue) / IMAGE_RESCALE;
        imgData[0][i][j][1] = (float) Color.green(pixelValue) / IMAGE_RESCALE;
        imgData[0][i][j][2] = (float) Color.blue(pixelValue) / IMAGE_RESCALE;
    }
}
```

```java
// run and show result
interpreter.run(inputs, inputOutputOptions)
        .addOnSuccessListener(
            result -> {
                float[][] outputs = result.getOutput(0);
                int prediction = getPrediction(outputs[0]);
                float probability = outputs[0][prediction];
                String message = labels.get(prediction) + "\n" + String.format("%.2f", probability * 100) + "%";

                // show result on the TextureView
                resultView.setTextSize(20);
                resultView.setTextColor(Color.WHITE);
                resultView.setText(message);
            });
```

### Result

See the result of application [here](https://drive.google.com/file/d/1HgIrvtW8sPcwJKx4yRg_zsS3dYYzUViN/view?usp=sharing).