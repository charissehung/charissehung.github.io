---
layout: post
title: Identifying Fake News with TensorFlow
---

In this post, I'll explain how to create a fake news classifier using TensorFlow. We will be creating three models to predict whether news articles contain fake news, and then evaluate them.

First, let us download the necessary packages. As we can see, we will need many functionalities of TensorFlow.

```python
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses

import pandas as pd
import re
import string
from matplotlib import pyplot as plt
import numpy as np
import plotly.express as px

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
```

## Acquire Training Data

The data that we will use to train the model can be found at the link below.

```python
train_url = "https://github.com/PhilChodrow/PIC16b/blob/master/datasets/fake_news_train.csv?raw=true"
```

Let's read this data into a pandas dataframe.

```python
train_data = pd.read_csv(train_url)
train_data
```

PUT DATAFRAME HERE


We see that this dataset contains information about 22449 articles. Each row represents an article and there are three columns that contain the title, the full article text, and a boolean value that indicates whether the article is fake or not. This fake column is 0 when the article is true and 1 when the article contains fake news.


## Create a Dataset

We're going to create a function called `make_dataset`. This will convert our pandas dataframe to a TensorFlow Dataset. This allows us to stay organized while creating our data pipeline. In this function, we will remove stop words from the title and text columns of each article. Stop words in English are words like "the", "and", or "but". Then we will define the input and output components of our dataset. We want the model to evaluate the title and text of the article, so these are the inputs. The output will be 0 or 1, depending if the article contains fake news or not.

```python
def make_dataset(train_data):
  stop = stopwords.words('english')

  # remove stopwords from titles
  train_data['title'] = train_data['title'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

  # remove stopwords from text
  train_data['text'] = train_data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

  # create TensorFlow dataset
  data = tf.data.Dataset.from_tensor_slices(
    (
        {
            "title" : train_data[["title"]],
            "text" : train_data["text"]
        },
        {
            "fake" : train_data[["fake"]]
        }
    )
)
  data.batch(100) #batching the data allows it to efficiently train in chunks
  return data
```

Let's use our `make_dataset` function on our pandas dataframe.

```python
data = make_dataset(train_data)
```

Now let's split our data into training and validation sets. We will use 80% for training and 20% for validation. Again, we will batch the data so it will train more efficiently.

```python
data = data.shuffle(buffer_size = len(data))

train_size = int(0.8 * len(data)) # allocate 80% of data for training

train = data.take(train_size).batch(100) # set training data
val = data.skip(train_size).batch(100) # set validation data
```

We can check to see the size of the sets `train` and `val`.

```python
len(train), len(val)
```

```python
(180, 45)
```

Here we see that the training data is four times larger than the validation data, which is what we expect. Because we batched the data, each value here actually represents 100 articles. Thus this sums to 22500 articles, which is roughly the amount of articles in our original pandas dataframe.

## Preprocessing

Before we get into the models, we need to do some data cleaning. The first step is to standardize the data. We will accomplish this by turning all the text to lower case and then removing any punctuation. This is where we use the `re` (regular expression) and `string` libraries.

```python
def standardization(input_data):
    lowercase = tf.strings.lower(input_data) # all letters to lowercase
    no_punctuation = tf.strings.regex_replace(lowercase,
                                  '[%s]' % re.escape(string.punctuation),'') # remove punctuation
    return no_punctuation
```

Next, we will vectorize the data. In this case, we will convert our text data into numbers so that our models will be able to understand them. The number associated with each word represents how frequently it appears in the dataset. This means a word with a vectorized value of 347 is the 347th most popular word in the data. I have arbitrarily decided to only consider the top 5000 words in the dataset.

```python
size_vocabulary = 5000

vectorize_layer = TextVectorization(
    standardize=standardization,
    max_tokens=size_vocabulary, # only consider this many words
    output_mode='int',
    output_sequence_length=500)
```

Let's apply the vectorization to the `title` and `text` columns of the dataset!

```python
vectorize_layer.adapt(train.map(lambda x, y: x["title"]))
vectorize_layer.adapt(train.map(lambda x, y: x["text"]))
```

As a last step before creating the models, we will define the input types of the data. The inputs `title` and `text` are both one dimensional vectors of strings, and this is reflected in the following code.

```python
title_input = keras.Input(
    shape = (1,),
    name = "title",
    dtype = "string"
)

text_input = keras.Input(
    shape = (1,),
    name = "text",
    dtype = "string"
)
```

## Creating the Title Model

The first model we will create will determine whether an article contains fake news or not solely based on its title. To do so, we will use the functional API of TensorFlow. We will define a pipeline of hidden layers to process the titles. First, we explicitly define an embedding layer. This is so we can reuse it in a later model. Then we define different layers for the text data. The dropout layers prevent overfitting.

```python
embedding = layers.Embedding(size_vocabulary, 12, name = "embedding") # define an embedding layer

title_features = vectorize_layer(title_input) # vectorize the title inputs
title_features = embedding(title_features) # use the defined embedding layer
title_features = layers.Dropout(0.2)(title_features) # prevent overfitting
title_features = layers.GlobalAveragePooling1D()(title_features)
title_features = layers.Dropout(0.2)(title_features) # prevent overfitting
title_features = layers.Dense(32, activation='relu')(title_features)
```

Now we will add an additional dense layer and then define the output. Recall, we want the output to be 0 or 1 depending on whether the article contains fake news. Since there are only 2 options for the output, the number of units of the layer is 2, and notice we name it `fake`.

```python
main_title = title_features
main_title = layers.Dense(32, activation='relu')(main_title)
output_title = layers.Dense(2, name = "fake")(main_title)
```

Now let's put the model together by defining the inputs and outputs.

```python
model_title = keras.Model(
    inputs = title_input,
    outputs = output_title
)
```

Let's look at our model summary to understand how the layers are working together.

```python
model_title.summary()
```

```
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
title (InputLayer)           [(None, 1)]               0         
_________________________________________________________________
text_vectorization (TextVect (None, 500)               0         
_________________________________________________________________
embedding (Embedding)        (None, 500, 12)           60000     
_________________________________________________________________
dropout (Dropout)            (None, 500, 12)           0         
_________________________________________________________________
global_average_pooling1d (Gl (None, 12)                0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 12)                0         
_________________________________________________________________
dense (Dense)                (None, 32)                416       
_________________________________________________________________
dense_1 (Dense)              (None, 32)                1056      
_________________________________________________________________
fake (Dense)                 (None, 2)                 66        
=================================================================
Total params: 61,538
Trainable params: 61,538
Non-trainable params: 0
```

This may seem confusing, so we can look at a diagram that represents the layers of the model.

```python
keras.utils.plot_model(model_title)
```

INSEST MODEL DIAGRAM HERE

We can see how the title is vectorized, passed through the embedding layer, and then continues through multiple layers until the end when it outputs `fake`.

Let's compile `model_title`.

```python
model_title.compile(optimizer = "adam",
              loss = losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']
)
```

Now we can run `model_title`. Notice we use the training and the validation data. The number of epochs represents how many times the model is trained.

```python
history_title = model_title.fit(train,
                    validation_data=val,
                    epochs = 20)
```

```
Epoch 1/20
/usr/local/lib/python3.7/dist-packages/tensorflow/python/keras/engine/functional.py:595: UserWarning: Input dict contained keys ['text'] which did not match any model input. They will be ignored by the model.
  [n for n in tensors.keys() if n not in ref_input_names])
180/180 [==============================] - 5s 21ms/step - loss: 0.6925 - accuracy: 0.5135 - val_loss: 0.6870 - val_accuracy: 0.5276
Epoch 2/20
180/180 [==============================] - 3s 16ms/step - loss: 0.6411 - accuracy: 0.6493 - val_loss: 0.2182 - val_accuracy: 0.9508
Epoch 3/20
180/180 [==============================] - 3s 16ms/step - loss: 0.1736 - accuracy: 0.9507 - val_loss: 0.1064 - val_accuracy: 0.9615
Epoch 4/20
180/180 [==============================] - 3s 16ms/step - loss: 0.0946 - accuracy: 0.9681 - val_loss: 0.0925 - val_accuracy: 0.9675
Epoch 5/20
180/180 [==============================] - 3s 15ms/step - loss: 0.0807 - accuracy: 0.9736 - val_loss: 0.0728 - val_accuracy: 0.9748
Epoch 6/20
180/180 [==============================] - 3s 16ms/step - loss: 0.0676 - accuracy: 0.9761 - val_loss: 0.0546 - val_accuracy: 0.9811
Epoch 7/20
180/180 [==============================] - 3s 17ms/step - loss: 0.0521 - accuracy: 0.9819 - val_loss: 0.0460 - val_accuracy: 0.9829
Epoch 8/20
180/180 [==============================] - 3s 17ms/step - loss: 0.0586 - accuracy: 0.9788 - val_loss: 0.0435 - val_accuracy: 0.9855
Epoch 9/20
180/180 [==============================] - 3s 17ms/step - loss: 0.0491 - accuracy: 0.9820 - val_loss: 0.0306 - val_accuracy: 0.9893
Epoch 10/20
180/180 [==============================] - 3s 16ms/step - loss: 0.0526 - accuracy: 0.9808 - val_loss: 0.0380 - val_accuracy: 0.9889
Epoch 11/20
180/180 [==============================] - 3s 16ms/step - loss: 0.0396 - accuracy: 0.9867 - val_loss: 0.0269 - val_accuracy: 0.9915
Epoch 12/20
180/180 [==============================] - 3s 16ms/step - loss: 0.0370 - accuracy: 0.9870 - val_loss: 0.0298 - val_accuracy: 0.9909
Epoch 13/20
180/180 [==============================] - 3s 17ms/step - loss: 0.0375 - accuracy: 0.9879 - val_loss: 0.0258 - val_accuracy: 0.9927
Epoch 14/20
180/180 [==============================] - 3s 16ms/step - loss: 0.0312 - accuracy: 0.9895 - val_loss: 0.0303 - val_accuracy: 0.9920
Epoch 15/20
180/180 [==============================] - 3s 17ms/step - loss: 0.0331 - accuracy: 0.9892 - val_loss: 0.0259 - val_accuracy: 0.9920
Epoch 16/20
180/180 [==============================] - 3s 17ms/step - loss: 0.0270 - accuracy: 0.9912 - val_loss: 0.0268 - val_accuracy: 0.9927
Epoch 17/20
180/180 [==============================] - 3s 16ms/step - loss: 0.0243 - accuracy: 0.9927 - val_loss: 0.0272 - val_accuracy: 0.9906
Epoch 18/20
180/180 [==============================] - 3s 16ms/step - loss: 0.0259 - accuracy: 0.9915 - val_loss: 0.0187 - val_accuracy: 0.9947
Epoch 19/20
180/180 [==============================] - 3s 17ms/step - loss: 0.0245 - accuracy: 0.9921 - val_loss: 0.0205 - val_accuracy: 0.9947
Epoch 20/20
180/180 [==============================] - 3s 16ms/step - loss: 0.0238 - accuracy: 0.9922 - val_loss: 0.0163 - val_accuracy: 0.9947
```            

After 20 epochs, the model is has over 99% accuracy on both the training and validation data. This is great! Let's visualize the accuracy over time.

```python
plt.plot(history_title.history["accuracy"], label = "training")
plt.plot(history_title.history["val_accuracy"], label = "validation")
plt.gca().set(xlabel = "epoch", ylabel = "accuracy")
plt.legend()
```

INSERT GRAPH Here

We can see that the training and validation data appear to have similar accuracy, which indicates we did not overfit the model. Additionally, the accuracies of both have "leveled off" so we probably would not benefit from more training. Our first model is complete!

## Creating the Text Model

For the second model, we will determine whether an article contains fake news or not solely based on the article text. This process will be very similar to the process above for creating the title model. We begin by creating some layers for the text input. Notice we use the embedding layer that we defined above.

```python
text_features = vectorize_layer(text_input) # vectorize the text
text_features = embedding(text_features) #use defined embedding layer
text_features = layers.Dropout(0.2)(text_features) # prevent overfitting
text_features = layers.GlobalAveragePooling1D()(text_features)
text_features = layers.Dropout(0.2)(text_features) # prevent overfitting
text_features = layers.Dense(32, activation='relu')(text_features)
```

Again, we add another dense layer, and then create the output layer which only has 2 possible values. Then we put the model together by defining its inputs and outputs.

```python
main_text = text_features
main_text = layers.Dense(32, activation='relu')(main_text)
output_text = layers.Dense(2, name = "fake")(main_text) # output layer

# create model
model_text = keras.Model(
    inputs = text_input,
    outputs = output_text
)
```

The summary of the model shows its different layers and details regarding the shape of the data and how many parameters there are.

```python
model_text.summary()
```

```
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
text (InputLayer)            [(None, 1)]               0         
_________________________________________________________________
text_vectorization (TextVect (None, 500)               0         
_________________________________________________________________
embedding (Embedding)        (None, 500, 12)           60000     
_________________________________________________________________
dropout_2 (Dropout)          (None, 500, 12)           0         
_________________________________________________________________
global_average_pooling1d_1 ( (None, 12)                0         
_________________________________________________________________
dropout_3 (Dropout)          (None, 12)                0         
_________________________________________________________________
dense_2 (Dense)              (None, 32)                416       
_________________________________________________________________
dense_3 (Dense)              (None, 32)                1056      
_________________________________________________________________
fake (Dense)                 (None, 2)                 66        
=================================================================
Total params: 61,538
Trainable params: 61,538
Non-trainable params: 0
```

Let's also look at the diagram to better understand the structure of the model.

```python
keras.utils.plot_model(model_text)
```

INSERT MODEL DIAGRAM HERE

This diagram is very similar to the one for `model_title` since both have the same structure.

Now let's compile our model and train it.

```python
# compile the model
model_text.compile(optimizer = "adam",
              loss = losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']
)

# train the model
history_text = model_text.fit(train,
                    validation_data=val,
                    epochs = 20)
```

```
Epoch 1/20
/usr/local/lib/python3.7/dist-packages/tensorflow/python/keras/engine/functional.py:595: UserWarning: Input dict contained keys ['title'] which did not match any model input. They will be ignored by the model.
  [n for n in tensors.keys() if n not in ref_input_names])
180/180 [==============================] - 5s 26ms/step - loss: 0.5729 - accuracy: 0.7164 - val_loss: 0.2078 - val_accuracy: 0.9341
Epoch 2/20
180/180 [==============================] - 4s 24ms/step - loss: 0.1857 - accuracy: 0.9358 - val_loss: 0.1392 - val_accuracy: 0.9604
Epoch 3/20
180/180 [==============================] - 4s 25ms/step - loss: 0.1451 - accuracy: 0.9579 - val_loss: 0.1103 - val_accuracy: 0.9715
Epoch 4/20
180/180 [==============================] - 5s 25ms/step - loss: 0.1024 - accuracy: 0.9719 - val_loss: 0.0910 - val_accuracy: 0.9746
Epoch 5/20
180/180 [==============================] - 5s 25ms/step - loss: 0.0864 - accuracy: 0.9749 - val_loss: 0.0700 - val_accuracy: 0.9826
Epoch 6/20
180/180 [==============================] - 5s 25ms/step - loss: 0.0731 - accuracy: 0.9818 - val_loss: 0.0550 - val_accuracy: 0.9846
Epoch 7/20
180/180 [==============================] - 5s 25ms/step - loss: 0.0649 - accuracy: 0.9839 - val_loss: 0.0453 - val_accuracy: 0.9895
Epoch 8/20
180/180 [==============================] - 5s 25ms/step - loss: 0.0526 - accuracy: 0.9873 - val_loss: 0.0471 - val_accuracy: 0.9889
Epoch 9/20
180/180 [==============================] - 5s 25ms/step - loss: 0.0485 - accuracy: 0.9888 - val_loss: 0.0340 - val_accuracy: 0.9927
Epoch 10/20
180/180 [==============================] - 5s 25ms/step - loss: 0.0378 - accuracy: 0.9915 - val_loss: 0.0317 - val_accuracy: 0.9933
Epoch 11/20
180/180 [==============================] - 5s 25ms/step - loss: 0.0319 - accuracy: 0.9927 - val_loss: 0.0256 - val_accuracy: 0.9951
Epoch 12/20
180/180 [==============================] - 5s 25ms/step - loss: 0.0272 - accuracy: 0.9935 - val_loss: 0.0210 - val_accuracy: 0.9962
Epoch 13/20
180/180 [==============================] - 5s 25ms/step - loss: 0.0289 - accuracy: 0.9931 - val_loss: 0.0189 - val_accuracy: 0.9969
Epoch 14/20
180/180 [==============================] - 5s 25ms/step - loss: 0.0218 - accuracy: 0.9959 - val_loss: 0.0153 - val_accuracy: 0.9976
Epoch 15/20
180/180 [==============================] - 5s 25ms/step - loss: 0.0194 - accuracy: 0.9951 - val_loss: 0.0132 - val_accuracy: 0.9973
Epoch 16/20
180/180 [==============================] - 5s 25ms/step - loss: 0.0229 - accuracy: 0.9943 - val_loss: 0.0141 - val_accuracy: 0.9973
Epoch 17/20
180/180 [==============================] - 5s 25ms/step - loss: 0.0157 - accuracy: 0.9961 - val_loss: 0.0130 - val_accuracy: 0.9976
Epoch 18/20
180/180 [==============================] - 5s 25ms/step - loss: 0.0124 - accuracy: 0.9969 - val_loss: 0.0078 - val_accuracy: 0.9987
Epoch 19/20
180/180 [==============================] - 5s 25ms/step - loss: 0.0128 - accuracy: 0.9969 - val_loss: 0.0077 - val_accuracy: 0.9987
Epoch 20/20
180/180 [==============================] - 5s 25ms/step - loss: 0.0152 - accuracy: 0.9954 - val_loss: 0.0104 - val_accuracy: 0.9976
```

Again, we can see that the model achieves over 99% accuracy on the training and validation data. Let's plot its progress.


```python
plt.plot(history_text.history["accuracy"], label = "training")
plt.plot(history_text.history["val_accuracy"], label = "validation")
plt.gca().set(xlabel = "epoch", ylabel = "accuracy")
plt.legend()
```

INSERT SECOND GRAPH Here

We can see that the validation data has a slightly higher accuracy than the training data, so we did not overfit the model. Both lines have "leveled off" so we are done training this model!

## Combined Model

The last model will use both the title and text of the article to determine whether the article contains fake news. Since we have already defined the layers for `title_features` and `text_features`, we can combine them to create our new model.

```python
main = layers.concatenate([title_features, text_features], axis = 1) # combine the layers of title and text
```

We will add another dense layer and then create our output layer. Then we can create the model by defining the inputs and outputs. Since we have two inputs this time, `title_input` and `text_input` are passed as a list.

```python
main = layers.Dense(32, activation = 'relu')(main)
output = layers.Dense(2, name = "fake")(main)

model = keras.Model(
    inputs = [title_input, text_input],
    outputs = output
)
```

Let's look at the summary for the model.

```python
model.summary()
```

```
Model: "model_2"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
title (InputLayer)              [(None, 1)]          0                                            
__________________________________________________________________________________________________
text (InputLayer)               [(None, 1)]          0                                            
__________________________________________________________________________________________________
text_vectorization (TextVectori (None, 500)          0           title[0][0]                      
                                                                 text[0][0]                       
__________________________________________________________________________________________________
embedding (Embedding)           (None, 500, 12)      60000       text_vectorization[0][0]         
                                                                 text_vectorization[1][0]         
__________________________________________________________________________________________________
dropout (Dropout)               (None, 500, 12)      0           embedding[0][0]                  
__________________________________________________________________________________________________
dropout_2 (Dropout)             (None, 500, 12)      0           embedding[1][0]                  
__________________________________________________________________________________________________
global_average_pooling1d (Globa (None, 12)           0           dropout[0][0]                    
__________________________________________________________________________________________________
global_average_pooling1d_1 (Glo (None, 12)           0           dropout_2[0][0]                  
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 12)           0           global_average_pooling1d[0][0]   
__________________________________________________________________________________________________
dropout_3 (Dropout)             (None, 12)           0           global_average_pooling1d_1[0][0]
__________________________________________________________________________________________________
dense (Dense)                   (None, 32)           416         dropout_1[0][0]                  
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 32)           416         dropout_3[0][0]                  
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 64)           0           dense[0][0]                      
                                                                 dense_2[0][0]                    
__________________________________________________________________________________________________
dense_4 (Dense)                 (None, 32)           2080        concatenate[0][0]                
__________________________________________________________________________________________________
fake (Dense)                    (None, 2)            66          dense_4[0][0]                    
==================================================================================================
Total params: 62,978
Trainable params: 62,978
Non-trainable params: 0
```

There is a lot goin on here. Notice that unlike the previous two models, there is an additional column that shows how the layers are connected. Let's visualize this.

```python
keras.utils.plot_model(model)
```

INSERT MODEL structure

Notice how both the text and title go through the same embedding layer. Recall, we defined our own embedding layer and then had both the title and text use it. Another cool thing to note is how the two "branches" of layers come together at the concatenate layer. We explicitly did this when we created `main`. Let's compile our model and train it.

```python
# compule the model
model.compile(optimizer = "adam",
              loss = losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']
)

# train the model
history = model.fit(train,
                    validation_data=val,
                    epochs = 20)
```

```
Epoch 1/20
180/180 [==============================] - 8s 39ms/step - loss: 0.3635 - accuracy: 0.8894 - val_loss: 0.0369 - val_accuracy: 0.9978
Epoch 2/20
180/180 [==============================] - 7s 37ms/step - loss: 0.0299 - accuracy: 0.9975 - val_loss: 0.0136 - val_accuracy: 0.9987
Epoch 3/20
180/180 [==============================] - 7s 37ms/step - loss: 0.0148 - accuracy: 0.9978 - val_loss: 0.0099 - val_accuracy: 0.9987
Epoch 4/20
180/180 [==============================] - 7s 37ms/step - loss: 0.0087 - accuracy: 0.9989 - val_loss: 0.0080 - val_accuracy: 0.9987
Epoch 5/20
180/180 [==============================] - 7s 37ms/step - loss: 0.0065 - accuracy: 0.9991 - val_loss: 0.0050 - val_accuracy: 0.9987
Epoch 6/20
180/180 [==============================] - 7s 37ms/step - loss: 0.0060 - accuracy: 0.9987 - val_loss: 0.0027 - val_accuracy: 0.9996
Epoch 7/20
180/180 [==============================] - 7s 38ms/step - loss: 0.0035 - accuracy: 0.9992 - val_loss: 0.0029 - val_accuracy: 0.9998
Epoch 8/20
180/180 [==============================] - 7s 38ms/step - loss: 0.0036 - accuracy: 0.9994 - val_loss: 0.0018 - val_accuracy: 0.9998
Epoch 9/20
180/180 [==============================] - 7s 38ms/step - loss: 0.0042 - accuracy: 0.9990 - val_loss: 0.0016 - val_accuracy: 0.9996
Epoch 10/20
180/180 [==============================] - 7s 37ms/step - loss: 0.0031 - accuracy: 0.9993 - val_loss: 0.0023 - val_accuracy: 0.9998
Epoch 11/20
180/180 [==============================] - 7s 37ms/step - loss: 0.0021 - accuracy: 0.9992 - val_loss: 7.6552e-04 - val_accuracy: 1.0000
Epoch 12/20
180/180 [==============================] - 7s 38ms/step - loss: 0.0019 - accuracy: 0.9996 - val_loss: 6.2970e-04 - val_accuracy: 1.0000
Epoch 13/20
180/180 [==============================] - 7s 37ms/step - loss: 0.0010 - accuracy: 0.9998 - val_loss: 7.9002e-04 - val_accuracy: 0.9998
Epoch 14/20
180/180 [==============================] - 7s 38ms/step - loss: 0.0014 - accuracy: 0.9997 - val_loss: 7.3809e-04 - val_accuracy: 0.9998
Epoch 15/20
180/180 [==============================] - 7s 37ms/step - loss: 8.2648e-04 - accuracy: 0.9998 - val_loss: 0.0026 - val_accuracy: 0.9991
Epoch 16/20
180/180 [==============================] - 7s 38ms/step - loss: 0.0017 - accuracy: 0.9993 - val_loss: 2.5253e-04 - val_accuracy: 1.0000
Epoch 17/20
180/180 [==============================] - 7s 38ms/step - loss: 5.8991e-04 - accuracy: 1.0000 - val_loss: 4.9001e-04 - val_accuracy: 1.0000
Epoch 18/20
180/180 [==============================] - 7s 38ms/step - loss: 6.7730e-04 - accuracy: 0.9998 - val_loss: 1.0663e-04 - val_accuracy: 1.0000
Epoch 19/20
180/180 [==============================] - 7s 37ms/step - loss: 2.7695e-04 - accuracy: 1.0000 - val_loss: 1.6648e-04 - val_accuracy: 1.0000
Epoch 20/20
180/180 [==============================] - 7s 37ms/step - loss: 5.2524e-04 - accuracy: 1.0000 - val_loss: 1.3989e-04 - val_accuracy: 1.0000
```

Wow! after 20 epochs we are able to get 100% accuracy on the training and validation data. Let's visualize this.

```python
plt.plot(history.history["accuracy"], label = "training")
plt.plot(history.history["val_accuracy"], label = "validation")
plt.gca().set(xlabel = "epoch", ylabel = "accuracy")
plt.legend()
```

INSET GRAPH HERE

We can see that the validation and training data reaches a similar level so we did not overfit the data. Additionally the accuracies have leveled off, so we are done training!

Since all three models are able to detect fake news in articles with at least 99% accuracy, I would recommend using the model with just title as it is the most efficient. However, if you are striving for the best accuracy, use the model with both title and text.

## Testing Out the Model

Now, let's test the model on data that it has never seen before. In this section I will use the `model` which has both `title` and `text` as inputs, but you could use any of the three models we created.

Let's read in the test data. Like before, we will read the data into a pandas dataframe. Then we will use the `make_dataset` function that we created to convert it to a TensorFlow dataset.

```python
test_url = "https://github.com/PhilChodrow/PIC16b/blob/master/datasets/fake_news_test.csv?raw=true"
test_data = pd.read_csv(test_url) # data to pandas dataframe
test = make_dataset(test_data) # convert data to Tensorflow dataset
```

Now we can use our model to evaluate the test data. We will batch the data for efficiency.

```python
test_results = test.take(len(test)).batch(100)
model.evaluate(test_results)
```

```
225/225 [==============================] - 3s 14ms/step - loss: 0.0476 - accuracy: 0.9917
[0.047586873173713684, 0.9916700124740601]
```

Wow! We achieved 99% accuracy on the test data. Our model is great at detecting fake news in articles.


## Visualizing

So our model can predict fake news with over 99% accuracy, but what has it learned? Let's visualize this.

```python
weights = model.get_layer('embedding').get_weights()[0] # get the weights from the embedding layer
vocab = vectorize_layer.get_vocabulary()                # get the vocabulary from our data prep for later

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
weights = pca.fit_transform(weights)

embedding_df = pd.DataFrame({
    'word' : vocab,
    'x0'   : weights[:,0],
    'x1'   : weights[:,1]
})
```

Remember how we created an embedding layer. A word embedding allows us to visualize what the model learned about the words. Words that are similar should be close together while words that are different are far apart. We will use plotly to create an interactive plot so we can see how the words are related to each other. The words that the model associates with fake news will tend towards one side and the words that the model associates with real news will be on the other.

```python
import plotly.express as px
fig = px.scatter(embedding_df,
                 x = "x0",
                 y = "x1",
                 size = list(np.ones(len(embedding_df))),
                 size_max = 2,
                 hover_name = "word")

fig.show()
```

PLOTLY FIGURE HERE

On the far left we can see words such as "breaking", "video", and "watch" which remind me of clickbait. Other notable word on the left are "KKK", "21wire" (a conspiracy new site), and "tucker" (maybe Tucker Carlson). It is clear that words on the left side are those that the model associates with fake news. On the right there are many country names such as "myanmar", "russia", and "chinas". It seems that international news is less likely to be fake.

Thanks for reading this blog post about detecting fake news with TensorFlow. I hope you learned something new!
