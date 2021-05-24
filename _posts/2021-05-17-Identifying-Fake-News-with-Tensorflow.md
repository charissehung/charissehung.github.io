---
layout: post
title: Identifying Fake News with TensorFlow
---

In this post, I'll explain how to create a fake news classifier using TensorFlow. We will be creating three models to predict whether news articles contain fake news, and then evaluate them.

There are quite a few steps, so here is the process we will follow.

1. Acquire Training Data
2. Create a Dataset
3. Preprocessing
4. Creating the Title Model
5. Creating the Text Model
6. Creating the Combined Model
7. Testing Out the Model
8. Visualizing

Before we get into it, let's download the necessary packages. As we can see, we will need many functionalities of TensorFlow.

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

## 1. Acquire Training Data

The data that we will use to train the model can be found at the link below.

```python
train_url = "https://github.com/PhilChodrow/PIC16b/blob/master/datasets/fake_news_train.csv?raw=true"
```

Let's read this data into a pandas dataframe.

```python
train_data = pd.read_csv(train_url)
train_data
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>title</th>
      <th>text</th>
      <th>fake</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17366</td>
      <td>Merkel: Strong result for Austria's FPO 'big c...</td>
      <td>German Chancellor Angela Merkel said on Monday...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5634</td>
      <td>Trump says Pence will lead voter fraud panel</td>
      <td>WEST PALM BEACH, Fla.President Donald Trump sa...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>17487</td>
      <td>JUST IN: SUSPECTED LEAKER and “Close Confidant...</td>
      <td>On December 5, 2017, Circa s Sara Carter warne...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12217</td>
      <td>Thyssenkrupp has offered help to Argentina ove...</td>
      <td>Germany s Thyssenkrupp, has offered assistance...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5535</td>
      <td>Trump say appeals court decision on travel ban...</td>
      <td>President Donald Trump on Thursday called the ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>22444</th>
      <td>10709</td>
      <td>ALARMING: NSA Refuses to Release Clinton-Lynch...</td>
      <td>If Clinton and Lynch just talked about grandki...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22445</th>
      <td>8731</td>
      <td>Can Pence's vow not to sling mud survive a Tru...</td>
      <td>() - In 1990, during a close and bitter congre...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22446</th>
      <td>4733</td>
      <td>Watch Trump Campaign Try To Spin Their Way Ou...</td>
      <td>A new ad by the Hillary Clinton SuperPac Prior...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22447</th>
      <td>3993</td>
      <td>Trump celebrates first 100 days as president, ...</td>
      <td>HARRISBURG, Pa.U.S. President Donald Trump hit...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22448</th>
      <td>12896</td>
      <td>TRUMP SUPPORTERS REACT TO DEBATE: “Clinton New...</td>
      <td>MELBOURNE, FL is a town with a population of 7...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>22449 rows × 4 columns</p>
</div>

We see that this dataset contains information about 22449 articles. Each row represents an article and there are three columns that contain the title, the full article text, and a boolean value that indicates whether the article is fake or not. This fake column is 0 when the article is true and 1 when the article contains fake news.


## 2. Create a Dataset

We're going to create a function called `make_dataset`. This will convert our pandas dataframe to a TensorFlow Dataset. This allows us to stay organized while creating our data pipeline. In this function, we will remove stop words from the title and text columns of each article. Stop words in English are words like "the", "and", or "but". We use a lambda function which loops through all the words, and then removes the word if it is a stopword as determined in the `nltk` package.

{::options parse_block_html="true" /}
<div class="got-help">
A peer suggested that I add more explanation regarding the `make_dataset` function because contains many trick steps such as getting the stopwords from `nltk` and using lambda functions.
</div>
{::options parse_block_html="false" /}

Then we will define the input and output components of our dataset. We want the model to evaluate the title and text of the article, so these are the inputs. The output will be 0 or 1, depending if the article contains fake news or not.

```python
def make_dataset(train_data):
  # get list of stopwords from nltk
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
            "text" : train_data[["text"]]
        },
        {
            "fake" : train_data[["fake"]]
        }
    )
)
  data.batch(100) # batching the data allows it to efficiently train in chunks

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

## 3. Preprocessing

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

Let's apply the vectorization to the `title` and `text` columns of the dataset! Here lambda functions are used to apply the vectorization to all the titles and texts of the articles.

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

## 4. Creating the Title Model

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

{::options parse_block_html="true" /}
<div class="gave-help">
I gave a suggestion to a classmate to create an embedding layer independent of the models. This allows all the models to share the embedding layer and then one single text embedding visualization can be created from it.
</div>
{::options parse_block_html="false" /}

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

Let's look at a diagram that represents the layers of the model to understand its structure.

```python
keras.utils.plot_model(model_title)
```

![blog3model1]({{ site.baseurl }}/images/blog3model1.png)

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
.....
Epoch 18/20
180/180 [==============================] - 3s 16ms/step - loss: 0.0259 - accuracy: 0.9915 - val_loss: 0.0187 - val_accuracy: 0.9947
Epoch 19/20
180/180 [==============================] - 3s 17ms/step - loss: 0.0245 - accuracy: 0.9921 - val_loss: 0.0205 - val_accuracy: 0.9947
Epoch 20/20
180/180 [==============================] - 3s 16ms/step - loss: 0.0238 - accuracy: 0.9922 - val_loss: 0.0163 - val_accuracy: 0.9947
```

{::options parse_block_html="true" /}
<div class="got-help">
Initially, I printed the output from all 20 epochs, and it was quite overwhelming. A peer suggested that I condense these outputs, since the following graph shows more clearly how the accuracy of the model changes with each epoch.
</div>
{::options parse_block_html="false" /}


After 20 epochs, the model is has over 99% accuracy on both the training and validation data. This is great! Let's visualize the accuracy over time.

```python
plt.plot(history_title.history["accuracy"], label = "training")
plt.plot(history_title.history["val_accuracy"], label = "validation")
plt.gca().set(xlabel = "epoch", ylabel = "accuracy")
plt.legend()
```

![blog3plot1]({{ site.baseurl }}/images/blog3plot1.png)

We can see that the training and validation data appear to have similar accuracy, which indicates we did not overfit the model. Additionally, the accuracies of both have "leveled off" so we probably would not benefit from more training. Our first model is complete!

{::options parse_block_html="true" /}
<div class="gave-help">
I gave a suggestion to a classmate to plot the accuracy of the model. This makes it easier to see when the model is done training or whether it could benefit from additional training.
</div>
{::options parse_block_html="false" /}

## 5. Creating the Text Model

For the second model, we will determine whether an article contains fake news or not solely based on the article text. This process will be very similar to the process above for creating the title model. We begin by creating some layers for the text input. Notice we use the embedding layer that we defined above.

```python
text_features = vectorize_layer(text_input) # vectorize the text
text_features = embedding(text_features) # use defined embedding layer
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

Let's look at the diagram to better understand the structure of the model.

```python
keras.utils.plot_model(model_text)
```

![blog3model2]({{ site.baseurl }}/images/blog3model2.png)

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
.....
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

![blog3plot2]({{ site.baseurl }}/images/blog3plot2.png)

We can see that the validation data has a slightly higher accuracy than the training data, so we did not overfit the model. Both lines have "leveled off" so we are done training this model!

## 6. Creating the Combined Model

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

Let's visualize the structure of the model.

```python
keras.utils.plot_model(model)
```

![blog3model3]({{ site.baseurl }}/images/blog3model3.png)

Notice how both the text and title go through the same embedding layer. Recall, we defined our own embedding layer and then had both the title and text use it. Another cool thing to note is how the two "branches" of layers come together at the concatenate layer. We explicitly did this when we created `main`. Let's compile our model and train it.

```python
# compile the model
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
.....
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

![blog3plot3]({{ site.baseurl }}/images/blog3plot3.png)

We can see that the validation and training data reaches a similar level so we did not overfit the data. Additionally the accuracies have leveled off, so we are done training!

Since all three models are able to detect fake news in articles with at least 99% accuracy, I would recommend using the model with just title as it is the most efficient. However, if you are striving for the best accuracy, use the model with both title and text.

## 7. Testing Out the Model

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


## 8. Visualizing

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

Remember how we created an embedding layer? A word embedding allows us to visualize what the model learned about the words. Words that are similar should be close together while words that are different are far apart. We will use `plotly` to create an interactive plot so we can see how the words are related to each other. The words that the model associates with fake news will tend towards one side and the words that the model associates with real news will be on the other.

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

{% include word_embedding.html %}

On the far left we can see words such as "breaking", "video", and "watch" which remind me of clickbait. Other notable word on the left are "KKK", "21wire" (a conspiracy new site), and "tucker" (maybe Tucker Carlson). It is clear that words on the left side are those that the model associates with fake news. On the right there are many country names such as "myanmar", "russia", and "chinas". It seems that international news is less likely to be fake.

Thanks for reading this blog post about detecting fake news with TensorFlow. I hope you learned something new!
