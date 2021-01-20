#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import tensorflow as tf
from glob import glob
from PIL import Image
import collections
import numpy as np
import zipfile
import random
import pickle
import time
import json
import re
import os




image_model                  = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
new_input                    = image_model.input
hidden_layer                 = image_model.layers[-1].output
image_features_extract_model = tf.keras.Model(new_input, hidden_layer)



encode_train  = sorted(set(img_name_vector))   # Get unique images
image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
image_dataset = image_dataset.map( load_image, num_parallel_calls = tf.data.experimental.AUTOTUNE).batch(16)

for img, path in image_dataset:
    batch_features = image_features_extract_model(img)
    batch_features = tf.reshape(batch_features,
                              (batch_features.shape[0], -1, batch_features.shape[3]))

    for bf, p in zip(batch_features, path):
        path_of_feature = p.numpy().decode("utf-8")
        np.save(path_of_feature, bf.numpy())

        

top_k     = 5000 # Choose the top 5000 words from the vocabulary
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                  oov_token="<unk>",
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
tokenizer.fit_on_texts(train_captions)
train_seqs                    = tokenizer.texts_to_sequences(train_captions)
tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0]       = '<pad>'
train_seqs                    = tokenizer.texts_to_sequences(train_captions) # Create the tokenized vectors
cap_vector                    = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post') 
# If you do not provide a max_length value, pad_sequences calculates it automatically
max_length                    = calc_max_length(train_seqs) 
# Calculates the max_length, which is used to store the attention weights




img_to_cap_vector = collections.defaultdict(list)
for img, cap in zip(img_name_vector, cap_vector):
    img_to_cap_vector[img].append(cap)

# Create training and validation sets using an 80-20 split randomly.
img_keys                               = list(img_to_cap_vector.keys())
random.shuffle(img_keys)
slice_index                            = int(len(img_keys)*0.8)
img_name_train_keys, img_name_val_keys = img_keys[:slice_index], img_keys[slice_index:]
img_name_train                         = []
cap_train                              = []
for imgt in img_name_train_keys:
    capt_len = len(img_to_cap_vector[imgt])
    img_name_train.extend([imgt] * capt_len)
    cap_train.extend(img_to_cap_vector[imgt])

img_name_val = []
cap_val      = []
for imgv in img_name_val_keys:
    capv_len = len(img_to_cap_vector[imgv])
    img_name_val.extend([imgv] * capv_len)
    cap_val.extend(img_to_cap_vector[imgv])

len(img_name_train), len(cap_train), len(img_name_val), len(cap_val)



# Feel free to change these parameters according to your system's configuration
BATCH_SIZE               = 128
BUFFER_SIZE              = 1000
embedding_dim            = 256
units                    = 512
vocab_size               = top_k + 1
num_steps                = len(img_name_train) // BATCH_SIZE
features_shape           = 2048  # Shape of the vector extracted from InceptionV3 is (64, 2048)
attention_features_shape = 64 # These two variables represent that vector shape
# Load the numpy files


dataset_train = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))
dataset_val   = tf.data.Dataset.from_tensor_slices((img_name_val, cap_val))

# Use map to load the numpy files in parallel
dataset_train = dataset_train.map(lambda item1, item2: tf.numpy_function(
          map_func, [item1, item2], [tf.float32, tf.int32]),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)

dataset_val   = dataset_val.map(lambda item1, item2: tf.numpy_function(
          map_func, [item1, item2], [tf.float32, tf.int32]),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)
# Shuffle and batch
dataset_train = dataset_train.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset_train = dataset_train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
dataset_val   = dataset_val.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset_val   = dataset_val.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


encoder     = CNN_Encoder(embedding_dim)
decoder     = RNN_Decoder(embedding_dim, units, vocab_size)
optimizer   = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')



checkpoint_path = "/content/drive/My Drive/dd/check_attention"
ckpt            = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer = optimizer )
ckpt_manager    = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)
start_epoch     = 0
if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
    ckpt.restore(ckpt_manager.latest_checkpoint) # restoring the latest checkpoint in checkpoint_path

try :
    loss_plot       = np.load("/content/drive/My Drive/dd/train_loss.npy").tolist()
    val_losses_list = np.load("/content/drive/My Drive/dd/val_loss.npy").tolist()
except :
    loss_plot       = []
    val_losses_list = []
    
    
    

EPOCHS          = 50
#track_val_loss = 100000
track_val_loss  = min(val_losses_list)
tracker         = 0
print(track_val_loss)
for epoch in range(start_epoch, EPOCHS):
    start            = time.time()
    total_loss_train = 0

    for (batch, (img_tensor, target)) in enumerate(dataset_train):
        batch_loss, t_loss = train_step(img_tensor, target)
        total_loss_train  += t_loss

        if batch % 100 == 0:
            print ('Epoch {} Batch {} Loss {:.4f}'.format( epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))
    
    loss_plot.append(total_loss_train / num_steps) # storing the epoch end loss value to plot later
    val_losses = []
    for (batch, (img_tensor, target)) in enumerate(dataset_val):
        loss_val  = 0
        hidden    = decoder.reset_state(batch_size=target.shape[0])
        dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)
        features  = encoder(img_tensor)

        for i in range(1, target.shape[1]):
            predictions, hidden, _ = decoder(dec_input, features, hidden) # passing the features through the decoder
            loss_val              += loss_function(target[:, i], predictions)
            dec_input              = tf.expand_dims(target[:, i], 1) # using teacher forcing

        total_loss_val = (loss_val / int(target.shape[1]))
        val_losses.append(total_loss_val)
    val_losses_list.append(np.mean(val_losses))
    if track_val_loss > np.mean(val_losses) :
        track_val_loss = np.mean(val_losses)
        tracker        = 0
    else :
        tracker +=1

    if tracker == 5 :
        break
    
    if epoch % 1 == 0 and tracker == 0:
        ckpt_manager.save()
    np.save("/content/drive/My Drive/dd/val_loss" , np.array(val_losses_list))
    np.save("/content/drive/My Drive/dd/train_loss" , np.array(loss_plot))
    
    print ('Epoch {} Loss_train {:.6f}'.format(epoch + 1,
                                         total_loss_train/num_steps))
    
    print ('Epoch {} Loss_val {:.6f}'.format(epoch + 1,
                                         np.mean(val_losses)))
    
    print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
    

    
    
# captions on the validation set
rid                    = np.random.randint(0, len(img_name_val))
image                  = img_name_val[rid]
real_caption           = ' '.join([tokenizer.index_word[i] for i in cap_val[rid] if i not in [0]])
result, attention_plot = evaluate(image)

print ('Real Caption:', real_caption)
print ('Prediction Caption:', ' '.join(result))
plot_attention(image, result, attention_plot)



image_url       = 'https://tensorflow.org/images/surf.jpg'
image_extension = image_url[-4:]
image_path      = tf.keras.utils.get_file('image'+image_extension,
                                     origin=image_url)
#image_path = img_name_val[90]
#image_path = "/content/drive/My Drive/photo-1503023345310-bd7c1de61c7d.jpg"

result, attention_plot = evaluate(image_path)
print ('Prediction Caption:', ' '.join(result))
plot_attention(image_path, result, attention_plot)
# opening the image
Image.open(image_path)


