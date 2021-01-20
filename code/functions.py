#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# functions

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path



# Find the maximum length of any caption in our dataset
def calc_max_length(tensor):
    return max(len(t) for t in tensor)

def map_func(img_name, cap):
    img_tensor = np.load(img_name.decode('utf-8')+'.npy')
    return img_tensor, cap




def loss_function(real, pred):
    mask   = tf.math.logical_not(tf.math.equal(real, 0))
    loss_  = loss_object(real, pred)

    mask   = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)



@tf.function
def train_step(img_tensor, target):
    loss      = 0
    # initializing the hidden state for each batch ,  because the captions are not related from image to image
    hidden    = decoder.reset_state(batch_size=target.shape[0])
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)

    with tf.GradientTape() as tape:
        features = encoder(img_tensor)

    for i in range(1, target.shape[1]):
        predictions, hidden, _ = decoder(dec_input, features, hidden) # passing the features through the decoder
        loss                  += loss_function(target[:, i], predictions)
        # using teacher forcing
        dec_input              = tf.expand_dims(target[:, i], 1)

    total_loss          = (loss / int(target.shape[1]))
    trainable_variables = encoder.trainable_variables + decoder.trainable_variables
    gradients           = tape.gradient(loss, trainable_variables)

    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, total_loss






def evaluate(image):
    attention_plot = np.zeros((max_length, attention_features_shape))
    hidden         = decoder.reset_state(batch_size=1)
    temp_input     = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))
    features       = encoder(img_tensor_val)
    dec_input      = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result         = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)
        attention_plot[i]                      = tf.reshape(attention_weights, (-1, )).numpy()
        predicted_id                           = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(tokenizer.index_word[predicted_id])
        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot
        dec_input = tf.expand_dims([predicted_id], 0)
        
    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot


def plot_attention(image, result, attention_plot):
    temp_image = np.array(Image.open(image))
    fig        = plt.figure(figsize=(10, 10))
    len_result = len(result)
    
    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (8, 8))
        ax       = fig.add_subplot(len_result//2, len_result//2, l+1)
        ax.set_title(result[l])
        img      = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())
        plt.savefig("image_{}".format(l), fig)

    plt.tight_layout()
    plt.show()
    
  

