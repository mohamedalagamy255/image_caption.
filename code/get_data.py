#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# get_data






os.environ['KAGGLE_USERNAME'] = "******" # username from the json file
os.environ['KAGGLE_KEY']      = "********" # key from the json file
get_ipython().system('kaggle datasets download -d nadaibrahim/coco2014')

zip_file = zipfile.ZipFile("/content/coco2014.zip")
for name in zip_file.namelist():
    if name.startswith('train2014/train2014/COCO_train2014'):
        zip_file.extract(name)

zip_file.extract('captions/annotations/captions_train2014.json')


#img_name_vector = np.load("/content/drive/My Drive/attention_image_pathes.npy").tolist()
#train_captions  = np.load("/content/drive/My Drive/attention_caption_pathes.npy").tolist()
#for name in img_name_vector:
    #zip_file.extract(name[9:])

    
    

with open("/content/captions/annotations/captions_train2014.json", 'r') as f:
    annotations = json.load(f)

PATH                  = "/content/train2014/train2014/"
# Group all captions together having the same image ID.
image_path_to_caption = collections.defaultdict(list)
for val in annotations['annotations']:
    caption    = f"<start> {val['caption']} <end>"
    image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (val['image_id'])
    image_path_to_caption[image_path].append(caption)


image_paths       = list(image_path_to_caption.keys())
random.shuffle(image_paths)
train_image_paths = image_paths[:40000]
print(len(train_image_paths))

train_captions = []
img_name_vector = []

for image_path in train_image_paths:
    caption_list = image_path_to_caption[image_path]
    train_captions.extend(caption_list)
    img_name_vector.extend([image_path] * len(caption_list))

np.save(r"/content/drive/My Drive/attention_image_pathes",np.array(img_name_vector))
np.save(r"/content/drive/My Drive/attention_caption_pathes",np.array(train_captions))





