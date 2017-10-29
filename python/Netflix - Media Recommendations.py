# Databricks notebook source
# MAGIC %md
# MAGIC # References
# MAGIC ## Data Sets
# MAGIC * Netflix data set: https://opendata.stackexchange.com/questions/7883/netflix-data-set/7884
# MAGIC   * https://www.kaggle.com/netflix-inc/netflix-prize-data
# MAGIC   * https://archive.org/details/nf_prize_dataset.tar
# MAGIC     * https://archive.org/download/nf_prize_dataset.tar/nf_prize_dataset.tar.gz (665 MB)
# MAGIC   * https://web.archive.org/web/20090925184737/http://archive.ics.uci.edu/ml/datasets/Netflix+Prize
# MAGIC   * http://academictorrents.com/details/9b13183dc4d60676b773c9e2cd6de5e5542cee9a
# MAGIC * MovieLens: https://grouplens.org/datasets/movielens/
# MAGIC   * http://files.grouplens.org/datasets/movielens/ml-20m.zip (190 MB)
# MAGIC ## Source Code
# MAGIC * Source code: http://github.com/telecoms-intelligence/netflix-study
# MAGIC # Dependencies
# MAGIC * That notebook needs a lot of memory to read the Netflix movie rating files, at least 64GB.
# MAGIC A cluster known to run properly is ``Single - 122GB - DB 33.3``

# COMMAND ----------

# MAGIC %sh
# MAGIC # It takes roughly 3 minutes on 'Single - 122 GB'
# MAGIC # pip install -U pip
# MAGIC pip install -U word2veckeras

# COMMAND ----------

import os, re, logging
import random
import datetime as dt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import gensim.models as gm
import matplotlib.pyplot as plt


# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC ls -laFh /dbfs/mnt/data-science/use-cases/uc01-recommendation-engine/netflix/

# COMMAND ----------

# MAGIC %fs
# MAGIC ls /mnt/data-science/use-cases/uc01-recommendation-engine/netflix

# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC head -5 /dbfs/mnt/data-science/use-cases/uc01-recommendation-engine/netflix/training_set/mv_0000190.txt

# COMMAND ----------

# MAGIC %md
# MAGIC Jump to [Read the DataFrame from a Pickle file section to avoid re-reading the Netflix data files](#notebook/5005/command/5698)

# COMMAND ----------

# Takes 27-30 hours on the 'Single - 122 GB - DB 3.3' cluster
def read_nf_training_set(nf_filepath):
  file_col_names = ['User', 'Rating', 'DoR']
  file_col_types = {'User': np.int64, 'Rating': np.int64}
  
  files = os.listdir(nf_filepath)
  # Read all files from the directory
  df = pd.DataFrame()
  for fil_index, fil in enumerate(files):
    if fil_index % 100 == 0:
        print (fil_index)
    df_tmp = pd.read_csv(nf_filepath + fil, names = file_col_names, skiprows=[0], parse_dates = [2], dtype = file_col_types)
    name = re.findall(r'\d+', fil)
    df_tmp = df_tmp.reset_index(drop=True)
    df_tmp['Mov'] = name * len(df_tmp)
    df = pd.concat([df, df_tmp], axis=0)
    
  print('done')
  return df

#
#df = read_nf_training_set("/dbfs/mnt/data-science/use-cases/uc01-recommendation-engine/netflix/training_set/")
#df.dtypes

# COMMAND ----------

# Dump/Serialize the DataFrame into a file (with Pickle)
# Note that the Pickle format/protocol is highly dependent on the Python version used. Here Python 2 is used,
# and later Pickle versions (eg, protocol version 4) are not supported
# Takes almost 13 minutes on the 'Single - 122 GB - DB 3.3' cluster
#df.to_pickle('/dbfs/mnt/data-science/use-cases/uc01-recommendation-engine/netflix/nfc.pkl')


# COMMAND ----------

# Takes almost 2 minutes on the 'Single - 122 GB - DB 3.3' cluster
df = pd.read_pickle('/dbfs/mnt/data-science/use-cases/uc01-recommendation-engine/netflix/nfc.pkl')

# COMMAND ----------

# The following has to be uncommented when 'df' is read from 'nfb.pkl', rather than from 'nfc.pkl'
# Takes 30 seconds on the 'Single - 122 GB - DB 3.3' cluster
#df['DoV'] = df['DoV'].astype('datetime64[ns]')
df.dtypes

# COMMAND ----------

# Temporary hack for 'nfb.pkl', which misses movie rating data files
def read_nf_missing_training_set_from_file(df_glob, nf_name, nf_filepath):
  file_col_names = ['User', 'Rating', 'DoV']
  file_col_types = {'User': np.int64, 'Rating': np.int64}
  
  df_tmp = pd.read_csv(nf_filepath, names = file_col_names, skiprows = [0], parse_dates = [2], dtype = file_col_types)
  df_tmp = df_tmp.reset_index()
  del df_tmp['index']
  df_tmp['Mov'] = nf_name * len(df_tmp)
  df_glob = pd.concat([df_glob, df_tmp], axis=0)
  return df_glob

#
def read_nf_missing_training_set():
  df_glob = pd.DataFrame()
  with open('/dbfs/mnt/data-science/use-cases/uc01-recommendation-engine/netflix/missing_files.txt') as lfile:
    missing_files = lfile.readlines()
  missing_files = [x.strip() for x in missing_files]

  #
  for missing_file in missing_files:
    missing_filepath = "/dbfs/mnt/data-science/use-cases/uc01-recommendation-engine/netflix/training_set/" + missing_file
    missing_movie_name_as_list = re.findall(r'\d+', missing_file)
    movie_name = missing_movie_name_as_list[0]
    print("Name: " + movie_name + ", filepath: " + missing_filepath)
    df_glob = read_nf_missing_training_set_from_file(df_glob, movie_name, missing_filepath)

  return df_glob


# COMMAND ----------

# Takes 2 minutes on the 'Single - 122 GB - DB 3.3' cluster
#df_missing = read_nf_missing_training_set()
#df_missing.dtypes

# COMMAND ----------

#df = pd.concat([df, df_missing], axis=0)
#df.dtypes

# COMMAND ----------

# The following has to be uncommented when 'df' is read from 'nfb.pkl', rather than from 'nfc.pkl'
#df['DoR'] = df['DoV']
#df = df.drop(['DoV'], axis = 1)
#df.dtypes

# COMMAND ----------

# Aggregate the movie reviews per (movie, date of review) pairs.
# For each movie, the number of unique users and the average rating are computed.
# Takes 4 minutes on the 'Single - 122 GB - DB 3.3' cluster
h = {'User': ['count'], 'Rating': ['mean']}
df_mov_g = df.groupby(['Mov', 'DoR']).agg(h).reset_index()
df_mov_g.columns = ['Mov', 'DoR', 'Rating_mean', 'User_count']

# Sort by date of review
df_mov_g = df_mov_g.sort_values(['Mov', 'DoR'])

# Add a column materializing the change of movie
df_mov_g['first_day_mov'] = np.where(df_mov_g.Mov != df_mov_g.Mov.shift(1), 1, 0)

# Keep only the first date of review for every movie
df_mov_g = df_mov_g[df_mov_g['first_day_mov'] == 1]


# COMMAND ----------

# MAGIC %md
# MAGIC Please see [comment about dumping DataFrame with Pickle](#notebook/5005/command/5703)

# COMMAND ----------

# Takes almost 13 minutes on the 'Single - 122 GB - DB 3.3' cluster
#df_mov_g.to_pickle('/dbfs/mnt/data-science/use-cases/uc01-recommendation-engine/netflix/nfb_mov_first.pkl')

# COMMAND ----------

#df_mov_g = pd.read_pickle('/dbfs/mnt/data-science/use-cases/uc01-recommendation-engine/netflix/nfb_mov_first.pkl')
#df_mov_g.dtypes
df_mov_g

# COMMAND ----------

fig, ax = plt.subplots()

ax.set_ylim((0,5))
ax.set_xlim((0,700))
ax.set_xlabel('Number of times movie watched by unq viewrs on the release day', fontsize=18)
ax.set_ylabel('Average rating', fontsize=18)
ax.scatter(df_mov_g.User_count, df_mov_g.Rating_mean) 
# Conclusion- To solve cold start problem for a movie, there is no pattern
# New Movies should be indexed by some meta data

display(fig)

# COMMAND ----------

# Takes 50 seconds on the 'Single - 122 GB - DB 3.3' cluster
f = {'Mov': ['count'], 'Rating': ['mean']}
df_usr_g = df.groupby(['User','DoR']).agg(f).reset_index()
df_usr_g.columns = ['User', 'DoR', 'Rating_mean', 'Mov_count']

df_usr_g= df_usr_g.sort_values(['User', 'DoR'])
df_usr_g['first_day_usr'] = np.where(df_usr_g.User != df_usr_g.User.shift(1), 1, 0 )
df_usr_g= df_usr_g[df_usr_g['first_day_usr'] == 1]


# COMMAND ----------

# MAGIC %md
# MAGIC Please see [comment about dumping DataFrame with Pickle](#notebook/5005/command/5703)

# COMMAND ----------

# Takes almost 13 minutes on the 'Single - 122 GB - DB 3.3' cluster
#df_usr_g.to_pickle('/dbfs/mnt/data-science/use-cases/uc01-recommendation-engine/netflix/nfb_usr_first.pkl')

# COMMAND ----------

#df_usr_g = pd.read_pickle('/dbfs/mnt/data-science/use-cases/uc01-recommendation-engine/netflix/nfb_usr_first.pkl')
#df_usr_g.dtypes
df_usr_g

# COMMAND ----------

fig, ax = plt.subplots()

ax.set_ylim((0,5))
ax.set_xlim((0,2000))
ax.set_xlabel('Number of movies rated by unq user on their first day', fontsize=18)
ax.set_ylabel('Average rating', fontsize=18)
ax.scatter(df_usr_g.Mov_count, df_usr_g.Rating_mean) 
# Conclusion- To solve cold start problem for a user, there is no pattern
# New Users should be asked in some way
# However when users rate a lot of movie it reverts to mean

display(fig)


# COMMAND ----------

df.dtypes

# COMMAND ----------

df_usr_g.dtypes

# COMMAND ----------

# Takes 25 minutes on the 'Single - 122 GB - DB 3.3' cluster
df_join = pd.merge(df, df_usr_g, on='User', how='left', suffixes=('_x', '_y'))
df_join = df_join.drop(['Rating_mean', 'Mov_count', 'first_day_usr'], axis=1)
df_join['day_count'] = (df_join['DoR_x'] - df_join['DoR_y']).dt.days
df_join = df_join.drop(['DoR_y'], axis=1)
df_join.columns = ['User', 'Rating', 'DoR', 'Mov', 'day_count']
df_join.dtypes

# COMMAND ----------

# MAGIC %md
# MAGIC Please see [comment about dumping DataFrame with Pickle](#notebook/5005/command/5703)

# COMMAND ----------

# Takes almost 13 minutes on the 'Single - 122 GB - DB 3.3' cluster
#df_join.to_pickle('/dbfs/mnt/data-science/use-cases/uc01-recommendation-engine/netflix/nfb_join.pkl')

# COMMAND ----------

#df_join = pd.read_pickle('/dbfs/mnt/data-science/use-cases/uc01-recommendation-engine/netflix/nfb_join.pkl')
df_join

# COMMAND ----------

# Takes 45 seconds on the 'Single - 122 GB - DB 3.3' cluster
g = {'Rating': ['mean', 'count' , 'std']}
df_time = df_join.groupby(['User', 'day_count']).agg(g).reset_index()
df_time.columns = ['User', 'day_count', 'rating_mean', 'rating_count', 'rating_std']

# COMMAND ----------

df_time

# COMMAND ----------

df_time_t = df_time[df_time['User'] == 6]

# COMMAND ----------

df_time_t.sort_values(['day_count'])

# COMMAND ----------

fig, ax = plt.subplots()

ax.set_ylim((0,5))
ax.set_xlim((0,600))
ax.set_xlabel('Customer Days since his first rating', fontsize=18)
ax.set_ylabel('Average rating', fontsize=18)
ax.scatter(df_time_t.day_count, df_time_t.rating_mean) 
#Users have a global baseline. They only like to rate above a baseline
#Hence models have to be designed to give special preference to low weights as heavily biased 
# towards positive ones
# some people have chosen to rank all the movies in a day 2040859 
# These users habitually rank all the movie

display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC * Is there a change in taste of customer
# MAGIC * Is there a change in rating behaviour of customer in long term

# COMMAND ----------

# Just check the 'Mov' column (movie ID)
df_join['Mov']

# COMMAND ----------

# Takes 15 minutes on the 'Single - 122 GB - DB 3.3' cluster
# mov_ind goes up to 100,140,000 (100 millions), and takes roughly 15 minutes on 'Single - 122 GB'
movie = df_join['Mov'].tolist()
day_count = df_join['day_count'].tolist()
sentence = []
sentence_tmp = []
sentence_tmp.append(str(movie[0]))
for mov_ind, mov in enumerate(movie):
    if mov_ind % 10000 == 0:
        print(mov_ind)
    if mov_ind > 0:
        if day_count[mov_ind] !=  day_count[mov_ind-1]:
            sentence.append(sentence_tmp)
            sentence_tmp = []
        sentence_tmp.append(str(mov))

sentence

# COMMAND ----------

# sentence is a list, and cannot be dumped with Pickle.
# Shelve (https://docs.python.org/3/library/shelve.html) may be used instead
#sentence.to_pickle('/dbfs/mnt/data-science/use-cases/uc01-recommendation-engine/netflix/nf_sentence.pkl')

# COMMAND ----------

# See above about Pickle vs Shelve
#sentence = pd.read_pickle('/dbfs/mnt/data-science/use-cases/uc01-recommendation-engine/netflix/nf_sentence.pkl')

# COMMAND ----------

# Step 3. Vector representation of movies
# It takes roughly 13 minutes on 'Single - 122 GB'

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

# Set values for various parameters
num_features = 100    # Word vector dimensionality                      
min_word_count = 3   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 1         # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words
hs =1
negative =0
# Initialize and train the model (this will take some time)
print("Training model...")
# gm stands for gensim.models
model = gm.word2vec.Word2Vec(sentence, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

model.init_sims(replace=True)

model_name = "100features_3minmovs_1context"
model.save(model_name)


# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC grep -e "15124" -e "15164" -e "15246" -e "15436" /dbfs/mnt/data-science/use-cases/uc01-recommendation-engine/netflix/movie_titles.txt

# COMMAND ----------

# 15124, 1996, Independence Day
# 15164, 2000, The X-Files: Season 8
# 15246, 1992, Alien 3: Collector's Edition
# 15436, 1998, A Perfect Murder
model.doesnt_match(['0015124', '00015164', '0015246', '0015436'])


# COMMAND ----------

# 15124,1996,Independence Day
# 15164,2000,The X-Files: Season 8
# 15246,1992,Alien 3: Collector's Edition
# 15436,1998,A Perfect Murder
model.wv.most_similar(positive=['0015164', '0015246'], negative=['0015436'])
#model.score(['0015124','00015164','0015246','0015436'])


# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC head -5 /dbfs/mnt/data-science/use-cases/uc01-recommendation-engine/netflix/training_set/mv_0009733.txt

# COMMAND ----------

# Takes 60 minutes on the 'Single - 122 GB - DB 3.3' cluster
# For a while, the mv_0009733.txt file (and many others) was missing, triggering the following error:
# KeyError: "word '0009733' not in vocabulary"
tr = []
for item in range(1, 17771):
    item_r = str('000000') + str(item)
    item_r = item_r[-7:]
    tr.append(model.wv[item_r])


# COMMAND ----------

tr = np.array(tr)

# COMMAND ----------

# Takes 4 minutes on 'Single - 122 GB'
# TSNE comes from sklearn.manifold
model_tsne = TSNE(n_components=2)
vis_tsne = model_tsne.fit_transform(tr)

# COMMAND ----------

fig, ax = plt.subplots()

vis_x = vis_tsne[:, 0]
vis_y = vis_tsne[:, 1]

ax.scatter(vis_x, vis_y,s=1)

display(fig)


# COMMAND ----------

# Takes 16 minutes on 'Single - 122 GB - DB 3.3'
# TSNE comes from from sklearn.manifold
model_tsne = TSNE(n_components=3)
vis_tsne3 = model_tsne.fit_transform(tr)

# COMMAND ----------

# Takes 2 minutes on the 'Single - 122 GB - DB 3.3' cluster
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import *

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
for item in range(0, len(vis_tsne)):
    ax.scatter(vis_tsne3[item,0], vis_tsne3[item,1], vis_tsne3[item,2],s=1)
    #ax.text(vec[item,0], vec[item,1], vec[item,2], '%s' % (str(text[item])), size = 7,
                #zorder =1, color ='r')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

display(fig)


# COMMAND ----------

# Takes 1 minute on the 'Single - 122 GB - DB 3.3' cluster
df_h = df.copy(deep = False)
df_h.sort_values('User', inplace = True)

# COMMAND ----------

# Takes 15 minutes on the 'Single - 122 GB - DB 3.3' cluster
#df_h.to_pickle('/dbfs/mnt/data-science/use-cases/uc01-recommendation-engine/netflix/nf_intermediary.pkl')

# COMMAND ----------

#df_h = pd.read_pickle('/dbfs/mnt/data-science/use-cases/uc01-recommendation-engine/netflix/nf_intermediary.pkl')

# COMMAND ----------

# Takes 2 minutes on the 'Single - 122 GB - DB 3.3' cluster
model = gm.Word2Vec.load("100features_3minmovs_1context")
print('done')

# COMMAND ----------

# Takes 11 minutes of the 'Single - 122 GB - DB 3.3' cluster
user = df_h['User'].tolist()
rating = df_h['Rating'].tolist()
mov = df_h['Mov'].tolist()
dor =  df_h['DoR'].tolist()

# COMMAND ----------

# Takes 1.2 hour of the 'Single - 122 GB - DB 3.3' cluster
flag = []
past_m = []
past_s = []
current_m = []
current_s = []
u = []
user_dic = {}
user_count = 0
ag = []
nb_user = len(user)
user_not_fit = []
user_not_fit_dict = {}
for counter in range(0, nb_user-5):
  # Users where we do not have the 5 movies in history will not be recommended from this model
  current_user = user[counter]
  if current_user == user[counter+5]:
    # setting the flag
    if (dor[counter+5] - dor[counter]).days > 30:
      flag.append(1)
    else:
      flag.append(0)
    # setting the user
    if not current_user in user_dic:
      user_count += 1
      user_dic[current_user] = user_count

    past_m_tmp = []
    past_s_tmp = []
        
    for c in range(0, 5):
      past_m_tmp.append(model.wv[mov[counter+c]])
      past_s_tmp.append(rating[counter+c])
    past_m.append(past_m_tmp)
    past_s.append(past_s_tmp)
    ag.append(sum(past_s_tmp)/5)
    current_m.append(model.wv[mov[counter+5]])
    current_s.append(rating[counter+5])
    u.append(user_dic[current_user])
  else:
    user_not_fit.append(current_user)
    if not current_user in user_not_fit_dict:
      user_not_fit_dict[current_user] = 1
    pass
    
  interval = int(nb_user / 100)
  if counter % interval == 0:
    print ("Done " + str(int(100 * counter / nb_user)) + "%")
print('done')


# COMMAND ----------

print("Nb of rows: " + str(nb_user/1e6) + ", nb of fitting rows: " + str(len(user_dic)/1e6) + ", nb of non fitting rows: " + str(len(user_not_fit)/1e6))
print("user_dic: " + str(user_dic))
print("user_not_fit_dict: " + str(user_not_fit_dict))

# COMMAND ----------

df_train = pd.DataFrame()
df_train['user'] = u
df_train['past_m']= past_m
df_train['past_s']= past_s
df_train['current_m'] = current_m
df_train['current_s'] = current_s
df_train['ag'] = ag
df_train['flag'] = flag

# COMMAND ----------

# Takes 15 minutes on the 'Single - 122 GB - DB 3.3' cluster
#df_train.to_pickle('/dbfs/mnt/data-science/use-cases/uc01-recommendation-engine/netflix/nf_train_mod.pkl')

# COMMAND ----------

#df_train = pd.read_pickle('/dbfs/mnt/data-science/use-cases/uc01-recommendation-engine/netflix/nf_train_mod.pkl')
df_train

# COMMAND ----------

# df_train = pd.read_pickle('/dbfs/mnt/data-science/use-cases/uc01-recommendation-engine/netflix/nf_train_mod.pkl')
df_train = df_train.sample(frac=1)
df_train_r = df_train[((df_train['flag']== 0) & (df_train['ag'] < 2.5)) | ((df_train['flag']== 0) & (df_train['ag'] > 3.5))]


# COMMAND ----------

X1= np.array(df_train_r['past_m'].tolist())
X1=X1.reshape(len(X1),5, 100,1)
X2= np.array(df_train_r['past_s'].tolist())
X2=X2.reshape(len(X2),5)
X4= np.array(df_train_r['current_m'].tolist())
X4=X4.reshape(len(X4),1,100)

# COMMAND ----------

from keras.layers import Embedding
from keras.layers import Input, Convolution2D, Dense, merge, Flatten, Dropout, MaxPooling2D, Input, Reshape
from keras.models import Model
X3= np.random.random((250,100))
MAX_SEQUENCE_LENGTH = 1
EMBEDDING_DIM =100
embedding_layer = Embedding(250,
                            EMBEDDING_DIM,
                            weights=[X3],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,))
embedded_sequences = embedding_layer(sequence_input)
em = np.array(df_train_r['user'].tolist())

# COMMAND ----------

y= np.array(df_train_r['current_s'].tolist())
y= y.reshape(len(y),1)
y_class = []
for g in range(0, len(y)):
    tmp = [0] * 5
    tmp[y[g]-1] = 1
    y_class.append(tmp)
y_class = np.array(y_class).reshape(len(y_class),1,5)

# COMMAND ----------

#building the model in Keras with 2 models sharing the weights

from keras.layers import Input, Convolution2D, Dense, merge, Flatten, Dropout, MaxPooling2D, Input, Reshape
from keras.models import Model

pool_size = (2, 1)
max_val=5
nb_epoch=10
batch_size=128
# convolution kernel size
nb_filters = 1200
kernel_size1 = (1, 100)

print('start...')

#Input1
img1 = Input(shape=(max_val, 100,1))#Just the shape matters hence addition of 1 

#Convolution

img1_layer1 = Convolution2D(nb_filters, kernel_size1[0], kernel_size1[1],
                        border_mode='valid',activation='relu')(img1)

#Aggregating the filter output by flattening and comparing the value
img1_layer2= Flatten()(img1_layer1)
img1_layer3 = Dense(5)(img1_layer2)
#Input2

img2 = Input(shape=(5,))#Just the shape matters hence addition of 1 

# Merging them

merge_layer1 = merge([img1_layer3, img2], mode = 'sum')
merge_layer1_expand = Dense(100)(merge_layer1 )
reshape = Reshape((1,100))(merge_layer1_expand)
# Bringing embedding layer ,  current movie and the flag
img3 = Input(shape=(1,100))
merge_layer2 = merge([embedded_sequences, img3,], mode = 'sum')
merge_layer3 = merge([merge_layer2 , reshape], mode = 'sum')
#img1_layer4 = Dense(5, activation ='relu')(merge_layer2)

predictions = Dense(5, activation='softmax')(merge_layer3)


model = Model(input=[img1, img2, sequence_input,img3 ], output=[predictions] )

model.compile(loss='categorical_crossentropy',
              optimizer= 'adagrad',
              metrics=['accuracy'])

model.fit([X1[:640000],X2[:640000], em[:640000],X4[:640000]], y_class[:640000], batch_size=batch_size, nb_epoch=nb_epoch,
           validation_data=([X1[640000:],X2[640000:], em[640000:],X4[640000:]],[y_class[640000:]]),verbose=1)
print('done')

# COMMAND ----------

# Prediction on 4000 Recommendations
predicted = model.predict([X1[640000:],X2[640000:], em[640000:],X4[640000:]], verbose=0) 
pr1 = []
pr2 =[]
for rows in range(0,len(predicted)):
    score= np.argsort(predicted[rows])[-2:][0]
    pr1.append(score[-1]+1)
    pr2.append(score[-2]+1)
df_visual = pd.DataFrame()
df_visual['User'] = em[640000:]
df_visual['Actual'] = y[640000:,0]
df_visual['Rating1_score'] = predicted[:,0,0]
df_visual['Rating2_score'] = predicted[:,0,1]
df_visual['Rating3_score'] = predicted[:,0,2]
df_visual['Rating4_score'] = predicted[:,0,3]
df_visual['Rating5_score'] = predicted[:,0,4]
df_visual['First_Predicted_Rating'] =  pr1
df_visual['Second_Predicted_Rating'] =  pr2
df_visual_good = df_visual[(df_visual['Actual'] == df_visual['First_Predicted_Rating'] ) | (df_visual['Actual'] == df_visual['Second_Predicted_Rating'] )]
df_visual_bad = df_visual[(df_visual['Actual'] != df_visual['First_Predicted_Rating'] ) & (df_visual['Actual'] != df_visual['Second_Predicted_Rating'] )]

print('done')

# COMMAND ----------

df_visual

# COMMAND ----------

df_visual_good

# COMMAND ----------

df_visual_bad

# COMMAND ----------


