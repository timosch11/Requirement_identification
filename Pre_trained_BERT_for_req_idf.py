#Imports
from random import randrange
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd 
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix
import re
from sklearn.metrics import multilabel_confusion_matrix
from sklearn import metrics
import string
from sklearn.metrics import accuracy_score
from bs4 import BeautifulSoup as bs
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow.keras import layers, Model
from tensorflow.keras import regularizers
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.losses import CategoricalCrossentropy
import seaborn as sns
from wordcloud import WordCloud
from official.nlp.optimization import create_optimizer
import matplotlib.pyplot as plt
from matplotlib import rcParams
from platform import python_version
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

#Load NLP Models and set Themes for Visuals
nlp = spacy.load('en_core_web_sm') 
pd.set_option('display.max_colwidth', None) 
rcParams['figure.figsize'] = (10, 6) 
sns.set_theme(palette='muted', style='whitegrid') 

#Load Data
df = pd.read_csv("/home/tschwind/project/data/vector-requirements/RequirmentsForBert.csv")
df_dic = pd.DataFrame()

#Data Cleaning and Preprocessing
df_dic["fixed.ReqIF.ChapterName"] = pd.DataFrame(df["fixed.ReqIF.ChapterName"].unique())
df_dic["ClassNo"] = df_dic.index
df = pd.merge(df,df_dic,left_on="fixed.ReqIF.ChapterName", right_on="fixed.ReqIF.ChapterName")
df.drop(columns=["Unnamed: 0","Unnamed: 0.1"], inplace=True)
df.reset_index(inplace=True)
duplicates = df[df.duplicated(['ReqIF.Text'], keep=False)]
df.drop_duplicates(["ReqIF.Text"], inplace=True, ignore_index=True)
df.dropna(inplace=True)
df = df.sample(frac=1).reset_index(drop=True)
def lemmatize_text(text, nlp=nlp):
    doc = nlp(text)    
    lemma_sent = [i.lemma_ for i in doc if not i.is_stop]    
    
    return ' '.join(lemma_sent)  

def standardize_text(text_data):    
    url_pattern = re.compile(r'(?:\@|http?\://|https?\://|www)\S+')
    
    digit_pattern = re.compile(r'[\d]+')
    url_strip = text_data.apply(lambda x: re.sub(url_pattern, '', x) if pd.isna(x) != True else x)
    lowercase = url_strip.apply(lambda x: str.lower(x) if pd.isna(x) != True else x)       
    digit_strip = lowercase.apply(lambda x: re.sub(digit_pattern, '', x) if pd.isna(x) != True else x) 
    punct_strip = digit_strip.apply(lambda x: re.sub(f'[{re.escape(string.punctuation)}]', '', x) if pd.isna(x) != True else x) 
    lemma_and_stop = punct_strip.apply(lambda x: lemmatize_text(x) if pd.isna(x) != True else x)

    return lemma_and_stop

clean_text = np.asarray(standardize_text(df["ReqIF.Text"]))
print(clean_text.shape)
df.drop(columns=["ReqIF.Text"], inplace=True)
df["ReqIF.Text"] = clean_text

#Prepare Data to correct Representation 
def configure_dataset(dataset, shuffle=False, test=False):
     # Configure the tf dataset for cache, shuffle, batch, and prefetch
    if shuffle:
        dataset = dataset.cache()\
                         .shuffle(BUFFER_SIZE, reshuffle_each_iteration=True)\
                         .batch(BATCH_SIZE, drop_remainder=True).prefetch(AUTOTUNE)
    elif test:
        dataset = dataset.cache()\
                         .batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    else:
        dataset = dataset.cache()\
                         .batch(BATCH_SIZE, drop_remainder=True).prefetch(AUTOTUNE)
    return dataset  

enc = OneHotEncoder(sparse=False)   #Load 1HE
train, validate, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])
test_x, test_y = test["ReqIF.Text"], enc.fit_transform(test["fixed.ReqIF.ChapterName"].values.reshape(-1, 1))
test_ds = tf.data.Dataset.from_tensor_slices(test_x)
test_ds = configure_dataset(test_ds, test=True)


#Prepare K-Fold (k-fold cross validation is a procedure used to estimate the skill of the model on new data)
NUM_FOLDS = 2  #Prepare folding
df.reset_index(inplace=True)
df["fold_id"] = df["index"].apply(lambda x: randrange(0,NUM_FOLDS))  
#print(df.head(10))  #just to look

#Set Hyperparameters
AUTOTUNE = tf.data.experimental.AUTOTUNE
BUFFER_SIZE = 1000
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 5e-5

#Load Pretrained BERT NLP Model
bert_preprocessor = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3', name='BERT_preprocesser')
bert_encoder = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4', trainable=True, name='BERT_encoder')
# Keyword embedding layer
nnlm_embed = hub.KerasLayer('https://tfhub.dev/google/nnlm-en-dim50/2', name='embedding_layer')
test_preds = 0
acc = []

#Build Neural Network ontop of BERT Encoder Output
def build_model():
    
    # Construct text layers
    text_input = layers.Input(shape=(), dtype=tf.string, name='Content') # Name matches df heading
    encoder_inputs = bert_preprocessor(text_input)
    encoder_outputs = bert_encoder(encoder_inputs)
    # pooled_output returns [batch_size, hidden_layers]
    pooled_output = encoder_outputs["pooled_output"]          
    bert_dropout = layers.Dropout(0.1, name='BERT_dropout')(pooled_output)   
    dense = layers.Dense(128, activation='elu', kernel_regularizer=regularizers.l2(1e-4))(bert_dropout)
    dropout = layers.Dropout(0.5, name='merged_dropout')(dense)    
    clf = layers.Dense(15, activation='softmax', name='classifier')(dropout)  #15 Class Classification
    
    return Model([text_input], clf, name='BERT_classifier')






#Start Training and Evaluation Phase

for i in range(0, NUM_FOLDS):
    train_x, train_y = train["ReqIF.Text"][train["fold_id"] != i], enc.fit_transform(train["fixed.ReqIF.ChapterName"][train["fold_id"] != i].values.reshape(-1, 1))  
    val_x, val_y = validate["ReqIF.Text"][validate["fold_id"] == i], enc.fit_transform(validate["fixed.ReqIF.ChapterName"][validate["fold_id"] == i].values.reshape(-1, 1))
    train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    val_ds = tf.data.Dataset.from_tensor_slices((val_x, val_y))
    train_ds = configure_dataset(train_ds, shuffle=True)
    val_ds = configure_dataset(val_ds)  
    STEPS_PER_EPOCH = int(train_ds.unbatch().cardinality().numpy() / BATCH_SIZE)
    VAL_STEPS = int(val_ds.unbatch().cardinality().numpy() / BATCH_SIZE)
    # Calculate the train and warmup steps for the optimizer
    TRAIN_STEPS = STEPS_PER_EPOCH * EPOCHS
    WARMUP_STEPS = int(TRAIN_STEPS * 0.1)
    
    bert_classifier = build_model()  #Builds the defined Model
    adamw_optimizer = create_optimizer(   #Create optimizer for Gradient Descent Algorithm
    init_lr=LEARNING_RATE,
    num_train_steps=TRAIN_STEPS,
    num_warmup_steps=WARMUP_STEPS
    )
    bert_classifier.compile(   #Compile Model
    loss=CategoricalCrossentropy(), 
    optimizer= adamw_optimizer,
    metrics=[BinaryAccuracy(name='accuracy')]
    )

    

    history = bert_classifier.fit(   #Start Training
    train_ds, 
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,    
    validation_data= val_ds,
    validation_steps=VAL_STEPS
    )
	
    val_target = np.asarray([i[1] for i in list(val_ds.unbatch().as_numpy_iterator())])
    val_target = val_target.argmax(axis=-1)   #Get real class label for validation data
    val_predict = bert_classifier.predict(val_ds).argmax(axis=1)   #Get model prediction for validation data
    print(metrics.classification_report(val_target, val_predict, digits=3, target_names=enc.categories_[0]))
    
    test_preds = bert_classifier.predict(test_ds, batch_size=1024, verbose=2)
    acc = accuracy_score(test_y.argmax(axis=-1), test_preds.argmax(axis=-1))
    print("accuracy on testset:   %0.3f" % acc)   #Get Accuracy for validation Data

    #Create a Confusion Matrix Plot for Fold
	cm = confusion_matrix(val_target, val_predict)
	class_names = df["fixed.ReqIF.ChapterName"].unique()
	fig = plt.figure(figsize=(16, 14))
	ax= plt.subplot()
	sns.heatmap(cm, annot=True, ax = ax, fmt = 'g'); #annot=True to annotate cells
	# labels, title and ticks
	ax.set_xlabel('Predicted', fontsize=20)
	ax.xaxis.set_label_position('bottom')
	plt.xticks(rotation=90)
	ax.xaxis.set_ticklabels(class_names, fontsize = 10)
	ax.xaxis.tick_bottom()

	ax.set_ylabel('True', fontsize=20)
	ax.yaxis.set_ticklabels(class_names, fontsize = 10)
	plt.yticks(rotation=0)

	plt.title('Refined Confusion Matrix', fontsize=20)

	plt.show()