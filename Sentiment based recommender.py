#!/usr/bin/env python
# coding: utf-8

# # <span style='color:Blue'>**Sentiment Based Product Recommendation System**

# ## <span style='color:Red'>Problem Statement : - 
# ### <span style='color:blue'>An e-commerce company named 'Ebuss' has captured a huge market share in many fields, and it sells the products in various categories such as household essentials, books, personal care products, medicines, cosmetic items, beauty products, electrical appliances, kitchen and dining products and health care products.
# 
# ### <span style='color:blue'> And we have to build a model that will improve the recommendations given to the users given their past reviews and ratings. 

# ## <span style='color:Green'>Solution: -
# ### <span style='color:blue'>We will be building a sentiment-based product recommendation system, which includes the following tasks.
# 
#     1.Data sourcing and sentiment analysis
#     2.Building a recommendation system
#     3.Improving the recommendations using the sentiment analysis model
#     4.Deploying the end-to-end project with a user interface

# In[1]:


# importing the libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# # 1. Reading the data

# In[2]:


ebuss = pd.read_csv('sample30.csv', sep=",", encoding="ISO-8859-1", header=0)


# In[3]:


ebuss.head()


# In[4]:


ebuss.shape


# In[5]:


ebuss.describe()


# In[6]:


ebuss.info()


# # 2. Data Cleaning and Pre-Processing

# In[7]:


# checking for column-wise null percentages here

round(100*(ebuss.isnull().sum()/len(ebuss.index)), 2)


# ### Here I will remove the  rows for which the missing percentage in columns is very low as it will not remove lot of data points. For the two columns with missing values of about 94 and 100%, I'll drop those columns as imputing them will create huge outliers and data imbalance. 

# In[8]:


#drop rows with the missing values in column 'manufacturer'
ebuss = ebuss[pd.notnull(ebuss['manufacturer'])]


# In[ ]:





# In[9]:


#drop rows with the missing values in column 'reviews_title'
ebuss = ebuss[pd.notnull(ebuss['reviews_title'])]


# In[10]:


#drop rows with the missing values in column 'reviews_username'
ebuss = ebuss[pd.notnull(ebuss['reviews_username'])]


# In[11]:


#dropping columns 'reviews_userProvince' & 'reviews_userCity'
ebuss.drop(['reviews_userProvince', 'reviews_userCity'], axis = 1, inplace = True)


# ### Now we see that the column `reviews_date` is an `object` data type. So we will change that also. 
# - Also there are some entries in the reviews_data column that say ' hooks slide or swivel into any desired position." '. So I'll first convert those to NaN values and then correct the data type and then will remove the rows missing the values. 

# In[12]:


ebuss['reviews_date'].replace(' hooks slide or swivel into any desired position."',np.NaN,inplace = True)
ebuss.reviews_date = pd.to_datetime(ebuss.reviews_date)


# In[13]:


#drop rows with the missing values in column 'reviews_date'
ebuss = ebuss[pd.notnull(ebuss['reviews_date'])]


# In[14]:


ebuss.info()


# In[15]:


# checking for column-wise null percentages here

round(100*(ebuss.isnull().sum()/len(ebuss.index)), 2)


# ### For columns `reviews_didPurchase` & `reviews_doRecommend`, I'll replace the missing values as `False(boolean)`.

# In[16]:


#replacing 'reviews_didPurchase' and & 'reviews_doRecommend' missing value (NaN) with 'False'
ebuss['reviews_didPurchase'].fillna(False,inplace=True)
ebuss['reviews_doRecommend'].fillna(False,inplace=True)


# In[ ]:





# In[17]:


# Write your code for column-wise null percentages here

round(100*(ebuss.isnull().sum()/len(ebuss.index)), 2)


# In[18]:


ebuss.info()


# ### The percentage shows 0.0 for missing values. But info shows there is one missing value in uder_sentiment. 

# In[19]:


ebuss['user_sentiment'].isna().sum()


# In[20]:


#Filling with negative
ebuss['user_sentiment'].fillna('Negative',inplace=True)


# In[21]:


#verifying
ebuss['user_sentiment'].isna().sum()


# In[22]:


ebuss.info()


# ### Now we have data frame with no missing values. 

# In[23]:


#checking shape
ebuss.shape


# ### So we have 29559 rows retained out of 30000, which is pretty good.

# In[24]:


ebuss.head()


# In[25]:


ebuss.info()


# In[26]:


ebuss.user_sentiment.unique()


# ## Text preprocessing

# ### <span style='color:green'>There are two text columns that are most importanat the sentiment analysis, i.e `reviews_text` & `reviews_title`. So all the text preprocessing will happen onto these 2 columns.
# 1. I'll convert them to lower cases. 
# 2. I'll remove the stop words & punctuations. 
# 3. I'll concatenate both the columns to have one single column for feature creation, `clean_review`
# 4. I'll remove 10 most common used words.
# 5. I'll run stemmer and lemmetizer the newly created `clean_review`column

# ###  converting to lower case

# In[27]:



ebuss['reviews_text'] = ebuss['reviews_text'].str.lower()


# In[28]:


ebuss['reviews_title'] = ebuss['reviews_title'].str.lower()


# In[29]:


ebuss.head()


# ### Removing punctuations

# In[30]:


import string 
PUNCT_TO_REMOVE = string.punctuation


# In[31]:


PUNCT_TO_REMOVE


# In[32]:


def remove_punctuation(text):
    return text.translate(str.maketrans('','', PUNCT_TO_REMOVE))


# In[33]:


ebuss['reviews_text'] = ebuss['reviews_text'].apply(lambda text: remove_punctuation(text))
ebuss['reviews_title'] = ebuss['reviews_title'].apply(lambda text: remove_punctuation(text))


# In[34]:


ebuss.head()


# ### Removing stop words

# In[35]:


from nltk.corpus import stopwords
", ".join(stopwords.words('english'))


# In[36]:


STOPWORDS = set(stopwords.words('english'))


# In[37]:


def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])


# In[38]:


ebuss['reviews_text'] = ebuss['reviews_text'].apply(lambda text: remove_stopwords(text))


# In[39]:


ebuss.head()


# ### Concatenation of the columns

# In[40]:


ebuss['clean_review'] = ebuss[['reviews_title', 'reviews_text']].apply(lambda x: " ".join(str(y) for y in x if str(y) != 'nan'), axis = 1)
ebuss = ebuss.drop(['reviews_title', 'reviews_text'], axis = 1)
ebuss.head()


# In[41]:


# calculating tokens in order to measure of final cleaned tokens

from nltk.tokenize import word_tokenize
raw_tokens=len([w for t in (ebuss["clean_review"].apply(word_tokenize)) for w in t])
print('Number of tokens: {}'.format(raw_tokens))


# ### Stemming and Lemmatizing (removing special characters, non-ascii words, numbers etc)

# In[42]:



import nltk
import re, string, unicodedata
from bs4 import BeautifulSoup
from nltk import word_tokenize, sent_tokenize
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from contractions import contractions_dict


# In[43]:


# special_characters removal
def remove_special_characters(text, remove_digits=True):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text


def remove_non_ascii(words):
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words


def replace_numbers(words):
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words


def stem_words(words):
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems


def lemmatize_verbs(words):
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def normalize(words):
    words = remove_non_ascii(words)
    return words

def lemmatize(words):
    lemmas = lemmatize_verbs(words)
    return lemmas

def normalize_and_lemmaize(input):
    sample = remove_special_characters(input)
    words = nltk.word_tokenize(sample)
    words = normalize(words)
    lemmas = lemmatize(words)
    return ' '.join(lemmas)


# In[44]:


ebuss['clean_review'] = ebuss['clean_review'].map(lambda text: normalize_and_lemmaize(text))


# In[45]:


ebuss.head()


# #### Removing top 10 most common words

# In[46]:


from collections import Counter
cnt = Counter()
for text in ebuss['clean_review'].values:
    for word in text.split():
        cnt[word] += 1
        
cnt.most_common(10)


# In[47]:


# freqwords = set([w for (w, wc) in cnt.most_common[10]])
# def remove_freqwords:
#     return "".join([word for word in str(text).split() if word not in freqwords])
# ebuss['cleaned_reviews'] = ebuss['cleaned_reviews'].apply(lambda text: remove_freqwords(text))


# ### <span style='color:red'>Here I could have removed the frequent words, however, some of the words are great, love, clean, and these positive words can effect the sentiment analysis. So I have decided not to remove these words. 

# In[48]:


ebuss.head()


# In[49]:


# calculating tokens in order to measure of cleaned tokens

from nltk.tokenize import word_tokenize
raw_tokens=len([w for t in (ebuss["clean_review"].apply(word_tokenize)) for w in t])
print('Number of tokens: {}'.format(raw_tokens))


# In[50]:


from nltk.tokenize import RegexpTokenizer
def RegExpTokenizer(Sent):
    tokenizer = RegexpTokenizer(r'\w+')
    return tokenizer.tokenize(Sent)

ListWords = []
for m in ebuss['clean_review']:
    n = RegExpTokenizer(str(m))
    ListWords.append(n)
print(ListWords[10])


# In[51]:


#Checking for all words
from nltk import FreqDist
def Bag_Of_Words(ListWords):
    all_words = []
    for m in ListWords:
        for w in m:
            all_words.append(w.lower())
    all_words1 = FreqDist(all_words)
    return all_words1


# ## Visual analysis of the top words and top brands

# In[52]:


plt.figure(figsize = (10,8))
import seaborn as sns
from sklearn.manifold import TSNE
all_words = Bag_Of_Words(ListWords)
count = []
Words  = []
for i in all_words.most_common(10):
    count.append(i[1])
    Words.append(i[0])
sns.barplot(Words,count)
plt.show()


# In[53]:


top_brands = ebuss["brand"].value_counts()
plt.figure(figsize=(10,8))
top_brands[:10].plot(kind='bar')
# sns.barplot(top_brands[:10],ebuss["brand"].value_counts())
plt.title("Total reviews for top 10 brands")
plt.xlabel('Brand Name')
plt.ylabel('Number of Reviews')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# # Feature extraction

# In[54]:


# creating a column token
def token (text):
    token = [w for w in nltk.word_tokenize(text)]
    return token

ebuss['token'] = ebuss['clean_review'].apply(token)


# In[55]:


ebuss.head()


# ### Now we have converted the cleaned reviews into tokens. 
# ### <span style='color:green'>We will now drop the unnecesarry columns and will keep only `name`, `reviews_rating`, `reviews_username`, `user_sentiment`, `clean_review` & `token` columns, as we will need only these columns for feature extraction and later develop our sentiment model. 

# In[56]:


#Dropping unncessary columns
df_final = ebuss[['name','reviews_rating',
       'reviews_username', 'user_sentiment', 'clean_review',
       'token']]


# In[57]:


df_final.head()


# ### <span style='color:green'> Now we also need to change the user_sentiment column to binary column. I'll impute a `1` for `Positive`, and a `0` for `Negative`.

# In[58]:


df_final['user_sentiment'] = df_final['user_sentiment'].apply(lambda x: 1 if x == 'Positive' else 0)


# In[59]:


df_final.head()


# # Splitting into test and train sets

# In[60]:


# Importing Libraries 
from sklearn.model_selection import train_test_split


# In[61]:


# Splitting the Data Set into Train and Test Sets
X = df_final['clean_review']
y = df_final['user_sentiment']


# In[62]:


# Splitting Dataset into train and test set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)


# In[63]:


# checking train and test set shape
print ('Train Set Shape:{}\nTest Set Shape:{}'.format(X_train.shape, X_test.shape))


# In[64]:


# checking train and test set shape
print ('Train Set Shape:{}\nTest Set Shape:{}'.format(y_train.shape, y_test.shape))


# ## Checking for class imbalance

# In[65]:


y_train.value_counts()/len(y_train)


# In[66]:


y_test.value_counts()/len(y_test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## <span style='color:green'> We can see here that there is a large class imbalance between 1 and 0. So we will handle the class imbalance also.
# ## But bfore that, we need to perform feature extraction using TF-IDF .

# ## TF-IDF

# In[67]:


#importing library
from sklearn.feature_extraction.text import TfidfVectorizer


# In[68]:


# Create the word vector with TF-IDF Vectorizer
tfidf_vect = TfidfVectorizer(ngram_range=(1, 1),max_features=500)
tfidf_vect_train = tfidf_vect.fit_transform(X_train)
tfidf_vect_train = tfidf_vect_train.toarray()
tfidf_vect_test = tfidf_vect.transform(X_test)
tfidf_vect_test = tfidf_vect_test.toarray()


# In[69]:


# Printing vocabulary length
print('Vocabulary length :', len(tfidf_vect.get_feature_names()))


# In[70]:


model_tf_idf = list()
resample_tf_idf = list()
precision_tf_idf = list()
precision_neg_tf_idf = list()
recall_neg_tf_idf = list()
recall_tf_idf = list()
F1score_tf_idf = list()
AUCROC_tf_idf = list()


# In[71]:


def test_eval_tf_idf(clf_model,y_pred,y_prob, y_test, algo=None, sampling=None):
    # Test set prediction
    #y_prob=clf_model.predict_proba(X_test)
    #y_pred=clf_model.predict(X_test)

    print('Confusion Matrix')
    print('='*60)
    print(confusion_matrix(y_test,y_pred),"\n")
    print('Classification Report')
    print('='*60)
    print(classification_report(y_test,y_pred),"\n")
    print('AUC-ROC')
    print('='*60)
    print(roc_auc_score(y_test, y_prob[:,1]))
    score = f1_score(y_test, y_pred, average = 'weighted')
    # Printing evaluation metric (f1-score) 
    print("f1 score: {}".format(score))
          
    model_tf_idf.append(algo)
    precision_tf_idf.append(precision_score(y_test,y_pred))
    recall_tf_idf.append(recall_score(y_test,y_pred))
    F1score_tf_idf.append(score)
    AUCROC_tf_idf.append(roc_auc_score(y_test, y_prob[:,1]))
    resample_tf_idf.append(sampling)
    precision_neg_tf_idf.append(get_precision_negative_class(y_test,y_pred))
    recall_neg_tf_idf.append(get_recall_negative_class(y_test,y_pred))


# In[72]:


# Assign feature names of vector into a variable
vocab = tfidf_vect.get_feature_names()


# In[73]:


pd.DataFrame(tfidf_vect_train, columns = vocab).head()


# ### Using unsampled original data

# In[78]:


import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# In[79]:


clf_LR_tfidf = LogisticRegression(random_state = 123)
clf_RF_tfidf = RandomForestClassifier(random_state = 123)
clf_XGB_tfidf = XGBClassifier(random_state = 123)


# In[80]:


# #hyperparameter
lr_params = {'penalty': ['l1', 'l2'], 
             'solver': ['lbfgs', 'liblinear'], 
             'C': [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1]}
rf_params = {'n_estimators':[200,300,500], 'max_depth':[4,6]
              , 'min_samples_split':[100,300], 'min_samples_leaf':[50,100]}
xgb_params = {'n_estimators':[200],
    "learning_rate"    : [0.1, 0.30],
 "max_depth"        : [6,10],
 "gamma"            : [0,10,30]}


# In[81]:


# storing model object and hyperparameters in a dictionary
model_dict = {
             'logistic_reg': [clf_LR_tfidf, lr_params], 
             'random_forest': [clf_RF_tfidf, rf_params],
              'xgboost': [clf_XGB_tfidf, xgb_params]}


# In[84]:


# list to store best model and results
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

trained_models = []
# train each ensemble model one by one
# tqdm to show progress
for model_name in tqdm(model_dict):
    print('Training %s model ' %model_name)
    model_entity = model_dict[model_name]
    estimator = model_entity[0]
    estimator_params = model_entity[1]
    # Grid search for hyperparameter tuning
    # Randomized search for faster hyperparameter tuning (Use n_iter=10 or any number to limit in case of randomized search, use n_jobs=-1 to use all cores, use k in KFold() if needed)
    # StratiedfiedKFold for model cross-validation
    estimator_grid_model = GridSearchCV(estimator=estimator, param_grid=estimator_params, cv=StratifiedKFold(), n_jobs=-1, verbose=4)
#     estimator_grid_model = RandomizedSearchCV(estimator=estimator, param_distributions=estimator_params, cv=StratifiedKFold(), n_jobs=1, n_iter=10)
    estimator_grid_model.fit(tfidf_vect_train, y_train)
    print('\n %s Model training complete.' %model_name)
    # get test results
    y_pred_grid_model = estimator_grid_model.predict(tfidf_vect_test)
    # get test probabilities
    y_pred_prob_grid_model = estimator_grid_model.predict_proba(tfidf_vect_test)
    # store model name with best estimator and results in a dict object
    trained_model_dict = {}
    trained_model_dict['model_name'] = model_name
    trained_model_dict['best_estimator'] = estimator_grid_model.best_estimator_
    trained_model_dict['y_pred'] = y_pred_grid_model
    trained_model_dict['y_pred_prob'] = y_pred_prob_grid_model
    # append to results to list
    trained_models.append(trained_model_dict)


# ### Logistic Regression with TF-IDF

# In[85]:


clf_LR_tfidf = trained_models[0]['best_estimator']
y_pred_tfidf_LR = trained_models[0]['y_pred']
y_prob_tfidf_LR = trained_models[0]['y_pred_prob']


# In[97]:


from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

def get_recall_negative_class(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred).ravel()
    tn, fp, fn, tp = matrix
    recall_neg = tn / (tn + fp)
    return recall_neg

def get_precision_negative_class(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred).ravel()
    tn, fp, fn, tp = matrix
    precision_neg = tn / (tn + fn)
    return precision_neg


# In[98]:


test_eval_tf_idf(clf_LR_tfidf,y_pred_tfidf_LR,y_prob_tfidf_LR,y_test,'Logistic Regression TF-IDF','actual')


# In[99]:


# save the model to disk
import pickle
filename = 'LR_tfidfU_actual.sav'
pickle.dump(clf_LR_tfidf, open(filename, 'wb'))


# ### Random Forest with TF IDF

# In[100]:


clf_RF_tfidf = trained_models[1]['best_estimator']
y_pred_tfidf_RF = trained_models[1]['y_pred']
y_prob_tfidf_RF = trained_models[1]['y_pred_prob']


# In[101]:


test_eval_tf_idf(clf_RF_tfidf,y_pred_tfidf_RF,y_prob_tfidf_RF,y_test,'Random Forest TF-IDF','actual')


# In[102]:


# save the model to disk
filename = 'RF_tfidfU_actual.sav'
pickle.dump(clf_RF_tfidf, open(filename, 'wb'))


# ### XGBoost with TFIDF

# In[103]:


clf_XGB_tfidf = trained_models[2]['best_estimator']
y_pred_tfidf_XGB = trained_models[2]['y_pred']
y_prob_tfidf_XGB = trained_models[2]['y_pred_prob']


# In[104]:


test_eval_tf_idf(clf_XGB_tfidf,y_pred_tfidf_XGB,y_prob_tfidf_XGB,y_test,'XGBoost TF-IDF','actual')


# In[105]:


# save the model to disk
filename = 'XGB_tfidfU_actual.sav'
pickle.dump(clf_XGB_tfidf, open(filename, 'wb'))


# ### Naive Bayes with TF IDF

# In[108]:


from sklearn.naive_bayes import MultinomialNB
clf_NB_tfidf = MultinomialNB()


# In[109]:


clf_NB_tfidf.fit(tfidf_vect_train,y_train)


# In[110]:


y_prob_tfidf_NB=clf_NB_tfidf.predict_proba(tfidf_vect_test)
y_pred_tfidf_NB=clf_NB_tfidf.predict(tfidf_vect_test)


# In[111]:


test_eval_tf_idf(clf_NB_tfidf,y_pred_tfidf_NB,y_prob_tfidf_NB,y_test,'Naive Bayes TF-IDF','actual')


# In[112]:


# save the model to disk
filename = 'NB_tfidfU_actual.sav'
pickle.dump(clf_XGB_tfidf, open(filename, 'wb'))


# ### Using SMOTE Technique

# In[113]:


from imblearn.over_sampling import SMOTE

counter = Counter(y_train)
print('Before',counter)
# oversampling the train dataset using SMOTE
smt = SMOTE()
#X_train, y_train = smt.fit_resample(X_train, y_train)
X_train_sm, y_train_sm = smt.fit_resample(tfidf_vect_train, y_train)

counter = Counter(y_train_sm)
print('After',counter)


# In[114]:


clf_LR_tfidf_smote = LogisticRegression(random_state = 123)
clf_RF_tfidf_smote = RandomForestClassifier(random_state = 123)
clf_XGB_tfidf_smote = XGBClassifier(random_state = 123)


# In[115]:


# storing model object and hyperparameters in a dictionary
model_dict_smote = {
             'logistic_reg': [clf_LR_tfidf_smote, lr_params], 
             'random_forest': [clf_RF_tfidf_smote, rf_params],
              'xgboost': [clf_XGB_tfidf_smote, xgb_params]}


# In[116]:


# list to store best model and results
trained_models_smote = []
# train each ensemble model one by one
# tqdm to show progress
for model_name in tqdm(model_dict_smote):
    print('Training %s model ' %model_name)
    model_entity = model_dict_smote[model_name]
    estimator = model_entity[0]
    estimator_params = model_entity[1]
    # Grid search for hyperparameter tuning
    # Randomized search for faster hyperparameter tuning (Use n_iter=10 or any number to limit in case of randomized search, use n_jobs=-1 to use all cores, use k in KFold() if needed)
    # StratiedfiedKFold for model cross-validation
    estimator_grid_model = GridSearchCV(estimator=estimator, param_grid=estimator_params, cv=StratifiedKFold(), n_jobs=-1, verbose=4)
    #estimator_grid_model = RandomizedSearchCV(estimator=estimator, param_distributions=estimator_params, cv=StratifiedKFold(), n_jobs=1, n_iter=10)
    estimator_grid_model.fit(X_train_sm, y_train_sm)
    print('\n %s Model training complete.' %model_name)
    # get test results
    y_pred_grid_model = estimator_grid_model.predict(tfidf_vect_test)
    # get test probabilities
    y_pred_prob_grid_model = estimator_grid_model.predict_proba(tfidf_vect_test)
    # store model name with best estimator and results in a dict object
    trained_model_dict = {}
    trained_model_dict['model_name'] = model_name
    trained_model_dict['best_estimator'] = estimator_grid_model.best_estimator_
    trained_model_dict['y_pred'] = y_pred_grid_model
    trained_model_dict['y_pred_prob'] = y_pred_prob_grid_model
    # append to results to list
    trained_models_smote.append(trained_model_dict)
# trained_models


# ### Logistic Regression with TF-IDF

# In[117]:


clf_LR_tfidf_smote = trained_models_smote[0]['best_estimator']
y_pred_tfidf_LR_smote = trained_models_smote[0]['y_pred']
y_prob_tfidf_LR_smote = trained_models_smote[0]['y_pred_prob']


# In[118]:


test_eval_tf_idf(clf_LR_tfidf_smote,y_pred_tfidf_LR_smote,y_prob_tfidf_LR_smote,y_test,'Logistic Regression TF-IDF','SMOTE')


# In[119]:


# save the model to disk
filename = 'LR_tfidfU_smote.sav'
pickle.dump(clf_LR_tfidf_smote, open(filename, 'wb'))


# ### Random Forest with TF-IDF

# In[120]:


clf_RF_tfidf_smote = trained_models_smote[1]['best_estimator']
y_pred_tfidf_RF_smote = trained_models_smote[1]['y_pred']
y_prob_tfidf_RF_smote = trained_models_smote[1]['y_pred_prob']


# In[121]:


test_eval_tf_idf(clf_RF_tfidf_smote,y_pred_tfidf_RF_smote,y_prob_tfidf_RF_smote,y_test,'Random Forest TF-IDF','SMOTE')


# In[122]:


# save the model to disk
filename = 'RF_tfidfU_smote.sav'
pickle.dump(clf_RF_tfidf_smote, open(filename, 'wb'))


# ### XGBoost with TF-IDF

# In[123]:


clf_XGB_tfidf_smote = trained_models_smote[2]['best_estimator']
y_pred_tfidf_XGB_smote = trained_models_smote[2]['y_pred']
y_prob_tfidf_XGB_smote = trained_models_smote[2]['y_pred_prob']


# In[124]:


test_eval_tf_idf(clf_XGB_tfidf_smote,y_pred_tfidf_XGB_smote,y_prob_tfidf_XGB_smote,y_test,'XGBoost TF-IDF','SMOTE')


# In[125]:


# save the model to disk
filename = 'XGB_tfidfU_smote.sav'
pickle.dump(clf_XGB_tfidf_smote, open(filename, 'wb'))


# ### Naive Bayes with TF-IDF

# In[126]:


clf_NB_tfidf_smote = MultinomialNB()


# In[127]:


clf_NB_tfidf_smote.fit(X_train_sm,y_train_sm)


# In[128]:


y_prob_tfidf_NB_smote=clf_NB_tfidf.predict_proba(tfidf_vect_test)
y_pred_tfidf_NB_smote=clf_NB_tfidf.predict(tfidf_vect_test)


# In[129]:


test_eval_tf_idf(clf_NB_tfidf_smote,y_pred_tfidf_NB_smote,y_prob_tfidf_NB_smote,y_test,'Naive Bayes TF-IDF','SMOTE')


# In[130]:


# save the model to disk
filename = 'NB_tfidf_smote.sav'
pickle.dump(clf_NB_tfidf_smote, open(filename, 'wb'))


# In[131]:


clf_eval_tf_idf_df = pd.DataFrame({'model':model_tf_idf,
                            'resample':resample_tf_idf,
                            'precision':precision_tf_idf,
                            'recall':recall_tf_idf,
                            'f1-score':F1score_tf_idf,
                            'AUC-ROC':AUCROC_tf_idf,
                           'recall_neg':recall_neg_tf_idf,
                           'precision_neg':precision_neg_tf_idf})


# In[132]:


clf_eval_tf_idf_df


# ### Upon inspecting all the scores, I have choosen the `XGBoost TF-IDF` with `SMOTE` resampling as the best model. 

# In[ ]:





# In[ ]:





# ### Reccomendation System

# - User based recommendation
# - User based prediction & evaluation
# - Item based recommendation
# - Item based prediction & evaluation
# 
# Different Approaches to develop Recommendation System -
# 
# 1. Content Based Recommendation System
# 
# 2. Collaborative filtering Recommendation System

# In[133]:


reviews_recco = df_final[['reviews_username','name','reviews_rating']]


# In[134]:


reviews_recco.head()


# In[135]:


"""## Dividing the dataset into train and test"""

# Test and Train split of the dataset.
train, test = train_test_split(reviews_recco, test_size=0.30, random_state=31)

print(train.shape)
print(test.shape)


# In[136]:


train['reviews_username'].value_counts()


# In[137]:


# Pivot the train ratings' dataset into matrix format in which columns are products and the rows are user IDs.
df_pivot = train.pivot_table(
    index='reviews_username',
    columns='name',
    values='reviews_rating'
).fillna(0)

df_pivot.head(3)


# ### Creating dummy train & dummy test dataset
# These dataset will be used for prediction 
# - Dummy train will be used later for prediction of the product which has not been rated by the user. To ignore the products rated by the user, we will mark it as 0 during prediction. The products not rated by user is marked as 1 for prediction in dummy train dataset. 
# 
# - Dummy test will be used for evaluation. To evaluate, we will only make prediction on the products rated by the user. So, this is marked as 1. This is just opposite of dummy_train.

# In[138]:


# Copy the train dataset into dummy_train
dummy_train = train.copy()

dummy_train.head()

# The products not rated by user is marked as 1 for prediction. 
dummy_train['reviews_rating'] = dummy_train['reviews_rating'].apply(lambda x: 0 if x>=1 else 1)


# In[139]:


dummy_train.head()


# In[140]:


# Convert the dummy train dataset into matrix format.
dummy_train = dummy_train.pivot_table(
    index='reviews_username',
    columns='name',
    values='reviews_rating',
).fillna(1)

dummy_train.head()


# **Cosine Similarity**
# 
# Cosine Similarity is a measurement that quantifies the similarity between two vectors [Which is Rating Vector in this case] 
# 
# **Adjusted Cosine**
# 
# Adjusted cosine similarity is a modified version of vector-based similarity where we incorporate the fact that different users have different ratings schemes. In other words, some users might rate items highly in general, and others might give items lower ratings as a preference. To handle this nature from rating given by user , we subtract average ratings for each user from each user's rating for different products.
# 
# ### User Similarity Matrix
# 
# #### Using Cosine Similarity

# In[141]:


from sklearn.metrics.pairwise import pairwise_distances

# Creating the User Similarity Matrix using pairwise_distance function.
user_correlation = 1 - pairwise_distances(df_pivot, metric='cosine')
user_correlation[np.isnan(user_correlation)] = 0
print(user_correlation)


# In[142]:


user_correlation.shape


# ### Using adjusted Cosine
# 
# #### Here, we are not removing the NaN values and calculating the mean only for the products rated by the user

# In[143]:


# Create a user-product matrix.
df_pivot_new = train.pivot_table(
    index='reviews_username',
    columns='name',
    values='reviews_rating'
)

df_pivot_new.head()


# ### Normalising the rating of the product for each user around 0 mean

# In[144]:


mean = np.nanmean(df_pivot_new, axis=1)
df_subtracted = (df_pivot_new.T-mean).T

df_subtracted.head()


# ### Finding cosine similarity

# In[145]:


from sklearn.metrics.pairwise import pairwise_distances

# Creating the User Similarity Matrix using pairwise_distance function.
user_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')
user_correlation[np.isnan(user_correlation)] = 0
print(user_correlation)


# ### Prediction - User User
# 
# Doing the prediction for the users which are positively related with other users, and not the users which are negatively related as we are interested in the users which are more similar to the current users. So, ignoring the correlation for values less than 0.

# In[146]:


user_correlation[user_correlation<0]=0
user_correlation


# ##### Rating predicted by the user (for products rated as well as not rated) is the weighted sum of correlation with the product rating (as present in the rating dataset).

# In[147]:


user_predicted_ratings = np.dot(user_correlation, df_pivot_new.fillna(0))
user_predicted_ratings


# In[148]:


user_predicted_ratings.shape


# #### Since we are interested only in the product not rated by the user, we will ignore the product rated by the user by making it zero.

# In[149]:


user_final_rating = np.multiply(user_predicted_ratings,dummy_train)
user_final_rating.head()


# ### Finding the top 5 recommendation for the user

# In[150]:


# Take the user name as input.
user_input = input("Enter your user name")
print(user_input)


# In[151]:


d = user_final_rating.loc[user_input].sort_values(ascending=False)[0:5]
d


# In[152]:


d = pd.merge(d,reviews_recco,left_on='name',right_on='name', how = 'left')
d.head()


# ### Evaluation - User User
# 
# Evaluation will we same as you have seen above for the prediction. The only difference being, you will evaluate for the product already rated by the user insead of predicting it for the product not rated by the user.

# In[153]:


# Find out the common users of test and train dataset.
common = test[test.reviews_username.isin(train.reviews_username)]
common.shape

common.head()


# In[154]:


# convert into the user-product matrix.
common_user_based_matrix = common.pivot_table(index='reviews_username', columns='name', values='reviews_rating')


# In[155]:


# Convert the user_correlation matrix into dataframe.
user_correlation_df = pd.DataFrame(user_correlation)


# In[156]:


user_correlation_df['reviews_username'] = df_subtracted.index

user_correlation_df.set_index('reviews_username',inplace=True)
user_correlation_df.head()


# In[157]:


common.head(1)


# In[158]:


list_name = common.reviews_username.tolist()

user_correlation_df.columns = df_subtracted.index.tolist()


# In[159]:


user_correlation_df_1 =  user_correlation_df[user_correlation_df.index.isin(list_name)]

user_correlation_df_1.shape


# In[160]:


user_correlation_df_2 = user_correlation_df_1.T[user_correlation_df_1.T.index.isin(list_name)]

user_correlation_df_3 = user_correlation_df_2.T


# In[161]:


user_correlation_df_3.head()


# In[162]:


user_correlation_df_3.shape


# In[163]:


user_correlation_df_3[user_correlation_df_3<0]=0

common_user_predicted_ratings = np.dot(user_correlation_df_3, common_user_based_matrix.fillna(0))
common_user_predicted_ratings


# In[164]:


dummy_test = common.copy()

dummy_test['reviews_rating'] = dummy_test['reviews_rating'].apply(lambda x: 1 if x>=1 else 0)

dummy_test = dummy_test.pivot_table(index='reviews_username', columns='name', values='reviews_rating').fillna(0)

dummy_test.shape


# In[165]:


common_user_based_matrix.head()


# In[166]:


dummy_test.head()


# In[167]:


common_user_predicted_ratings = np.multiply(common_user_predicted_ratings,dummy_test)

common_user_predicted_ratings.head()


# In[168]:


from sklearn.preprocessing import MinMaxScaler
from numpy import *

X  = common_user_predicted_ratings.copy() 
X = X[X>0]

scaler = MinMaxScaler(feature_range=(1, 5))
print(scaler.fit(X))
y = (scaler.transform(X))

print(y)


# In[169]:


common_ = common.pivot_table(index='reviews_username', columns='name', values='reviews_rating')


# In[170]:


# Finding total non-NaN value
total_non_nan = np.count_nonzero(~np.isnan(y))

rmse = (sum(sum((common_ - y )**2))/total_non_nan)**0.5
print(rmse)


# ### Using Item similarity
# 
# #### Item Based Similarity
# 
# Taking the transpose of the rating matrix to normalize the rating around the mean for different product name. In the user based similarity, we had taken mean for each user instead of each product.

# In[171]:


df_pivot_item = train.pivot_table(
    index='reviews_username',
    columns='name',
    values='reviews_rating'
    ).T

df_pivot_item.head()


# In[172]:


mean = np.nanmean(df_pivot_item, axis=1)
df_subtracted = (df_pivot_item.T-mean).T

df_subtracted.head()


# In[173]:


from sklearn.metrics.pairwise import pairwise_distances

# Item Similarity Matrix
item_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')
item_correlation[np.isnan(item_correlation)] = 0
print(item_correlation)

item_correlation.shape


# In[174]:


item_correlation[item_correlation<0]=0
item_correlation


# ### Prediction - Item Item
# 

# In[175]:


item_predicted_ratings = np.dot((df_pivot_item.fillna(0).T),item_correlation)
item_predicted_ratings

item_predicted_ratings.shape


# ### Filtering the rating only for the products not rated by the user for recommendation

# In[176]:


item_final_rating = np.multiply(item_predicted_ratings,dummy_train)
item_final_rating.head()


# ### Finding the top 5 recommendation for the user

# In[177]:


# Take the user ID as input
user_input = input("Enter your user name")
print(user_input)


# In[178]:


# Recommending the Top 5 products to the user.
d = item_final_rating.loc[user_input].sort_values(ascending=False)[0:5]
d


# ### Evaluation - Item Item
# 
# Evaluation will we same as you have seen above for the prediction. The only difference being, you will evaluate for the product already rated by the user insead of predicting it for the product not rated by the user.

# In[179]:


test.columns


# In[180]:


common =  test[test.name.isin(train.name)]
common.shape


# In[181]:


common.head(4)


# In[182]:


common_item_based_matrix = common.pivot_table(index='reviews_username', columns='name', values='reviews_rating').T
common_item_based_matrix.shape


# In[183]:


item_correlation_df = pd.DataFrame(item_correlation)
item_correlation_df.head(1)


# In[184]:


item_correlation_df['name'] = df_subtracted.index
item_correlation_df.set_index('name',inplace=True)
item_correlation_df.head()


# In[185]:


list_name = common.name.tolist()
item_correlation_df.columns = df_subtracted.index.tolist()
item_correlation_df_1 =  item_correlation_df[item_correlation_df.index.isin(list_name)]
item_correlation_df_2 = item_correlation_df_1.T[item_correlation_df_1.T.index.isin(list_name)]
item_correlation_df_3 = item_correlation_df_2.T


# In[186]:


item_correlation_df_3.head()


# In[187]:


item_correlation_df_3[item_correlation_df_3<0]=0
common_item_predicted_ratings = np.dot(item_correlation_df_3, common_item_based_matrix.fillna(0))
common_item_predicted_ratings


# In[188]:


common_item_predicted_ratings.shape


# #### Dummy test will be used for evaluation. To evaluate, we will only make prediction on the products rated by the user. So, this is marked as 1. This is just opposite of dummy_train

# In[189]:


dummy_test = common.copy()
dummy_test['reviews_rating'] = dummy_test['reviews_rating'].apply(lambda x: 1 if x>=1 else 0)
dummy_test = dummy_test.pivot_table(index='reviews_username', columns='name', values='reviews_rating').T.fillna(0)
common_item_predicted_ratings = np.multiply(common_item_predicted_ratings,dummy_test)


# #### The products not rated is marked as 0 for evaluation. And make the item- item matrix representaion

# In[190]:


common_ = common.pivot_table(index='reviews_username', columns='name', values='reviews_rating').T

from sklearn.preprocessing import MinMaxScaler
from numpy import *

X  = common_item_predicted_ratings.copy() 
X = X[X>0]

scaler = MinMaxScaler(feature_range=(1, 5))
print(scaler.fit(X))
y = (scaler.transform(X))

print(y)

# Finding total non-NaN value
total_non_nan = np.count_nonzero(~np.isnan(y))

rmse = (sum(sum((common_ - y )**2))/total_non_nan)**0.5
print(rmse)


# # Implementing the Sentiment Based Model on the Recco Engine

# - Best Model - XGBoost using SMOTE
# - Best Recco Algo - User-based recommendation system

# In[191]:


reviews = pd.DataFrame(X)


# In[192]:


reviews


# In[193]:


X = df_final['clean_review']


# In[194]:


df_final.to_csv("D:/project/final/df_final.csv",index = False)
reviews.to_csv("D:/project/final/reviews_cleaned.csv",index =False)


# In[195]:


y


# #### Loading the best model

# In[196]:


# load the model from disk
filename = "D:/project/final/XGB_tfidfU_smote.sav"
best_model = pickle.load(open(filename, 'rb'))


# In[197]:


best_model


# In[198]:


# Create the word vector with TF-IDF Vectorizer
tfidf_vect = TfidfVectorizer(ngram_range=(1, 1),max_features=500)
tfidf_vect_X = tfidf_vect.fit_transform(X)
tfidf_vect_X = tfidf_vect_X.toarray()


# In[199]:


tfidf_vect_X


# In[200]:


y_pred = best_model.predict(tfidf_vect_X)
y_prob = best_model.predict_proba(tfidf_vect_X)


# In[201]:


y_pred


# In[202]:


y


# In[203]:


df_final['predicted_sentiment'] = y_pred 


# In[204]:


df_final['predicted_sentiment'].value_counts()


# In[205]:


df_final


# In[206]:


product_sentiment_df = pd.DataFrame(df_final.groupby('name')['predicted_sentiment'].mean()).reset_index()


# In[207]:


product_sentiment_df


# In[208]:


reviews_recco = df_final[['reviews_username','name','reviews_rating']]
"""## Dividing the dataset into train and test"""

# Test and Train split of the dataset.
train, test = train_test_split(reviews_recco, test_size=0.30, random_state=42)

print(train.shape)
print(test.shape)


# In[209]:


# Create a user-product matrix.
df_pivot = reviews_recco.pivot_table(
    index='reviews_username',
    columns='name',
    values='reviews_rating'
)

df_pivot.head()


# In[210]:


mean = np.nanmean(df_pivot, axis=1)
df_subtracted = (df_pivot.T-mean).T

df_subtracted.head()


# In[211]:


from sklearn.metrics.pairwise import pairwise_distances

# Creating the User Similarity Matrix using pairwise_distance function.
user_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')
user_correlation[np.isnan(user_correlation)] = 0
print(user_correlation)


# In[212]:


user_correlation[user_correlation<0]=0
user_correlation


# In[213]:


user_predicted_ratings = np.dot(user_correlation, df_pivot.fillna(0))
user_predicted_ratings


# In[214]:


user_predicted_ratings.shape


# In[215]:


# Copy the train dataset into dummy_train
dummy = reviews_recco.copy()

dummy.head()

# The products not rated by user is marked as 1 for prediction. 
dummy['reviews_rating'] = dummy['reviews_rating'].apply(lambda x: 0 if x>=1 else 1)


# In[216]:


# Convert the dummy train dataset into matrix format.
dummy = dummy.pivot_table(
    index='reviews_username',
    columns='name',
    values='reviews_rating',
).fillna(1)

dummy.head()


# In[217]:


user_final_rating = np.multiply(user_predicted_ratings,dummy)
user_final_rating.head()


# ## Reccomending top 20 products

# In[219]:


# Take the user ID as input.
user_input = input("Enter your user name")
print(user_input)


# In[220]:


d = pd.DataFrame(user_final_rating.loc[user_input].sort_values(ascending=False)[0:20]).reset_index()
d


# In[221]:


final_recco = d.merge(product_sentiment_df,on='name',how='inner')


# In[222]:


final_recco = final_recco.sort_values(['predicted_sentiment'],ascending=False)[0:5]


# In[223]:


final_recco


# In[227]:


df_final.loc[(df_final['reviews_username'] == '00sab00') & (df_final['reviews_rating'] == 1),'name']


# In[ ]:





# In[ ]:




