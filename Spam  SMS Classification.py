#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import pandas as pd

# Read the dataset
dataset = pd.read_csv("SMSSpamCollection", sep="\t", names=['label', 'message'])

# Display the dataset
print(dataset)


# In[4]:


dataset.info()


# In[5]:


dataset.isnull().sum()


# In[6]:


dataset.describe()


# In[7]:


dataset["label"] = dataset["label"].map({'ham':0,'spam':1})
dataset


# In[8]:


dataset.describe()


# In[10]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(8,8))
p = sns.countplot(x = "label" , data = dataset)
p = plt.title('Countplot for Spam vs Ham as imbalanced dataset')
p = plt.xlabel('Is the SMS Spam?')
p = plt.ylabel('Count')
                  


# In[11]:


only_spam = dataset[dataset["label"] == 1]
only_spam


# In[12]:


dataset.shape


# In[13]:


count = int((dataset.shape[0] - only_spam.shape[0]) / only_spam.shape[0])
count


# In[14]:


for i in range(0, count-1):
    dataset = pd.concat([dataset, only_spam])
dataset


# In[15]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(8,8))
p = sns.countplot(x = "label" , data = dataset)
p = plt.title('Countplot for Spam vs Ham as imbalanced dataset')
p = plt.xlabel('Is the SMS Spam?')
p = plt.ylabel('Count')
                  


# In[18]:


dataset['word_count'] = dataset['message'].apply(lambda x: len(x.split()))
dataset


# In[19]:


plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
g = sns.histplot(dataset[dataset["label"] == 0].word_count, kde = True)
p = plt.title('Distribution of word_count for Ham SMS')

plt.subplot(1,2,2)
g = sns.histplot(dataset[dataset["label"] == 1].word_count, color = "red", kde = True)
p = plt.title('Distribution of word_count for Spam SMS')

plt.tight_layout()
plt.show()


# In[24]:


def currency(data):
    currency_symbols = ['$', '€', '£', '¥', '₹']
    for i in currency_symbols:
        if i in data:
            return 1
    return 0
dataset["contains_currency_symbol"] = dataset["message"].apply(currency)
dataset


# In[25]:


plt.figure(figsize=(8,8))
p = sns.countplot(x = "contains_currency_symbol" , data = dataset, hue = 'label')
p = plt.title('Countplot for Containing Numbers')
p = plt.xlabel('Does SMS contains any Number')
p = plt.ylabel('Count')
p = plt.legend(labels = ["Ham", "Spam"], loc = 9)                  


# In[29]:


import nltk
import re 
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer

# Initialize the WordNetLemmatizer
wnl = WordNetLemmatizer()

# Define the list of stopwords
stop_words = set(stopwords.words('english'))

corpus = []

# Preprocess each SMS message
for sms in list(dataset.message):
    # Remove non-alphanumeric characters and convert to lowercase
    message = re.sub('[^a-zA-Z]', ' ', sms)
    message = message.lower()
    # Tokenize the message
    words = message.split()
    # Lemmatize each word and remove stopwords
    filtered_words = [word for word in words if word not in set(stopwords.words('english'))]
    lemm_words = [wnl.lemmatize(word) for word in filtered_words]
    # Join the words back into a string
    message = ' '.join(lemm_words)
    corpus.append(message)


# In[28]:


corpus


# In[35]:


from sklearn.feature_extraction.text import TfidfVectorizer

# Create an instance of TfidfVectorizer
tfidf = TfidfVectorizer(max_features=500)

# Transform the corpus into TF-IDF vectors
vectors = tfidf.fit_transform(corpus).toarray()

# Get the feature names
feature_names = tfidf.get_feature_names_out()

# Create a DataFrame for the TF-IDF vectors with feature names as columns
x = pd.DataFrame(vectors, columns=feature_names)

# Get the labels
y = dataset['label']


# In[36]:


x


# In[37]:


y


# In[39]:


from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train


# In[41]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

mnb = MultinomialNB()

# Perform cross-validation
cv = cross_val_score(mnb, x, y, scoring='f1', cv=10)

# Print the mean and standard deviation of F1 scores
print(round(cv.mean(), 3))
print(round(cv.std(), 3))


# In[42]:


mnb.fit(x_train,y_train)
y_pred = mnb.predict(x_test)
y_pred


# In[43]:


y_test


# In[44]:


print(classification_report(y_test,y_pred))


# In[45]:


cm = confusion_matrix(y_test,y_pred)
cm


# In[46]:


# Assuming you have the confusion matrix stored in a variable named 'cm'
# Replace 'cm' with the actual variable name of your confusion matrix

plt.figure(figsize=(8, 8))
axis_labels = ['ham', 'spam']

# Create heatmap
g = sns.heatmap(data=cm, annot=True, cmap='Blues', fmt='g', cbar_kws={"shrink": 0.5}, xticklabels=axis_labels, yticklabels=axis_labels)

# Set labels and title
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Confusion Matrix')

plt.show()


# In[70]:


def predict_spam(sms):
    # Preprocess the message
    message = re.sub('[^a-zA-Z]', ' ', sms)  # Remove non-alphabetic characters
    message = message.lower()  # Convert to lowercase
    message = message.split()  # Tokenize the message
    filtered_words = [word for word in message if word not in set(stopwords.words('english'))] 
   
    # Lemmatize the message
    lemm_words = [wnl.lemmatize(word) for word in filtered_words]

    # Join the words back into a string
    message = ' '.join(lemm_words)
    temp = tfidf.transform([message]).toarray()
    return classifier.predict(temp)  # Return the predicted label

# Assuming you have defined the sample_message, classifier, tfidf_vectorizer, and other required variables
sample_message = 'IMPORTANT - You could be entitled up to $160 in compensation from mis-sold PPI on a credit card or loan'

# Print the prediction
if predict_spam(sample_message):
    print("The message is predicted to be spam.")
else:
    print("The message is predicted to be ham (not spam).")


# In[ ]:




