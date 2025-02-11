import pandas as pd 
import re  ## regular expression
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import seaborn as sns
import matplotlib.pyplot as plt
nltk.download('stopwords')

##loading data 
messages = pd.read_csv("spam.csv", sep=',', names = ['label', 'message'], encoding = 'latin1')

ps = PorterStemmer()
corpus = []

# Preprocess messages
for i in range(len(messages)):
    text = messages['message'].iloc[i]  # Use .iloc[i] to avoid FutureWarning
    
    if pd.isna(text):  # Handle missing values
        text = ""

    text = re.sub('[^a-zA-Z]', ' ', text)  # Remove non-alphabet characters
    text = text.lower()  # Convert to lowercase
    text = text.split()  # Tokenization

    # Remove stopwords and apply stemming
    filtered_text = [ps.stem(word) for word in text if word not in stopwords.words('english')]
    cleaned_text = ' '.join(filtered_text)

    corpus.append(cleaned_text)

# Check the first few cleaned messages
print(corpus[:5])

##creating  the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500) ## max_feature = number of columns
X = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(messages['label'])   ## one-hot encoding 
y = y.iloc[:,1].values   ## drops the column 0 , [: selects all row of , 1 column 2nd as it stars with zero]

## train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size= 0.2 , random_state = 4)

##train model using Naive bayes classifier
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train,y_train)

##prediction
y_pred = spam_detect_model.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report
print(classification_report(y_test, y_pred))

##confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True)
plt.show()


## accuracy 
from sklearn.metrics import accuracy_score
print("accuracy: " , accuracy_score(y_test, y_pred))



