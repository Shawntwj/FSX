# Google news scraper libraries
import pandas as pd
from GoogleNews import GoogleNews # pip install GoogleNews
from newspaper import Article # pip install newspaper3k
from newspaper import Config
import time

# Machine learning model libraries
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost 
from sklearn.metrics  import classification_report
from sklearn import metrics
import time
import pickle

# Data cleaning libraries
import re 
from sklearn.base import BaseEstimator
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
nltk.download('omw-1.4')

# Ftraineature engineering pipeline libraries
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from mlxtend.feature_selection import ColumnSelector



user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.82 Safari/537.36'
config = Config()
config.browser_user_agent = user_agent

# ---scraper code---

def getGoogleNewsLinks(terms, start_date, end_date):

    parsed_news = []
    googlenews=GoogleNews(start = start_date, end = end_date) #month/day/year
    googlenews.search(terms)
    
    for i in range(2, 3):
        googlenews.getpage(i)
        result=googlenews.result()
        df=pd.DataFrame(result)
    
    for ind in df.index:
        link = df['link'][ind]
        parsed_news.append([terms, link])
    return(parsed_news)
    
def getGoogleNews(terms, start, end):
    parsed_news = getGoogleNewsLinks(terms, start, end)
    news_df = getArticleSummary(terms, parsed_news)
    return(news_df)

def getArticleSummary(terms, parsed_news):
    
    dictionary = {
         'locust':'idiosyncratic',
         'covid19':'pandemic',
         'avalanche':'geophysical event',
         'contamination':'Man-Made Disaster',
         'cyberattack':'cyberattack',
         'cyclone':'Acute climatological event (cyclone)',
         'dengue':'idiosyncratic',
         'drought':'Acute climatological event (droughts)',
         'earthquake':'geophysical event',
         'ebola':'pandemic',
         'economic crisis':'Economic Crisis',
         'floods':'Acute climatological event (flood)',
         'heat stress':'Acute climatological event (heat stress)',
         'Influenza':'pandemic',
         'limnic eruption':'Acute climatological event (heat stress)',
         'nuclear':'Man-Made Disaster',
         'oil spills':'Man-Made Disaster',
         'pandemic':'pandemic',
         'SarS':'pandemic',
         'sinkhole':'geophysical event',
         'terror':'Terrorism',
         'tradedispute':'Trade Dispute',
         'tsunami':'geophysical event',
         'Unsafe':'idiosyncratic',
         'volcanic eruption':'Acute climatological event (heat stress)',
         'war':'Military Conflicts'
        }

    list=[]
    for ind in parsed_news:
        dicti={}
        article = Article(ind[1],config=config)
        try:
             
            article.download()
            article.parse()
            article.nlp()
            
            if article.publish_date == None:
                dicti['date'] = date
            else:
                dicti['date']= article.publish_date
                date = article.publish_date
            dicti['news title']=article.title
            dicti['news source(url)']=ind[1]
            dicti['content summary']= article.summary
            keywords = article.keywords
            dicti['keywords'] = ", ".join(keywords)
            dicti['class_name'] = terms
            dicti['new_class_name'] = dictionary[terms]
            dicti['full Article ']=article.text

            list.append(dicti)
        except:
            pass
    news_df=pd.DataFrame(list)
    return(news_df)

# edit the fields here
def scraper():
    
    search = [
     'locust',
    #  'covid19',
    #  'avalanche',
    #  'contamination',
    #  'cyberattack',
    #  'cyclone',
    #  'dengue',
    #  'drought',
    #  'earthquake',
    #  'ebola',
    #  'economic crisis',
    #  'floods',
    #  'heat stress',
    #  'Influenza',
    #  'limnic eruption',
    #  'nuclear',
    #  'oil spills',
    #  'pandemic',
    #  'SarS',
    #  'sinkhole',
    #  'terror',
    #  'tradedispute',
    #  'tsunami',
    #  'Unsafe',
    #  'volcanic eruption',
    #  'war'
    ]
    
    # prevent error429 by working with batches (per month)
    # time_periodDB = {1: ['01/01/2021', '01/31/2021'],
    #               2: ['02/01/2021', '02/28/2021'],
    #               3: ['03/01/2021', '03/31/2021'],
    #               4: ['04/01/2021', '04/30/2021'],
    #               5: ['05/01/2021', '05/31/2021'],
    #               6: ['06/01/2021', '06/30/2021'],
    #               7: ['07/01/2021', '07/31/2021'],
    #               8: ['08/01/2021', '08/31/2021'],
    #               9: ['09/01/2021', '09/30/2021'],
    #               10: ['10/01/2021', '10/30/2021'],
    #               11: ['11/01/2021', '11/30/2021'],
    #               12: ['12/01/2021', '12/30/2021']}

    time_period = {1: ['01/01/2021', '01/31/2021'],}
                # 2: ['02/01/2021', '02/28/2021'],
                # 3: ['03/01/2021', '03/31/2021'],}
    
    
    for toSearch in search:
        google_news_df = []
        for month_index in time_period: #change the range if from whenever it stopped at
            start_date, end_date = time_period[month_index][0], time_period[month_index][1]
            google_news_by_time_period = getGoogleNews(toSearch, start_date, end_date) # returns df
            google_news_df.append(google_news_by_time_period)
            print(f'done: month {month_index}')
        print('done',toSearch)

        google_news = pd.concat(google_news_df, ignore_index=True)
        google_news.to_csv(f"Google News Data/{toSearch}_googleNews.csv")
    print("whole scraping done")
        
    
# ---modelling code---
def trainTestSplit():
    cwd = os.path.abspath('Google News Data') 
    files = os.listdir(cwd) 
    df = pd.DataFrame()
    for file in files:
        if file.endswith('.csv'):
            df = df.append(pd.read_csv(cwd+"/"+file), ignore_index=True) 
            

    df = df.dropna()
    df.columns
    X = df.drop(['Unnamed: 0', 'date', 'news title', 'news source(url)',
        'keywords', 'class_name', 'new_class_name'], axis=1)

    y = df['new_class_name']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    return X_train, X_test, y_train, y_test


# define the class FeatureEngineering
# This will be our custom transformer that will create 3 new binary columns
# custom transformer must have methods fit and transform
class FeatureEngineering(BaseEstimator):

    def __init__(self):
        pass

    def fit(self, documents, y=None):
        return self
    
        
    def transform(self, x_dataset):
        """
        Function: split text into words and return the root form of the words
        Args:
          text(str): the article
        Return:
          lem(list of str): a list of the root form of the article words
        """
        def preprocess(text):

            # Normalize text
            text = re.sub(r"[^a-zA-Z]", " ", str(text).lower())

            # Tokenize text
            token = word_tokenize(text)

            # Remove stop words
            stop = stopwords.words("english")
            new_stop_words_list = ['said', 'us', 'also', 'mr']
            stop.extend(new_stop_words_list)
            words = [t for t in token if t not in stop]

            # Lemmatization
            lem = [WordNetLemmatizer().lemmatize(w) for w in words]

            return lem
        
        x_dataset.head()

        x_dataset["Preprocessed_Text"] = x_dataset['content summary'].apply(lambda x: preprocess(x))
        x_dataset['Preprocessed_Text2'] = x_dataset['Preprocessed_Text'].apply(' '.join)
        print("text cleaning done")
        return x_dataset

def featureEngineeringPipeline():
    preprocessor = Pipeline(steps=[('feature engineering', FeatureEngineering()), 
                                ('col_selector', ColumnSelector(cols=('Preprocessed_Text2'),drop_axis=True)),
                                ('tfidf',TfidfVectorizer()),
                                ])

    X_train, X_test, y_train, y_test = trainTestSplit()
    train_features = preprocessor.fit(X_train)
    test_features = preprocessor.fit(X_test)

    train_features = train_features.transform(X_train)
    test_features = test_features.transform(X_test)
    return train_features, y_train, test_features, y_test
    print("feature engineering done")



def fit_eval_model(cls_name,model, train_features, y_train, test_features, y_test):
    
    """
    Function: train and evaluate a machine learning classifier.
    Args:
      model: machine learning classifier
      train_features: train data extracted features
      y_train: train data lables
      test_features: train data extracted features
      y_test: train data lables
    Return:
      results(dictionary): a dictionary of the model training time and classification report
    """
    results ={}
    
    # Start time
    start = time.time()
    # Train the model
    model.fit(train_features, y_train)
    # End time
    end = time.time()
    # Calculate the training time
    results['train_time'] = end - start
    
    # Test the model
    train_predicted = model.predict(train_features)
    test_predicted = model.predict(test_features)
    
    # Save the model
    filename = cls_name + '.sav'
    pickle.dump(model, open(filename, 'wb'))
    
    # Classification report
    results['classification_report'] = classification_report(y_test, test_predicted)
        
    return results

def modelling():
    train_features, y_train, test_features, y_test = featureEngineeringPipeline()
    # user can add more models should they want it in the future
    # sv = svm.SVC()
    # ab = AdaBoostClassifier(random_state = 1)
    # gb = GradientBoostingClassifier(random_state = 1)
    xgb = xgboost.XGBClassifier(random_state = 1)
    # tree = DecisionTreeClassifier()
    # nb = MultinomialNB()

    # Fit and evaluate models
    results = {}
    # for cls in [sv, ab, gb, xgb, tree, nb]:
    for cls in [xgb]:
        cls_name = cls.__class__.__name__
        results[cls_name] = {}
        results[cls_name] = fit_eval_model(cls_name,cls, train_features, y_train, test_features, y_test)

    for res in results:
        print (res)
        print()
        for i in results[res]:
            print (i, ':')
            print(results[res][i])
            print()
        print ('-----')
        print()
    print("modelling done")
        
        
scraper()
modelling()