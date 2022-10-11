from flask import Flask, render_template, request, redirect, url_for
import re
from nltk.stem import WordNetLemmatizer
import pickle
import snscrape.modules.twitter as sntwitter
import pandas as pd

import regex
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes
from sklearn.metrics import roc_auc_score
from transformers import BertTokenizer
from Emoji_file import EMO_UNICODE
wo = WordNetLemmatizer()
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from flask import flash
import nltk
from nltk.corpus import stopwords
#from transformers import pipeline
#from gensim.parsing.preprocessing import  remove_stopwords


app = Flask(__name__)
app.secret_key = "super secret key"
stopset = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(use_idf=True, lowercase=True,
                            strip_accents='ascii', stop_words=stopset)

#classifer = pipeline('sentiment-analysis')
# function to preprocess input tweets
def preprocess(data):
    #preprocess
    a = re.sub('[^a-zA-Z]',' ',data)
    a = a.lower()
    a = a.split()
    a = [wo.lemmatize(word) for word in a ]
    a = ' '.join(a)
    a = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",a).split())
    
    

    return a

UNICODE_EMO = {v: k for k, v in EMO_UNICODE.items()} 
#tfidf_vectorizer = pickle.load(open('vectorizer.pkl','rb'))
#model =  pickle.load(open('Multinomial.pkl','rb'))

@app.route('/')
def home():
     return render_template('index_new.html')

# function to send username to twitter scrapper
@app.route('/submit',methods=['POST'])
def submit():
    username = request.form['username']
    return redirect(url_for('scraper',user = username))
   
# function to scrap the tweets from twitter
@app.route('/scraper/<string:user>')
def scraper(user):
    # Setting variables to be used below
    maxTweets = 100
    # Creating list to append tweet data to
    tweets_list1 = []
    # Using TwitterSearchScraper to scrape data and append tweets to list
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper('from:%s'%(user)).get_items()):
        if i>maxTweets:
            break
        tweets_list1.append([tweet.date, tweet.id, tweet.content, tweet.user.username])
    
    # import tkinter
    # from tkinter import messagebox
    # root = tkinter.Tk()
    # root.withdraw()

    #import win32api

    df = pd.DataFrame(tweets_list1)
    if df.empty:

        flash("Enter Valid Username","warning")
            
        #messagebox.showinfo("alert","Enter valid username")
       # win32api.MessageBox(0,"Enter valid username","alert")
        return render_template('index_new.html', pred = "")
    print(df.head())
    
    #print(df.2[0])
    df.rename(columns = {2:'Tweet'}, inplace = True)
    global df_first_5
    df_first_5 = df[['Tweet']].head(5)
    '''if isinstance(df_first_5, pd.DataFrame ):
        print("True")
    else:
        print("False")'''
    #print(df_first_5)
    #for row in df_first_5:
    '''html = df_first_5.to_html(classes='table table-striped')

    #write html to file
    text_file = open("index.html", "w")
    text_file.write(html)
    text_file.close()'''
    #for i in range(5):
        #msg = msg.append('*****'.join(tweets_list1[i][2]))
    #print(msg)    
    return render_template('index_new.html', tweetmsg = tweets_list1[0][2])
    # '****'.join(tweets_list1[i][2])



"""
@app.route('/predict', methods= ['POST'])

def home():
    if request.method == 'GET':
        print("hiii")
        #msg = request.form.get("pred", False) 
        #request.form['pred']
        msg = request.form['mood_pred']
        a = preprocess(msg)

        # example_counts = vectorizer.transform( [a] )
        # prediction = mnb.predict( example_counts )
        # prediction[0]

        result = model.predict(tfidf_vectorizer.transform([a]))[0]
        print(result)
    return render_template('index2.html',pred = result)

"""
@app.route('/predict',methods= ['POST'])
def predict():
    msg = request.form['mood_pred']
    
    
    
    def convert_emojis(msg):
        for emot in UNICODE_EMO:
            msg = re.sub(r'('+emot+')', "_".join(UNICODE_EMO[emot].replace(",","").replace(":","").split()), msg)
        return msg

    
    msg = convert_emojis(msg) 
    #TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
    #BERT_PATH = "../bert_base_uncased"
    #tokenizer = BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
    
    
    #tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    #token_new = AutoTokenizer.save_pretrained(r'C:\Users\yogita\OneDrive\Desktop\depression_yogita\saved_models')
    #tokenizer = AutoTokenizer.from_pretrained(r'C:\Users\yogita\OneDrive\Desktop\depression_yogita\saved_models')
    #model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment') 


    saved_directory = "saved"
    '''tokenizer.save_pretrained(saved_directory)
    model.save_pretrained(saved_directory)'''


    tokenizer = AutoTokenizer.from_pretrained(saved_directory)
    model = AutoModelForSequenceClassification.from_pretrained(saved_directory)
    def sentiment_score(message):
        tokens = tokenizer.encode(message, return_tensors='pt')
        result = model(tokens)
        return int(torch.argmax(result.logits))+1



    print(msg) 
    a = preprocess(msg)
    print(a)
    #result = classifer(a)
    result = sentiment_score(a)
    #result = model.predict(tfidf_vectorizer.transform([a]))[0]
    #print(result1)
    print(result)
    #new = pd.DataFrame(result)

    #print(new)
    if( result<=3):
        return render_template('remedies.html',pred = "You are depressed")
    elif(result >3):
        return render_template('gif.html')




@app.route('/home2')
def home2():
    return render_template('index_new.html')

@app.route('/map')
def map():
    return render_template('own_map.html')


   



if __name__ == '__main__': 
   
    app.run(debug=True)



    