from flask import Flask, render_template, request, redirect, url_for, session
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import math, random
from datetime import datetime
import datetime as dt
import yfinance as yf
import tweepy
import preprocessor as p
import re
from sklearn.linear_model import LinearRegression
from textblob import TextBlob
import constants as ct
from Tweet import Tweet
#import nltk
import smtplib
import pyrebase
import chart_studio.plotly
import chart_studio.tools as tls
import cufflinks as cf
import chart_studio
import pandas as pd
import mplfinance as fplt
import plotly.graph_objects as go
#nltk.download('punkt')
  
  
app = Flask(__name__)
  
  
app.secret_key = 'your secret key'
  
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'nav@sql1'
app.config['MYSQL_DB'] = 'projectlogin'

 
mysql = MySQL(app)


firebaseConfig={
    'apiKey': "AIzaSyASagMF4GLxT0s-wTJf72KlPe2fkgZwMDk",
    'authDomain': "voyagecapital-f03c3.firebaseapp.com",
    'projectId': "voyagecapital-f03c3",
    'storageBucket': "voyagecapital-f03c3.appspot.com",
    'messagingSenderId': "214753414293",
    'appId': "1:214753414293:web:2c626e3b4dfff4f784df9c",
    'measurementId': "G-C7J3L4BKR9",
    'databaseURL': "xxxxx"
}
firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()

import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

@app.after_request
def add_header(response):
    response.headers['Pragma'] = 'no-cache'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Expires'] = '0'
    response.cache_control.max_age = 0
    return response
 
@app.route('/')
def index():
   return render_template('index.html', font_url="https://fonts.googleapis.com/css2?family=Poppins&display=swap")

@app.route('/aboutus')
def aboutus():
   return render_template('aboutus.html')
  
@app.route('/faq')
def faq():
   return render_template('faq.html')
  
@app.route('/contactus', methods = ["GET","POST"])
def contactus():
    msg=''
    if request.method=='POST':
        print('post')
        fullname = request.form.get("fullname")
        email = request.form.get('email')
        subject = request.form.get('subject')
        message = request.form.get('message')
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('INSERT INTO messages VALUES ( NULL , (SELECT id FROM userdetails WHERE email= %s LIMIT 1) , %s , %s , % s)', (email,fullname,email,message,))
        mysql.connection.commit()
        message=email + "\n" +message 
        def send_email(email, subject, message):
            try:
                server = smtplib.SMTP('smtp.gmail.com:587')
                server.ehlo()
                server.starttls()
                #server.login(email, password)
                server.login("voyagecapital04@gmail.com", "voyage@456")
                message = 'Subject: {}\n\n{}'.format(subject, message)
                server.sendmail("voyagecapital04@gmail.com", "voyagecapital04@gmail.com", message)
                server.quit()
                print("Success: Email sent!")
                msg="Your query has been successfully submitted. We will contact you shortly."
            except ValueError:
                print(ValueError)
                print("Email failed to send.")
                msg="Contact process failed"

        send_email(email, subject, message)
    return render_template('contactus.html', msg=msg)

@app.route('/result')
def result():
   return render_template('result.html')


@app.route('/filtertable')
def filtertable():
   return render_template('filtertable.html')
 
@app.route('/indexfaq')
def indexfaq():
   return render_template('indexfaq.html')

@app.route('/dashboard')
def dashboard():
   return render_template('dashboard.html')


@app.route('/indexaboutus')
def indexaboutus():
   return render_template('indexaboutus.html')

@app.route('/currencyconvert',methods = ["GET","POST"])
def currencyconvert():
    answer = 0
    if request.method == 'POST':
        base_currency = request.form.get('base_currency')
        target_currency = request.form.get('target_currency')
        amount = float(request.form.get('amount'))
        key = 'WXR4A3OET4JMFIXR'
        import requests
        req = 'https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency='+base_currency+'&to_currency='+target_currency+'&apikey='+key
        response = requests.get(req)
        answer = float(response.json()['Realtime Currency Exchange Rate']['5. Exchange Rate'])*amount
        
    else:
        answer = 0
    return render_template('currencyconvert.html', converted=answer)

global symbol123
@app.route('/tables', methods=("POST", "GET"))
def table():
    print("Done")
    global symbol123
    df=yf.Ticker(symbol123).financials
    df1=yf.Ticker(symbol123).recommendations
    df2=yf.Ticker(symbol123).quarterly_balance_sheet
    df3=yf.Ticker(symbol123).cashflow
    df4=yf.Ticker(symbol123).earnings
    return render_template('table.html',  tables=[df.to_html(classes='df'),df1.to_html(classes='data'),df2.to_html(classes='data'),df3.to_html(classes='data'),df4.to_html(classes='data')], 
                            titles=['na','Financials','Recommendations','Quarterly Balance Sheet','Cashflow','Earnings'])

global user_id123
@app.route('/enterticker', methods =['GET', 'POST'])
def enterticker():
    if request.method == 'POST' and 'symbol' in request.form:
        symbol = request.form['symbol']
        try:
            global symbol123
            symbol123=symbol
            global user_id123
            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            cursor.execute('SELECT id,symbol FROM history WHERE id=%s AND symbol=%s', (user_id123,symbol))    
            hist_done=cursor.fetchone()
            if(hist_done==None):
                cursor.execute('INSERT INTO history VALUES (%s,%s)', (user_id123,symbol))
                mysql.connection.commit()
        except:
            msg = ("Your session has expired")
            return render_template('login.html', msg = msg)
        def get_historical(quote):
            end = datetime.now()
            start = datetime(end.year-2,end.month,end.day)
            print('call')
            data = yf.download(quote, start=start, end=end)
            #data = pd.read_csv('AAPL.csv', index_col='Date', parse_dates=True)

            print('call end')

            df = pd.DataFrame(data=data)
            df.to_csv(''+quote+'.csv')
            if(df.empty):
                ts = TimeSeries(key='N6A6QT6IBFJOPJ70',output_format='pandas')
                data, meta_data = ts.get_daily_adjusted(symbol='BSE:'+quote, outputsize='full')
        #Format df
        #Last 2 yrs rows => 502, in ascending order => ::-1
                data=data.head(503).iloc[::-1]
                data=data.reset_index()
        #Keep Required cols only
                df=pd.DataFrame()
                df['Date']=data['date']
                df['Open']=data['1. open']
                df['High']=data['2. high']
                df['Low']=data['3. low']
                df['Close']=data['4. close']
                df['Adj Close']=data['5. adjusted close']
                df['Volume']=data['6. volume']
                df.to_csv(''+quote+'.csv',index=False)
            return

        def write_file(data, filename):
            # Convert binary data to proper format and write it on Hard Disk
            with open(filename, 'wb') as file:
                file.write(data)
        def convertToBinaryData(filename):
            # Convert digital data to binary format
            with open(filename, 'rb') as file:
                binaryData = file.read()
            return binaryData
        def ARIMA_ALGO(df):
            uniqueVals = df["Code"].unique()  
            len(uniqueVals)
            df=df.set_index("Code")
            #for daily basis
            def parser(x):
                return datetime.strptime(x, '%Y-%m-%d')
            def arima_model(train, test):
                history = [x for x in train]
                predictions = list()
                for t in range(len(test)):
                    model = ARIMA(history, order=(6,1 ,0))
                    model_fit = model.fit(disp=0)
                    output = model_fit.forecast()
                    yhat = output[0]
                    predictions.append(yhat[0])
                    obs = test[t]
                    history.append(obs)
                return predictions
            for company in uniqueVals[:10]:
                data=(df.loc[company,:]).reset_index()
                data['Price'] = data['Close']
                Quantity_date = data[['Price','Date']]
                Quantity_date.index = Quantity_date['Date'].map(lambda x: parser(x))
                Quantity_date['Price'] = Quantity_date['Price'].map(lambda x: float(x))
                Quantity_date = Quantity_date.fillna(Quantity_date.bfill())
                Quantity_date = Quantity_date.drop(['Date'],axis =1)
                fig = plt.figure(figsize=(7.2,4.8),dpi=65)
                plt.plot(Quantity_date)
                plt.savefig('static/Trends.png')
                plt.close(fig)
                trend_binary=convertToBinaryData("static/Trends.png")
                quantity = Quantity_date.values
                size = int(len(quantity) * 0.80)
                train, test = quantity[0:size], quantity[size:len(quantity)]
                #fit in model
                predictions = arima_model(train, test)
        
                cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
                #plot graph
                fig = plt.figure(figsize=(7.2,4.8),dpi=65)
                plt.plot(test,label='Actual Price')
                plt.plot(predictions,label='Predicted Price')
                plt.legend(loc=4)
                plt.savefig('static/ARIMA.png')
                plt.close(fig)
                arima_binary=convertToBinaryData("static/ARIMA.png")
                print()
                print("##############################################################################")
                arima_pred=predictions[-2]
                print("Tomorrow's",quote," Closing Price Prediction by ARIMA:",arima_pred)
                #rmse calculation
                error_arima = math.sqrt(mean_squared_error(test, predictions))
                print("ARIMA RMSE:",error_arima)
                print("##############################################################################")
                return arima_pred, error_arima, trend_binary, arima_binary
        def LSTM_ALGO(df):
            #Split data into training set and test set
            dataset_train=df.iloc[0:int(0.8*len(df)),:]
            dataset_test=df.iloc[int(0.8*len(df)):,:]
            ############# NOTE #################
            #TO PREDICT STOCK PRICES OF NEXT N DAYS, STORE PREVIOUS N DAYS IN MEMORY WHILE TRAINING
            # HERE N=7
            ###dataset_train=pd.read_csv('Google_Stock_Price_Train.csv')
            training_set=df.iloc[:,4:5].values# 1:2, to store as numpy array else Series obj will be stored
            #select cols using above manner to select as float64 type, view in var explorer

            #Feature Scaling
            from sklearn.preprocessing import MinMaxScaler
            sc=MinMaxScaler(feature_range=(0,1))#Scaled values btween 0,1
            training_set_scaled=sc.fit_transform(training_set)
            #In scaling, fit_transform for training, transform for test
            
            #Creating data stucture with 7 timesteps and 1 output. 
            #7 timesteps meaning storing trends from 7 days before current day to predict 1 next output
            X_train=[]#memory with 7 days from day i
            y_train=[]#day i
            for i in range(7,len(training_set_scaled)):
                X_train.append(training_set_scaled[i-7:i,0])
                y_train.append(training_set_scaled[i,0])
            #Convert list to numpy arrays
            X_train=np.array(X_train)
            y_train=np.array(y_train)
            X_forecast=np.array(X_train[-1,1:])
            X_forecast=np.append(X_forecast,y_train[-1])
            #Reshaping: Adding 3rd dimension
            X_train=np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))#.shape 0=row,1=col
            X_forecast=np.reshape(X_forecast, (1,X_forecast.shape[0],1))
            #For X_train=np.reshape(no. of rows/samples, timesteps, no. of cols/features)
            
            #Building RNN
            from keras.models import Sequential
            from keras.layers import Dense
            from keras.layers import Dropout
            from keras.layers import LSTM
            
            #Initialise RNN
            regressor=Sequential()
            
            #Add first LSTM layer
            regressor.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
            #units=no. of neurons in layer
            #input_shape=(timesteps,no. of cols/features)
            #return_seq=True for sending recc memory. For last layer, retrun_seq=False since end of the line
            regressor.add(Dropout(0.1))
            
            #Add 2nd LSTM layer
            regressor.add(LSTM(units=50,return_sequences=True))
            regressor.add(Dropout(0.1))
            
            #Add 3rd LSTM layer
            regressor.add(LSTM(units=50,return_sequences=True))
            regressor.add(Dropout(0.1))
            
            #Add 4th LSTM layer
            regressor.add(LSTM(units=50))
            regressor.add(Dropout(0.1))
            
            #Add o/p layer
            regressor.add(Dense(units=1))
            
            #Compile
            regressor.compile(optimizer='adam',loss='mean_squared_error')
            
            #Training
            regressor.fit(X_train,y_train,epochs=25,batch_size=32 )
            #For lstm, batch_size=power of 2
            
            #Testing
            ###dataset_test=pd.read_csv('Google_Stock_Price_Test.csv')
            real_stock_price=dataset_test.iloc[:,4:5].values
            
            #To predict, we need stock prices of 7 days before the test set
            #So combine train and test set to get the entire data set
            dataset_total=pd.concat((dataset_train['Close'],dataset_test['Close']),axis=0) 
            testing_set=dataset_total[ len(dataset_total) -len(dataset_test) -7: ].values
            testing_set=testing_set.reshape(-1,1)
            #-1=till last row, (-1,1)=>(80,1). otherwise only (80,0)
            
            #Feature scaling
            testing_set=sc.transform(testing_set)
            
            #Create data structure
            X_test=[]
            for i in range(7,len(testing_set)):
                X_test.append(testing_set[i-7:i,0])
                #Convert list to numpy arrays
            X_test=np.array(X_test)
            
            #Reshaping: Adding 3rd dimension
            X_test=np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
            
            #Testing Prediction
            predicted_stock_price=regressor.predict(X_test)
            
            #Getting original prices back from scaled values
            predicted_stock_price=sc.inverse_transform(predicted_stock_price)
            fig = plt.figure(figsize=(7.2,4.8),dpi=65)
            plt.plot(real_stock_price,label='Actual Price')  
            plt.plot(predicted_stock_price,label='Predicted Price')
            
            plt.legend(loc=4)
            plt.savefig('static/LSTM.png')
            plt.close(fig)
            
            lstm_binary=convertToBinaryData("static/LSTM.png")
            error_lstm = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
            
            
            #Forecasting Prediction
            forecasted_stock_price=regressor.predict(X_forecast)
            
            #Getting original prices back from scaled values
            forecasted_stock_price=sc.inverse_transform(forecasted_stock_price)
            
            lstm_pred=forecasted_stock_price[0,0]
            print()
            print("##############################################################################")
            print("Tomorrow's ",quote," Closing Price Prediction by LSTM: ",lstm_pred)
            print("LSTM RMSE:",error_lstm)
            print("##############################################################################")
            return lstm_pred,error_lstm, lstm_binary
        def LIN_REG_ALGO(df):
            #No of days to be forcasted in future
            forecast_out = int(7)
            #Price after n days
            df['Close after n days'] = df['Close'].shift(-forecast_out)
            #New df with only relevant data
            df_new=df[['Close','Close after n days']]

            #Structure data for train, test & forecast
            #lables of known data, discard last 35 rows
            y =np.array(df_new.iloc[:-forecast_out,-1])
            y=np.reshape(y, (-1,1))
            #all cols of known data except lables, discard last 35 rows
            X=np.array(df_new.iloc[:-forecast_out,0:-1])
            #Unknown, X to be forecasted
            X_to_be_forecasted=np.array(df_new.iloc[-forecast_out:,0:-1])
            
            #Traning, testing to plot graphs, check accuracy
            X_train=X[0:int(0.8*len(df)),:]
            X_test=X[int(0.8*len(df)):,:]
            y_train=y[0:int(0.8*len(df)),:]
            y_test=y[int(0.8*len(df)):,:]
            
            # Feature Scaling===Normalization
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
            
            X_to_be_forecasted=sc.transform(X_to_be_forecasted)
            
            #Training
            clf = LinearRegression(n_jobs=-1)
            clf.fit(X_train, y_train)
            
            #Testing
            y_test_pred=clf.predict(X_test)
            y_test_pred=y_test_pred*(1.04)
            import matplotlib.pyplot as plt2
            fig = plt2.figure(figsize=(7.2,4.8),dpi=65)
            plt2.plot(y_test,label='Actual Price' )
            plt2.plot(y_test_pred,label='Predicted Price')
            
            plt2.legend(loc=4)
            plt2.savefig('static/LR.png')
            plt2.close(fig)
            lr_binary=convertToBinaryData("static/LR.png")
            error_lr = math.sqrt(mean_squared_error(y_test, y_test_pred))
            
            
            #Forecasting
            
            forecast_set = clf.predict(X_to_be_forecasted)
            forecast_set=forecast_set*(1.04)
            mean=forecast_set.mean()
            lr_pred=forecast_set[0,0]
            print(forecast_set)
            print()
            print("##############################################################################")
            print("Tomorrow's ",quote," Closing Price Prediction by Linear Regression: ",lr_pred)
            print("Linear Regression RMSE:",error_lr)
            print("##############################################################################")
            return df, lr_pred, forecast_set, mean, error_lr,lr_binary
                
        #**************** SENTIMENT ANALYSIS **************************
        def retrieving_tweets_polarity(symbol1):
            try:
                stock_ticker_map = pd.read_csv('Yahoo-Finance-Ticker-Symbols.csv')
                stock_full_form = stock_ticker_map[stock_ticker_map['Ticker']==symbol1]
                symbol = stock_full_form['Name'].to_list()[0][0:12]

            except:
                symbol=symbol1

            auth = tweepy.OAuthHandler(ct.consumer_key, ct.consumer_secret)
            auth.set_access_token(ct.access_token, ct.access_token_secret)
            user = tweepy.API(auth)
            
            tweets = tweepy.Cursor(user.search, q=symbol, tweet_mode='extended', lang='en',exclude_replies=True).items(ct.num_of_tweets)
            
            tweet_list = [] #List of tweets alongside polarity
            global_polarity = 0 #Polarity of all tweets === Sum of polarities of individual tweets
            tw_list=[] #List of tweets only => to be displayed on web page
            #Count Positive, Negative to plot pie chart
            pos=0 #Num of pos tweets
            neg=1 #Num of negative tweets
            c=0
            for tweet in tweets:
                count=20 #Num of tweets to be displayed on web page
                #Convert to Textblob format for assigning polarity
                tw2 = tweet.full_text
                tw = tweet.full_text
                #Clean
                tw=p.clean(tw)
                #print("-------------------------------CLEANED TWEET-----------------------------")
                #print(tw)
                #Replace &amp; by &
                tw=re.sub('&amp;','&',tw)
                #Remove :
                tw=re.sub(':','',tw)
                #print("-------------------------------TWEET AFTER REGEX MATCHING-----------------------------")
                #print(tw)
                #Remove Emojis and Hindi Characters
                tw=tw.encode('ascii', 'ignore').decode('ascii')
                c+=1
                #print("-------------------------------TWEET AFTER REMOVING NON ASCII CHARS-----------------------------")
                #print(tw)
                blob = TextBlob(tw)
                polarity = 0 #Polarity of single individual tweet
                for sentence in blob.sentences:

                    polarity += sentence.sentiment.polarity
                    if polarity>0:
                        pos=pos+1
                    if polarity<0:
                        neg=neg+1
                    
                    global_polarity += sentence.sentiment.polarity
                if count > 0:
                    tw_list.append(tw2)
                    
                tweet_list.append(Tweet(tw, polarity))
                count=count-1
            if len(tweet_list) != 0:
                global_polarity = global_polarity / len(tweet_list)
            else:
                global_polarity = global_polarity
            neutral=ct.num_of_tweets-pos-neg
            if neutral<0:
                neg=neg+neutral
                neutral=20
            print()
            print(c)
            if neg<0:
                pos=pos+2*neg
                neg=neg*-1
            print("##############################################################################")
            print("Positive Tweets :",pos,"Negative Tweets :",neg,"Neutral Tweets :",neutral)
            print("##############################################################################")
            labels=['Positive','Negative','Neutral']
            sizes = [pos,neg,neutral]
            explode = (0, 0, 0)
            fig = plt.figure(figsize=(7.2,4.8),dpi=65)
            fig1, ax1 = plt.subplots(figsize=(7.2,4.8),dpi=65)
            ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=90)
            # Equal aspect ratio ensures that pie is drawn as a circle
            ax1.axis('equal')  
            plt.tight_layout()
            plt.savefig('static/SA.png')
            plt.close(fig)
            tweet_binary=convertToBinaryData("static/SA.png")
            #plt.show()
            if global_polarity>0:
                print()
                print("##############################################################################")
                print("Tweets Polarity: Overall Positive")
                print("##############################################################################")
                tw_pol="Overall Positive"
            else:
                print()
                print("##############################################################################")
                print("Tweets Polarity: Overall Negative")
                print("##############################################################################")
                tw_pol="Overall Negative"
            return global_polarity,tw_list,tw_pol,pos,neg,neutral,tweet_binary


        def recommending(df, global_polarity,today_stock,mean):
            if today_stock.iloc[-1]['Close'] < mean:
                if global_polarity > 0:
                    idea="RISE"
                    decision="BUY"
                    print()
                    print("##############################################################################")
                    print("According to the ML Predictions and Sentiment Analysis of Tweets, a",idea,"in",quote,"stock is expected => ",decision)
                elif global_polarity <= 0:
                    idea="FALL"
                    decision="SELL"
                    print()
                    print("##############################################################################")
                    print("According to the ML Predictions and Sentiment Analysis of Tweets, a",idea,"in",quote,"stock is expected => ",decision)
            else:
                idea="FALL"
                decision="SELL"
                print()
                print("##############################################################################")
                print("According to the ML Predictions and Sentiment Analysis of Tweets, a",idea,"in",quote,"stock is expected => ",decision)
            return idea, decision

        #**************GET DATA ***************************************
        quote=symbol
        #Try-except to check if valid stock symbol
        try:
            get_historical(quote)
        except:
            msg="Invalid Stock Symbol"
            return render_template('enterticker.html',msg=msg,not_found=True)
        else:
        
            #************** PREPROCESSUNG ***********************
            df = pd.read_csv(''+quote+'.csv')
            print("##############################################################################")
            print("Today's",quote,"Stock Data: ")
            today_stock=df.iloc[-1:]
            print(today_stock)
            print("##############################################################################")
            df = df.dropna()
            code_list=[]
            for i in range(0,len(df)):
                code_list.append(quote)
            df2=pd.DataFrame(code_list,columns=['Code'])
            df2 = pd.concat([df2, df], axis=1)
            df=df2
            
            quote_ratios = yf.Ticker(quote).info
            #print(quote_ratios)

            '''arima_pred, error_arima, trend_binary, arima_binary=ARIMA_ALGO(df)
            lstm_pred, error_lstm, lstm_binary=LSTM_ALGO(df)
            df, lr_pred, forecast_set,mean,error_lr,lr_binary=LIN_REG_ALGO(df)
            polarity,tw_list,tw_pol,pos,neg,neutral,tweet_binary = retrieving_tweets_polarity(quote)
            
            idea, decision=recommending(df, polarity,today_stock,mean)
            print()
            print("Forecasted Prices for Next 7 days:")
            print(forecast_set)
            print(forecast_set.shape)
            today_stock=today_stock.round(2)'''
            #return render_template('index.html')

            end=datetime.now()
            start = datetime(end.year,end.month,end.day)
            start=start.strftime('%Y-%m-%d')
            
            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            cursor.execute('SELECT * FROM cache_pred WHERE quote = % s and date=%s', (quote,start,))
            cache_pr = cursor.fetchone()
            cursor.execute('SELECT * FROM cache_twitter WHERE quote = % s and date=%s', (quote,start,))
            cache_tw = cursor.fetchone()
            cursor.execute('SELECT * FROM images_pred WHERE quote = % s and date=%s', (quote,start,))
            cache_ip = cursor.fetchone()
            print("======================================================")
            tw_list=[]
            forecast_set=[]
            if cache_pr!=None :
                print("In cache")
                arima_pred=cache_pr['arima_pred']
                lstm_pred=cache_pr['lstm_pred']
                lr_pred=cache_pr['lr_pred']
                error_lr=cache_pr['error_lr']
                error_lstm=cache_pr['error_lstm']
                error_arima=cache_pr['error_arima']
                for i in range(0,7):
                    forecast_set.append([cache_pr['forecast'+str(i+1)]])
                for i in range(0,12):
                    tw_list.append(cache_tw['tweet'+str(i+1)])
                idea=cache_tw['idea']
                decision=cache_tw['decision']
                tw_pol=cache_tw['tw_pol']
                write_file(cache_ip['trends'], 'static/Trends.png')
                write_file(cache_ip['arima'], 'static/ARIMA.png')
                write_file(cache_ip['lstm'], 'static/LSTM.png')
                write_file(cache_ip['lr'], 'static/LR.png')
                write_file(cache_ip['tweet'], 'static/SA.png')
                today_stock=today_stock.round(2)

            else:
                print("in else")
                arima_pred, error_arima, trend_binary, arima_binary=ARIMA_ALGO(df)
                lstm_pred, error_lstm, lstm_binary=LSTM_ALGO(df)
                df, lr_pred, forecast_set,mean,error_lr,lr_binary=LIN_REG_ALGO(df)
                polarity,tw_list,tw_pol,pos,neg,neutral,tweet_binary = retrieving_tweets_polarity(quote)
                cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
                #cursor.execute('DELETE FROM cache_pred WHERE quote = % s ', (quote,))
                #cursor.execute('DELETE FROM cache_twitter WHERE quote = % s ', (quote,))
                #cursor.execute('DELETE FROM images_pred WHERE quote = % s ', (quote,))
                cursor.execute('DELETE FROM cache_pred WHERE date <> % s ', (start,))
                cursor.execute('DELETE FROM cache_twitter WHERE date <> % s ', (start,))
                cursor.execute('DELETE FROM images_pred WHERE date <> % s ', (start,))
                print("deleted")
                mysql.connection.commit()
                idea, decision=recommending(df, polarity,today_stock,mean)
                print()
                print("Forecasted Prices for Next 7 days:")
                print(forecast_set)
                today_stock=today_stock.round(2)
                # cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
                arima_pred=round(arima_pred,2)
                lstm_pred=round(lstm_pred,2)
                lr_pred=round(lr_pred,2)
                error_lr=round(error_lr,2)
                error_lstm=round(error_lstm,2)
                error_arima=round(error_arima,2)
                tw_list=[i.encode('ascii', 'ignore').decode('ascii') for i in tw_list]
                try:
                    cursor.execute('INSERT INTO cache_pred VALUES (%s,%s, %s, %s, %s,%s, %s,%s, %s, %s, %s,%s, %s,%s,%s)', (quote,arima_pred,lstm_pred,lr_pred,error_lr,error_lstm,error_arima,forecast_set[0][0],forecast_set[1][0],forecast_set[2][0],forecast_set[3][0],forecast_set[4][0],forecast_set[5][0],forecast_set[6][0],start))
                    cursor.execute('INSERT INTO cache_twitter VALUES (%s,%s, %s, %s, %s,%s, %s, %s, %s,%s,%s, %s, %s, %s,%s,%s, %s)', (quote,tw_pol,tw_list[0],tw_list[1],tw_list[2],tw_list[3],tw_list[4],tw_list[5],tw_list[6],tw_list[7],tw_list[8],tw_list[9],tw_list[10],tw_list[11],idea,decision,start))
                    cursor.execute('INSERT INTO images_pred VALUES (%s,%s, %s, %s,%s,%s,%s)', (quote,arima_binary,lstm_binary,lr_binary,trend_binary,tweet_binary,start))
                    mysql.connection.commit()
                except:
                    print("It was killed")

            apple_df = pd.read_csv(''+quote+'.csv', index_col=0, parse_dates=True)
            apple_df1=apple_df.iloc[-82:-1,:]
            mc = fplt.make_marketcolors(up='tab:blue',down='tab:red',edge='lime',wick={'up':'blue','down':'red'},volume='lawngreen',)

            s  = fplt.make_mpf_style(base_mpl_style="seaborn", marketcolors=mc, mavcolors=["yellow","orange","skyblue"])
            fplt.plot(
                apple_df1,
                type="candle",
                title='JAN-MAY',
                ylabel='Price ($)',
                mav=2,
                figscale=2.0,
                style=s,
                savefig=dict(fname='static/candlestick.jpg',dpi=200,pad_inches=0.000001)
            )
            #set index
            #dynamic Graph
            figure = go.Figure(
                data = [
                    go.Candlestick(
                        x = apple_df1.index,
                        low = apple_df1.Low,
                        high = apple_df1.High,
                        close = apple_df1.Close,
                        open = apple_df1.Open,
                        increasing_line_color = 'deepskyblue',
                        decreasing_line_color = 'darkorchid'
                    )
                ]
            )
            figure.update_layout(
                title = quote+' Price',
                yaxis_title = quote+" Stock Price",
                xaxis_title = "Date (blue=up, purple=down)"
            )

            figure.write_html("templates\chart.html")
            '''end=datetime.now()
            start = datetime(end.year,end.month,end.day)
            start=start.strftime('%Y-%m-%d')
            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            quote=quote
            arima_pred=round(arima_pred,2)
            lstm_pred=round(lstm_pred,2)
            lr_pred=round(lr_pred,2)
            error_lr=round(error_lr,2)
            error_lstm=round(error_lstm,2)
            error_arima=round(error_arima,2)
            tw_list=[i.encode('ascii', 'ignore').decode('ascii') for i in tw_list]
            cursor.execute('INSERT INTO cache_pred VALUES (%s,%s, %s, %s, %s,%s, %s,%s, %s, %s, %s,%s, %s,%s,%s)', (quote,arima_pred,lstm_pred,lr_pred,error_lr,error_lstm,error_arima,forecast_set[0][0],forecast_set[1][0],forecast_set[2][0],forecast_set[3][0],forecast_set[4][0],forecast_set[5][0],forecast_set[6][0],start))
            cursor.execute('INSERT INTO cache_twitter VALUES (%s,%s, %s, %s, %s,%s, %s, %s, %s,%s,%s, %s, %s, %s,%s,%s, %s, %s, %s,%s,%s, %s, %s, %s,%s)', (quote,tw_pol,tw_list[0],tw_list[1],tw_list[2],tw_list[3],tw_list[4],tw_list[5],tw_list[6],tw_list[7],tw_list[8],tw_list[9],tw_list[10],tw_list[11],tw_list[12],tw_list[13],tw_list[14],tw_list[15],tw_list[16],tw_list[17],tw_list[18],tw_list[19],idea,decision,start))
            cursor.execute('INSERT INTO images_pred VALUES (%s,%s, %s, %s,%s,%s,%s)', (quote,arima_binary,lstm_binary,lr_binary,trend_binary,tweet_binary,start))
            mysql.connection.commit()   
            '''

            
            if quote_ratios['logo_url']==None or quote_ratios['logo_url']=='':
                return render_template('resultBSE.html',quote=quote,arima_pred=round(arima_pred,2),lstm_pred=round(lstm_pred,2),
                                lr_pred=round(lr_pred,2),open_s=today_stock['Open'].to_string(index=False),
                                close_s=today_stock['Close'].to_string(index=False),adj_close=today_stock['Adj Close'].to_string(index=False),
                                tw_list=tw_list,tw_pol=tw_pol,idea=idea,decision=decision,high_s=today_stock['High'].to_string(index=False),
                                low_s=today_stock['Low'].to_string(index=False),vol=today_stock['Volume'].to_string(index=False),
                                forecast_set=forecast_set,error_lr=round(error_lr,2),error_lstm=round(error_lstm,2),error_arima=round(error_arima,2))
            else:
                return render_template('result.html',quote=quote,arima_pred=round(arima_pred,2),lstm_pred=round(lstm_pred,2),
                                    lr_pred=round(lr_pred,2),open_s=today_stock['Open'].to_string(index=False),
                                    close_s=today_stock['Close'].to_string(index=False),adj_close=today_stock['Adj Close'].to_string(index=False),
                                    tw_list=tw_list,tw_pol=tw_pol,idea=idea,decision=decision,high_s=today_stock['High'].to_string(index=False),
                                    low_s=today_stock['Low'].to_string(index=False),vol=today_stock['Volume'].to_string(index=False), curr=str(quote_ratios['currency']),
                                    mc=str(quote_ratios['quoteType']),mrq=str(quote_ratios['mostRecentQuarter']),
                                    etr=str(quote_ratios['enterpriseToRevenue']), summary=str(quote_ratios['longBusinessSummary']),
                                    ss=str(quote_ratios['sharesShort']),pr=str(quote_ratios['profitMargins']),
                                    forecast_set=forecast_set,error_lr=round(error_lr,2),error_lstm=round(error_lstm,2),error_arima=round(error_arima,2))
    

    try:
        query = "SELECT symbol FROM history WHERE id='{userid}';".format(userid=user_id123)
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        symbols = []
    except:
        msg = ("Your session has expired")
        return render_template('login.html', msg = msg)
    try:
        cursor.execute(query)
        symbols = []
        x = cursor.fetchall()
        #print(x)
        for i in x:
            symbols.append(i['symbol'])
            #print(i)
            #print(type(i))
        print(symbols)
    except:
        print("problem in sql query")
    return render_template('enterticker.html',history_symbols=symbols)


global user123
@app.route('/stockpref', methods =['GET', 'POST'])
def stockpref():
    try:
        global user123
        if user123:
            session['stock1'] = user123['stock1']  
            session['stock2'] = user123['stock2']
            session['stock3'] = user123['stock3']
            quote1=session['stock1']
            quote2=session['stock2']
            quote3=session['stock3']
            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            cursor.execute('select symbol,count(symbol) from history group by symbol order by count(symbol) desc LIMIT 3')
            symbol_sql=cursor.fetchall()
            print(symbol_sql)
            symbol_list=[]
            if symbol_sql!=None :
                symbol_list = [i['symbol'] for i in symbol_sql]
            print(symbol_list)
            if(quote1=='NA'):
                if(len(symbol_list)==3):
                    quote1=symbol_list[0]
                else:
                    quote1='GOOGL'
            if(quote2=='NA'):
                if(len(symbol_list)==3):
                    quote2=symbol_list[1]
                else:
                    quote2='MSFT'
            if(quote3=='NA'):
                if(len(symbol_list)==3):
                    quote3=symbol_list[2]
                else:
                    quote3='AAPL'

        print(quote1,quote2,quote3)
        quote_ratios1 = yf.Ticker(quote1).info
        if quote_ratios1['logo_url']==None or quote_ratios1['logo_url']=='':
            quote1='GOOGL'
            quote_ratios1=yf.Ticker('GOOGL').info
        quote_ratios2 = yf.Ticker(quote2).info
        if quote_ratios2['logo_url']==None or quote_ratios2['logo_url']=='':
            quote2:'MSFT'
            quote_ratios2=yf.Ticker('MSFT').info
        quote_ratios3 = yf.Ticker(quote3).info
        if quote_ratios3['logo_url']==None or quote_ratios3['logo_url']=='':
            quote3='ABT'
            quote_ratios3=yf.Ticker('ABT').info
        return render_template('stockpref.html', quote1=quote1, curr1=str(quote_ratios1['currency']),
                                mc1=str(quote_ratios1['quoteType']),mrq1=str(quote_ratios1['mostRecentQuarter']),
                                etr1=str(quote_ratios1['enterpriseToRevenue']), summary1=str(quote_ratios1['longBusinessSummary']),
                                ss1=str(quote_ratios1['sharesShort']),pr1=str(quote_ratios1['profitMargins']), quote2=quote2, curr2=str(quote_ratios2['currency']),
                                mc2=str(quote_ratios2['quoteType']),mrq2=str(quote_ratios2['mostRecentQuarter']),
                                etr2=str(quote_ratios2['enterpriseToRevenue']), summary2=str(quote_ratios2['longBusinessSummary']),
                                ss2=str(quote_ratios2['sharesShort']),pr2=str(quote_ratios2['profitMargins']), quote3=quote3, curr3=str(quote_ratios3['currency']),
                                mc3=str(quote_ratios3['quoteType']),mrq3=str(quote_ratios3['mostRecentQuarter']),
                                etr3=str(quote_ratios3['enterpriseToRevenue']), summary3=str(quote_ratios3['longBusinessSummary']),
                                ss3=str(quote_ratios3['sharesShort']),pr3=str(quote_ratios3['profitMargins']))
    except:
        msg="There was a problem in fetching your stock preferences, please try again later"
        return render_template('enterticker.html',msg=msg,not_found=True)


@app.route('/updatestocks', methods =['GET', 'POST'])
def updatestocks():
    print("Entered us")
    msg=''
    if request.method == 'POST':
        print("entered if")
        email = request.form['email']
        password = request.form['password']
        stock1=request.form['stock1']
        stock2=request.form['stock2']
        stock3=request.form['stock3']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM userdetails WHERE email = % s', (email, ))
        userdetails = cursor.fetchone()
        if userdetails:
            print('entered userdetails')
            session['loggedin'] = True
            session['id'] = userdetails['id']
            session['email'] = userdetails['email']
            cursor.execute("SELECT * FROM userstock WHERE id = '{id}'".format(id = session['id'],))
            print('execute done')
            userstock= cursor.fetchone()
            userstock= cursor.fetchone()
            query="UPDATE userstock SET stock1='{stock1}', stock2='{stock2}', stock3='{stock3}' WHERE id= '{id}'".format(stock1=stock1,stock2=stock2,stock3=stock3,id=session['id'],)
            cursor.execute(query)
            print(query)
            mysql.connection.commit()
            msg="Updated your preferences"
            return render_template('updatestocks.html',msg=msg)
        else:
            msg = "Invalid email"
    return render_template('updatestocks.html', msg = msg)

@app.route('/chart')
def chart():
    return render_template('chart.html')


@app.route('/login', methods =['GET', 'POST'])
def login():
    msg = ''
    if request.method == 'POST' and 'email' in request.form and 'password' in request.form:
        email = request.form['email']
        password = request.form['password']
        print("log in...")
        try:
            login=auth.sign_in_with_email_and_password(email,password)
            verified = auth.get_account_info(login['idToken'])['users'][0]['emailVerified']
            if not verified:
                msg = ("Please verify your email")       #ACTUALLY PRINT THIS!!!!
                return render_template('login.html',msg=msg)
            print("Successful!")
        except:
            msg = "Invalid username or password!"   #ACTUALLY PRINT THIS!!!!
            return render_template('login.html',msg=msg)

        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM userdetails WHERE email = % s', (email, ))
        userdetails = cursor.fetchone()

        if userdetails:
            session['loggedin'] = True
            session['id'] = userdetails['id']
            session['email'] = userdetails['email']
            msg = 'Logged in successfully !'
            cursor.execute('SELECT * FROM userstock WHERE id = % s', (session['id'], ))
            userstock= cursor.fetchone()
            global user123
            user123=userstock
            global user_id123
            user_id123=userdetails['id']
            return render_template('dashboard.html')
        else:
            msg = ("Please register !")
    return render_template('login.html', msg = msg)

@app.route('/forgotpassword', methods =['GET', 'POST'])
def forgotpassword():
    msg = 'Please enter your registered email!'
    print("In forgot password")
    if request.method == 'POST' and 'email' in request.form:
        print('In post method')
        email = request.form['email']
        auth.send_password_reset_email(email)
        msg = 'Email sent, please check'
    print("Out of post")
    return render_template('forgotpassword.html',msg=msg)

@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('email', None)
    return redirect(url_for('index'))

  
@app.route('/register', methods =['GET', 'POST'])
def register():
    msg = ''
    if request.method == 'POST' and 'email' in request.form and 'password' in request.form:
        email = request.form['email']
        password=request.form['password']
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        address=request.form['address']
        stock1=request.form['stock1']
        stock2=request.form['stock2']
        stock3=request.form['stock3']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

         #check if password has ->
        #   1. At least 8 chars 
                # NOT DOING ->
                #   2. At least one numeric    
                #   3. At least one alphabet
                #   4. At least one uppercase

        if len(password)<8:
            msg = "Have a password greater than 8 characters!"
            return render_template('register.html', msg = msg)

            


        try:
            user = auth.create_user_with_email_and_password(email,password)
            auth.send_email_verification(user['idToken'])
            print("Done!")
            msg = 'Please go to your email to verify :)'
        except:
            print("Can't do that right now")
            msg = 'There was some problem, please try again :('
    
        print("TEST")
        cursor.execute('INSERT INTO userdetails VALUES (NULL, %s, %s, %s, %s)', (email,firstname, lastname, address))
        cursor.execute('INSERT INTO userstock VALUES (NULL, % s, % s, % s)', (stock1, stock2, stock3))
        mysql.connection.commit()
        
    elif request.method == 'POST':
        msg = 'Please fill out the form !'
    return render_template('register.html', msg = msg)

if __name__=="__main__":
    app.run(debug=True)
