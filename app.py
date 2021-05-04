
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
import nltk
import smtplib
nltk.download('punkt')
  
  
app = Flask(__name__)
  
  
app.secret_key = 'your secret key'
  
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'Shubhi@2107'
app.config['MYSQL_DB'] = 'projectlogin'
  
mysql = MySQL(app)

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
   return render_template('index.html')

@app.route('/aboutus')
def aboutus():
   return render_template('aboutus.html')
  
@app.route('/faq')
def faq():
   return render_template('faq.html')

@app.route('/forgotpassword')
def forgotpassword():
   return render_template('forgotpassword.html')
  
@app.route('/contactus', methods = ["GET","POST"])
def contactus():
    msg=''
    if request.method=='POST':
        print('post')
        fullname = request.form.get("fullname")
        email = request.form.get('email')
        password = request.form.get('password')
        subject = request.form.get('subject')
        message = request.form.get('message')
        def send_email(email, password, subject, message):
            try:
                server = smtplib.SMTP('smtp.gmail.com:587')
                server.ehlo()
                server.starttls()
                server.login(email, password)
                message = 'Subject: {}\n\n{}'.format(subject, message)
                server.sendmail("voyagecapital04@gmail.com", "voyagecapital04@gmail.com", message)
                server.quit()
                print("Success: Email sent!")
                msg="Your query has been successfully submitted. We will contact you shortly."
            except ValueError:
                print(ValueError)
                print("Email failed to send.")
                msg="Contact process failed"

        send_email(email, password, subject, message)
    return render_template('contactus.html', msg=msg)

@app.route('/result')
def result():
   return render_template('result.html')

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

@app.route('/tables', methods=("POST", "GET"))
def table(quote):
    print(quote)
    df=yf.Ticker('MSFT').financials
    df1=yf.Ticker('MSFT').recommendations
    df2=yf.Ticker('MSFT').quarterly_balance_sheet
    df3=yf.Ticker('MSFT').cashflow
    df4=yf.Ticker('MSFT').earnings
    return render_template('table.html',  tables=[df.to_html(classes='data'),df1.to_html(classes='data'),df2.to_html(classes='data'),df3.to_html(classes='data'),df4.to_html(classes='data')], 
                            titles=['na','Financials','Recommendations','Quarterly Balance Sheet','Cashflow','Earnings'])


@app.route('/enterticker', methods =['GET', 'POST'])
def enterticker():
    if request.method == 'POST' and 'symbol' in request.form:
        symbol = request.form['symbol']
        def get_historical(quote):
            end = datetime.now()
            start = datetime(end.year-2,end.month,end.day)
            data = yf.download(quote, start=start, end=end)
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
        
                quantity = Quantity_date.values
                size = int(len(quantity) * 0.80)
                train, test = quantity[0:size], quantity[size:len(quantity)]
                #fit in model
                predictions = arima_model(train, test)
        
                #plot graph
                fig = plt.figure(figsize=(7.2,4.8),dpi=65)
                plt.plot(test,label='Actual Price')
                plt.plot(predictions,label='Predicted Price')
                plt.legend(loc=4)
                plt.savefig('static/ARIMA.png')
                plt.close(fig)
                print()
                print("##############################################################################")
                arima_pred=predictions[-2]
                print("Tomorrow's",quote," Closing Price Prediction by ARIMA:",arima_pred)
                #rmse calculation
                error_arima = math.sqrt(mean_squared_error(test, predictions))
                print("ARIMA RMSE:",error_arima)
                print("##############################################################################")
                return arima_pred, error_arima
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
            return lstm_pred,error_lstm
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
            
            error_lr = math.sqrt(mean_squared_error(y_test, y_test_pred))
            
            
            #Forecasting
            forecast_set = clf.predict(X_to_be_forecasted)
            forecast_set=forecast_set*(1.04)
            mean=forecast_set.mean()
            lr_pred=forecast_set[0,0]
            print()
            print("##############################################################################")
            print("Tomorrow's ",quote," Closing Price Prediction by Linear Regression: ",lr_pred)
            print("Linear Regression RMSE:",error_lr)
            print("##############################################################################")
            return df, lr_pred, forecast_set, mean, error_lr
                
        #**************** SENTIMENT ANALYSIS **************************
        def retrieving_tweets_polarity(symbol):
            stock_ticker_map = pd.read_csv('Yahoo-Finance-Ticker-Symbols.csv')
            stock_full_form = stock_ticker_map[stock_ticker_map['Ticker']==symbol]
            symbol = stock_full_form['Name'].to_list()[0][0:12]

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
            return global_polarity,tw_list,tw_pol,pos,neg,neutral


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
            return render_template('index.html',not_found=True)
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
            
            table(quote)

            arima_pred, error_arima=ARIMA_ALGO(df)
            lstm_pred, error_lstm=LSTM_ALGO(df)
            df, lr_pred, forecast_set,mean,error_lr=LIN_REG_ALGO(df)
            polarity,tw_list,tw_pol,pos,neg,neutral = retrieving_tweets_polarity(quote)
            
            idea, decision=recommending(df, polarity,today_stock,mean)
            print()
            print("Forecasted Prices for Next 7 days:")
            print(forecast_set)
            today_stock=today_stock.round(2)
            #return render_template('index.html')
            return render_template('result.html',quote=quote,arima_pred=round(arima_pred,2),lstm_pred=round(lstm_pred,2),
                                lr_pred=round(lr_pred,2),open_s=today_stock['Open'].to_string(index=False),
                                close_s=today_stock['Close'].to_string(index=False),adj_close=today_stock['Adj Close'].to_string(index=False),
                                tw_list=tw_list,tw_pol=tw_pol,idea=idea,decision=decision,high_s=today_stock['High'].to_string(index=False),
                                low_s=today_stock['Low'].to_string(index=False),vol=today_stock['Volume'].to_string(index=False), curr=str(quote_ratios['currency']),
                                mc=str(quote_ratios['quoteType']),mrq=str(quote_ratios['mostRecentQuarter']),
                                etr=str(quote_ratios['enterpriseToRevenue']), summary=str(quote_ratios['longBusinessSummary']),
                                ss=str(quote_ratios['sharesShort']),pr=str(quote_ratios['profitMargins']),
                                forecast_set=forecast_set,error_lr=round(error_lr,2),error_lstm=round(error_lstm,2),error_arima=round(error_arima,2))
    return render_template('enterticker.html')

@app.route('/stockpref', methods =['GET', 'POST'])
def stockpref():
    return render_template('stockpref.html')

@app.route('/login', methods =['GET', 'POST'])
def login():
    msg = ''
    if request.method == 'POST' and 'email' in request.form and 'password' in request.form:
        email = request.form['email']
        password = request.form['password']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM userdetails WHERE email = % s AND password = % s', (email, password, ))
        userdetails = cursor.fetchone()
        if userdetails:
            session['loggedin'] = True
            session['id'] = userdetails['id']
            session['email'] = userdetails['email']
            msg = 'Logged in successfully !'
            cursor.execute('SELECT * FROM userstock WHERE id = % s', (session['id'], ))
            userstock= cursor.fetchone()
            if userstock:
                session['stock1'] = userstock['stock1']  
                session['stock2'] = userstock['stock2']
                session['stock3'] = userstock['stock3']
                quote1=session['stock1']
                quote2=session['stock2']
                quote3=session['stock3']          
            quote_ratios1 = yf.Ticker(quote1).info
            quote_ratios2 = yf.Ticker(quote2).info
            quote_ratios3 = yf.Ticker(quote3).info
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
            return render_template('enterticker.html')
        else:
            msg = 'Incorrect email / password !'
    return render_template('login.html', msg = msg)
  
@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('email', None)
    return redirect(url_for('login'))

  
@app.route('/register', methods =['GET', 'POST'])
def register():
    msg = ''
    if request.method == 'POST' and 'email' in request.form and 'username' in request.form and 'password' in request.form:
        email = request.form['email']
        username = request.form['username']
        password=request.form['password']
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        mobile=request.form['mobile']
        address=request.form['address']
        stock1=request.form['stock1']
        stock2=request.form['stock2']
        stock3=request.form['stock3']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM userdetails WHERE email = % s', (email, ))
        account = cursor.fetchone()
        if account:
            msg = 'Account already exists !'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address !'
        elif not password or not email or not username:
            msg = 'Please fill out the form !'
        else:
            print("TEST")
            cursor.execute('INSERT INTO userdetails VALUES (NULL, % s, % s, % s, %s, %s, %s, %s)', (email, username, password,firstname, lastname, mobile, address))
            mysql.connection.commit()
            cursor.execute('INSERT INTO userstock VALUES (NULL, % s, % s, % s)', (stock1, stock2, stock3))
            mysql.connection.commit()
            msg = 'You have successfully registered !'
    elif request.method == 'POST':
        msg = 'Please fill out the form !'
    return render_template('register.html', msg = msg)

if __name__=="__main__":
    app.run(debug=True)
