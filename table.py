
from flask import *
from flask import Flask, render_template, request, redirect, url_for, session
from flask_mysqldb import MySQL
import yfinance as yf
import pandas as pd
app = Flask(__name__)

  
app.secret_key = 'your secret key'

@app.route('/')
def table():
    print("Done")
    global symbol123
    df=yf.Ticker('AMZN').financials
    df1=yf.Ticker('AMZN').recommendations
    df2=yf.Ticker('AMZN').quarterly_balance_sheet
    df3=yf.Ticker('AMZN').cashflow
    df4=yf.Ticker('AMZN').earnings
    df=pd.DataFrame(df)
    df1=pd.DataFrame(df1)
    df2=pd.DataFrame(df2)
    df3=pd.DataFrame(df3)
    df4=pd.DataFrame(df4)
    return render_template('table.html',  tables=[df.to_html(classes='df'),df1.to_html(classes='df1'),df2.to_html(classes='df2'),df3.to_html(classes='df3'),df4.to_html(classes='df4')], 
                            titles=['na','Financials','Recommendations','Quarterly Balance Sheet','Cashflow','Earnings'])


if __name__=="__main__":
    app.run(debug=True)