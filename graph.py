import plotly.graph_objects as go
import pandas as pd
quote='AAPL'
aapl = pd.read_csv(quote+'.csv').iloc[-90:,:]

#set index
df= aapl.set_index(pd.DatetimeIndex(aapl['Date'].values))
df
figure = go.Figure(
    data = [
        go.Candlestick(
            x = df.index,
            low = df.Low,
            high = df.High,
            close = df.Close,
            open = df.Open,
            increasing_line_color = 'deepskyblue',
            decreasing_line_color = 'darkorchid'
        )
    ]
)
figure.update_layout(
    title = quote+' Price',
    yaxis_title = quote+" Stock Price",
    xaxis_title = "Date"
)

figure.write_html("templates\chart.html")