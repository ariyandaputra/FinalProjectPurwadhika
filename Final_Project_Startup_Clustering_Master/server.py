from flask import Flask, render_template, request, send_from_directory
import numpy as np
import pandas as pd
import seaborn as sns
import plotly
import plotly.graph_objs as go
import json
import joblib

app = Flask(__name__)

STARTUP_KMEANS6 = pd.read_csv('D:/Ari/Purwadhika/Files/Script/Final_Project_Startup_Clustering_Master/df_kmeans6.csv')
STARTUP_DF = pd.read_csv('D:/Ari/Purwadhika/Files/Script/Final_Project_Startup_Clustering_Master/STARTUP_DF_FINAL.csv')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/static/<path:x>')
def gal(x):
    return send_from_directory("static",x)

def scatter_plot():
    CLUSTER_DICT = {0:'Angel Startup',1:'Big Startup',2:'Venture Startup',3:'Early to Mid Startup',4:'Late Fund Startup',5:'Small Startup'}

    data = []

    for val in STARTUP_KMEANS6['Cluster'].unique():
        scatt = go.Scatter(
         x = STARTUP_KMEANS6[STARTUP_KMEANS6['Cluster'] == val]['LONGITUDE'],
         y = STARTUP_KMEANS6[STARTUP_KMEANS6['Cluster'] == val]['LATITUDE'],
         mode = 'markers',
         name = CLUSTER_DICT[val]
      )
        data.append(scatt)

    layout = go.Layout(
      title='World Map',
      title_x=0.5,
      yaxis=dict(range=[-90,90]),
      xaxis=dict(range=[-180,180])
    )
    res = {"data" : data, "layout" : layout}
    graphJSON = json.dumps(res,cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

@app.route('/scatt_fn')
def scatt_fn():
    plot = scatter_plot() 
    return render_template('scatter.html', plot=plot)


def scatter_plot1():
    CLUSTER_DICT = {0:'Angel Startup',1:'Big Startup',2:'Venture Startup',3:'Early to Mid Startup',4:'Late Fund Startup',5:'Small Startup'}

    data = []

    for val in STARTUP_KMEANS6['Cluster'].unique():
        scatt = go.Scatter(
         x = STARTUP_KMEANS6[STARTUP_KMEANS6['Cluster'] == val]['LONGITUDE'],
         y = STARTUP_KMEANS6[STARTUP_KMEANS6['Cluster'] == val]['LATITUDE'],
         mode = 'markers',
         name = CLUSTER_DICT[val]
      )
        data.append(scatt)

    layout = go.Layout(
      title='USA Map',
      title_x=0.5,
      yaxis=dict(range=[20,50]),
      xaxis=dict(range=[-140,-60])
    )
    res = {"data" : data, "layout" : layout}
    graphJSON = json.dumps(res,cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

@app.route('/scatt_fn1')
def scatt_fn1():
    plot = scatter_plot1() 
    return render_template('scatter1.html', plot=plot)


def scatter_plot2():
    CLUSTER_DICT = {0:'Angel Startup',1:'Big Startup',2:'Venture Startup',3:'Early to Mid Startup',4:'Late Fund Startup',5:'Small Startup'}

    data = []

    for val in STARTUP_KMEANS6['Cluster'].unique():
        scatt = go.Scatter(
         x = STARTUP_KMEANS6[STARTUP_KMEANS6['Cluster'] == val]['LONGITUDE'],
         y = STARTUP_KMEANS6[STARTUP_KMEANS6['Cluster'] == val]['LATITUDE'],
         mode = 'markers',
         name = CLUSTER_DICT[val]
      )
        data.append(scatt)

    layout = go.Layout(
      title='Europe Map',
      title_x=0.5,
      yaxis=dict(range=[20,70]),
      xaxis=dict(range=[-20,50])
    )
    res = {"data" : data, "layout" : layout}
    graphJSON = json.dumps(res,cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

@app.route('/scatt_fn2')
def scatt_fn2():
    plot = scatter_plot2() 
    return render_template('scatter2.html', plot=plot)


def pie_plot():
    CLUSTER_DICT = {0:'Angel Startup',1:'Big Startup',2:'Venture Startup',3:'Early to Mid Startup',4:'Late Fund Startup',5:'Small Startup'}

    result = STARTUP_KMEANS6['Cluster'].value_counts()

    labels_source = []
    values_source = []

    for item in result.iteritems():
        labels_source.append(CLUSTER_DICT[item[0]])
        values_source.append(item[1])

    data_source = [
        go.Pie(
            labels=labels_source,
            values=values_source
        )
    ]

    layout_source = go.Layout(
        title='Cluster Pie Chart',
        title_x=0.5
    )

    final = {"data" : data_source, "layout" : layout_source}

    graphJSON = json.dumps(final, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

@app.route('/pie_fn')
def pie_fn():
    plot_source = pie_plot()
    return render_template('pie.html', plot=plot_source)


def bar_plot():
    data = []

    CLUSTER_DICT = {0:'Angel Startup',1:'Big Startup',2:'Venture Startup',3:'Early to Mid Startup',4:'Late Fund Startup',5:'Small Startup'}

    NAME_LIST = []
    for i in STARTUP_KMEANS6['Cluster'].value_counts().index:
        NAME_LIST.append(CLUSTER_DICT[i])

    bar = go.Bar(
        x = NAME_LIST,
        y = STARTUP_KMEANS6['Cluster'].value_counts().values,
        text = STARTUP_KMEANS6['Cluster'].value_counts().values,
        textposition = 'auto'
                )
            
    data.append(bar)

    title = 'Cluster Count'

    layout = go.Layout(
        title=title,
        title_x=0.5,
    )

    final = {"data" : data, "layout" : layout}

    graphJSON = json.dumps(final, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

@app.route('/bar_fn')
def bar_fn():

    plot = bar_plot()

    return render_template(
        'bar.html', 
        plot=plot, 

    )


def bar_plot1():
    data = []

    CLUSTER_DICT = {'CLUSTER_0':'Angel Startup','CLUSTER_1':'Big Startup','CLUSTER_2':'Venture Startup','CLUSTER_3':'Early to Mid Startup','CLUSTER_4':'Late Fund Startup','CLUSTER_5':'Small Startup'}

    FUND_COUNT_DESC = pd.read_pickle('D:/Ari/Purwadhika/Files/Script/Final_Project_Startup_Clustering_Master/FUND_COUNT_DESC.pkl')

    NAME_LIST = []
    for i in FUND_COUNT_DESC.T['MEAN'].index:
        NAME_LIST.append(CLUSTER_DICT[i])

    bar = go.Bar(
        x = NAME_LIST,
        y = FUND_COUNT_DESC.T['MEAN'].values,
        text = FUND_COUNT_DESC.T['MEAN'].values,
        textposition = 'auto'
                )
            
    data.append(bar)

    title = 'Funding Count'

    layout = go.Layout(
        title=title,
        title_x=0.5,
    )

    final = {"data" : data, "layout" : layout}

    graphJSON = json.dumps(final, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

@app.route('/bar_fn1')
def bar_fn1():

    plot = bar_plot1()

    return render_template(
        'bar1.html', 
        plot=plot, 

    )


def bar_plot2():
    data = []

    CLUSTER_DICT = {'CLUSTER_0':'Angel Startup','CLUSTER_1':'Big Startup','CLUSTER_2':'Venture Startup','CLUSTER_3':'Early to Mid Startup','CLUSTER_4':'Late Fund Startup','CLUSTER_5':'Small Startup'}
    
    RAISED_USD_SUM_DESC = pd.read_pickle('D:/Ari/Purwadhika/Files/Script/Final_Project_Startup_Clustering_Master/RAISED_USD_SUM_DESC.pkl')

    NAME_LIST = []
    for i in RAISED_USD_SUM_DESC.T['MEAN'].index:
        NAME_LIST.append(CLUSTER_DICT[i])

    bar = go.Bar(
        x = NAME_LIST,
        y = RAISED_USD_SUM_DESC.T['MEAN'].values,
        text = RAISED_USD_SUM_DESC.T['MEAN'].values,
        textposition = 'auto'
                )
            
    data.append(bar)

    title = 'Total Funding Average'

    layout = go.Layout(
        title=title,
        title_x=0.5,
    )

    final = {"data" : data, "layout" : layout}

    graphJSON = json.dumps(final, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

@app.route('/bar_fn2')
def bar_fn2():

    plot = bar_plot2()

    return render_template(
        'bar2.html', 
        plot=plot, )

@app.route('/predict')
def predict():
    return render_template('predict.html')


@app.route('/Cluster_Result', methods=["POST", "GET"])
def Cluster_Predict():
    if request.method == "POST":
        input = request.form
        AvgFund = float(input['AvgFund'])
        Country = float(input['Country'])
        SeriesA = float(input['SeriesA'])
        SeriesB = float(input['SeriesB'])
        SeriesC = float(input['SeriesC'])
        Angel = float(input['Angel'])
        Venture = float(input['Venture'])
        PostIPO = float(input['PostIPO'])
        Other = float(input['Other'])
        FundCount = (SeriesA+SeriesB+SeriesC+Angel+Venture+PostIPO+Other)
    
        pred = gbc.predict([[FundCount,AvgFund,Country,SeriesA,SeriesB,SeriesC
        ,Angel,Venture,PostIPO,Other]])[0]

        CLUSTER_LIST = ['Angel Startup','Big Startup','Venture Startup','Early to Mid Startup','Late Fund Startup','Small Startup']
        CLUSTER_PRED = CLUSTER_LIST[pred]

        RAISED_USD_SUM_DESC = pd.read_pickle('D:/Ari/Purwadhika/Files/Script/Final_Project_Startup_Clustering_Master/RAISED_USD_SUM_DESC.pkl')
        CLUSTER_DICT = {0:'CLUSTER_0',1:'CLUSTER_1',2:'CLUSTER_2',3:'CLUSTER_3',4:'CLUSTER_4',5:'CLUSTER_5'}
        CLUSTER_LONG = CLUSTER_DICT[pred]
        MINI = RAISED_USD_SUM_DESC[CLUSTER_LONG]['MIN']
        MAXI = RAISED_USD_SUM_DESC[CLUSTER_LONG]['MAX']
        MEANI= RAISED_USD_SUM_DESC[CLUSTER_LONG]['MEAN']

        FUNDTYPE_LIST = [SeriesA , SeriesB , SeriesC , Angel , Venture , PostIPO , Other]
        FUNDTYPE_NAME = ['SeriesA' , 'SeriesB' , 'SeriesC' , 'Angel' , 'Venture' , 'PostIPO' , 'Other']
        
        FUNDTYPE = []

        for i in range(int(FundCount)):
            if FUNDTYPE_LIST[i] == 1:
                FUNDTYPE.append(FUNDTYPE_NAME[i])
            else:
                pass
        if len(FUNDTYPE) == 0:
            FUNDTYPE.append('None')
        else:
            pass

        return render_template('result.html',
        data=input, AvgFund=int(input['AvgFund'])
        ,FundCount=int(FundCount), prediction = CLUSTER_PRED,
        mini=int(MINI), maxi=int(MAXI), meani=int(MEANI),
        FundType=FUNDTYPE)


if __name__ == '__main__':
    gbc = joblib.load('D:/Ari/Purwadhika/Files/Script/Final_Project_Startup_Clustering_Master/GBC_STARTUP')
    app.run(debug=True, port=4000)