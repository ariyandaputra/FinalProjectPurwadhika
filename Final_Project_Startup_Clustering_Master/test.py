from flask import Flask, render_template, request, send_from_directory
import numpy as np
import pandas as pd
import seaborn as sns
import plotly
import plotly.graph_objs as go
import json
import joblib

gbc = joblib.load('D:/Ari/Purwadhika/Files/Script/Final_Project_Startup_Clustering_Master/GBC_STARTUP')

TEST = [2,500000,0,1,1,0,0,0,0,0]

pred = gbc.predict([TEST])

print(pred)