# Name: binance.py
# Author: Scott Condie (scott_condie@byu.edu)
# Description: Collects data from the binance.us api.

import time
import datetime
from pprint import pprint
import requests
import collections.abc as collections
import pandas as pd 
import numpy as np 
import sqlite3
import peewee as pw
import sys, os
import peewee as pw
from datetime import datetime, timedelta
#from exchange_libs.credentials import get_db
import credentials as creds

db = creds.get_db('binance')

class BaseModel(pw.Model):
    class Meta:
        database = db

class Snapshot(BaseModel):
    product_id = pw.CharField()
    lastUpdateId = pw.IntegerField()
    request_sent = pw.DateTimeField()
    time = pw.DateTimeField(index=True)
    bids = pw.TextField()
    asks = pw.TextField()

class L2Update(BaseModel):
    product_id = pw.CharField()
    time = pw.DateTimeField()
    type = pw.CharField()
    changes = pw.CharField()

class TickerUpdate(BaseModel):
    product_id = pw.CharField()
    time = pw.DateTimeField()
    type = pw.CharField()
    price = pw.CharField()
    side = pw.CharField()
    last_size = pw.CharField()
    best_bid = pw.CharField()
    best_ask = pw.CharField()
    open_24h = pw.CharField()
    volume_24h = pw.CharField()
    low_24h = pw.CharField()
    high_24h = pw.CharField()
    volume_30d = pw.CharField()
    best_bid_size = pw.CharField()
    best_ask_size = pw.CharField()
    trade_id = pw.IntegerField()
    sequence = pw.IntegerField()



host = "https://api.binance.us"
prefix = "/api/v3/depth"
headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}


currency_pair = 'BTCUSDT'
sz = 100 #We can go up to 500, has a weight of 25 when at 500 try 100

query_param = f"symbol={currency_pair}&limit={sz}"

def query_orderbook():
    request_sent = datetime.now()
    r = requests.request('GET', host + prefix + "?" + query_param, headers=headers)
    resp = r.json()
    resp['time'] = datetime.now()
    resp['product_id'] = currency_pair
    resp['request_sent'] = request_sent
    return resp


if __name__=="__main__":
    db.connect()

    if sys.argv[2] == 'True':
        try:
            db.drop_tables([Snapshot, L2Update, TickerUpdate])
        except:
            pprint("Problem deleting the tables.")
            pass

        db.create_tables([Snapshot, L2Update, TickerUpdate], safe=True)
        pprint("Created tables.")


    pprint("Created tables.")
    
    num_min = sys.argv[1]  # Adjust this value based on your requirement
    start_time = datetime.now()

    while (datetime.now() - start_time) < timedelta(minutes=float(num_min)):
        new_ob = Snapshot.create(**query_orderbook())  # Adjust based on your model fields
        time.sleep(5 - time.monotonic() % 1)

    db.close()
