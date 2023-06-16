import urllib.request
import json
import pandas as pd
from xml.etree import ElementTree


def getWebContent(url,mmsi):
    return json.loads(urllib.request.urlopen(f"{url}{mmsi}").read().decode())

def hifleet(mmsi): #船队在线
    url = 'https://v1.hifleet.com/searchVesselOL.do?keyword='
    content = getWebContent(url,mmsi)
    return content['l'],content['b']
    
def shipxy(mmsi): #船讯网
    url = 'https://www.shipxy.com/ship/GetShip?mmsi='
    content = getWebContent(url,mmsi)
    return content['data'][0]['length']/10, content['data'][0]['width']/10

def chinaports(mmsi): #港口网数据
    url = 'http://ship.chinaports.com/newshipquery/search?queryParam='
    content = getWebContent(url,mmsi)

    ship_id = content[0][-1]
    url= 'http://ship.chinaports.com/ShipInit/pospoint?&shipid='
    content = ElementTree.fromstring(urllib.request.urlopen(f"{url}{ship_id}").read().decode())
    return content[11].text,content[12].text

def msa(mmsi): #海事局
    url = 'https://ais.msa.gov.cn/api/app/baseOnMyshipsAISInfo/rtAndStaticData?mmsi='
    content = getWebContent(url,mmsi)
    
    return content['data']['length'],content['data']['width']



if __name__ == '__main__':
    df_raw = pd.DataFrame(pd.read_csv("QZHX_20211204_清洗.csv",encoding='gbk'))

    for func in [chinaports,shipxy,hifleet,msa]:
        df_raw[func.__name__+"_length"] =  -1
        df_raw[func.__name__+"_width"] =  -1

    for mmsi, content in df_raw.groupby("mmsi"):
        print(mmsi)
        for func in [chinaports,shipxy,hifleet,msa]:
            l,b  = func(mmsi)

            df_raw.loc[df_raw['mmsi']==mmsi,[func.__name__+"_length",func.__name__+"_width"]] = l,b










    df_raw.to_csv("QZHX_20211204_清洗_new.csv")
        