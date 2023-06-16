
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
from haversine import haversine
from shp import Tif, isLonLat
from sklearn.preprocessing import MinMaxScaler
import os
import warnings
from tqdm.contrib import tzip
from tqdm import tqdm
import urllib.request
import json
from xml.etree import ElementTree
from concurrent.futures import ThreadPoolExecutor
warnings.filterwarnings("ignore")

tif = None

#可视化轨迹
def visualTrack(df):
    #获取所有的mmsi
    mmsi_list = df["mmsi"].unique()
    mmsi_list[:5]

    #可视化ais轨迹
    for mmsi in mmsi_list:
        ais = df[df['mmsi']==mmsi]
        x = ais["经度"]
        y = ais["纬度"]
        plt.plot(x,y)
        plt.show()

#根据mmsi获取船长与船宽
def getLengthBreadth(mmsi):
    def getWebContent(url):
        return json.loads(urllib.request.urlopen(f"{url}{mmsi}").read().decode())

    def hifleet(): #船队在线
        url = 'https://v1.hifleet.com/searchVesselOL.do?keyword='
        content = getWebContent(url)
        return content['l'],content['b']
        
    def shipxy(): #船讯网
        url = 'https://www.shipxy.com/ship/GetShip?mmsi='
        content = getWebContent(url)
        return content['data'][0]['length']/10, content['data'][0]['width']/10

    def chinaports(): #港口网数据
        url = 'http://ship.chinaports.com/newshipquery/search?queryParam='
        content = getWebContent(url)
        nonlocal mmsi 
        mmsi = content[0][-1]
        url= 'http://ship.chinaports.com/ShipInit/pospoint?&shipid='
        content = ElementTree.fromstring(urllib.request.urlopen(f"{url}{mmsi}").read().decode())
        return content[11].text,content[12].text

    def msa(): #海事局
        url = 'https://ais.msa.gov.cn/api/app/baseOnMyshipsAISInfo/rtAndStaticData?mmsi='
        content = getWebContent(url)
        
        return content['data']['length'],content['data']['width']


    for func in [chinaports,shipxy,hifleet,msa]:
        try:    
            l,b = func()      
            l,b = float(l),float(b)
            if(l==0 or b==0):continue
        except:
            continue
        else:
            return l,b

    return -1,-1
    

#AIS数据清洗，并写入本地
def ais_clean(ais_path):
    
    print('\n----------------------------开始处理ais数据---------------------------\n')
    print(f'文件路径：{ais_path}')
    df_raw = pd.DataFrame(pd.read_csv(ais_path,encoding='utf-8-sig'))                          #读入文件
    raw_len = df_raw.shape[0]
    print(f"共{len(df_raw['mmsi'].unique())}艘船mmsi， {raw_len} 条ais数据")

    df_raw.drop_duplicates(subset=['mmsi','lon','lat'], keep='first',inplace=True)         # 去掉相同mmsi与经纬度的行
    df_raw.drop_duplicates(subset=['mmsi','updatetime'], keep='first',inplace=True)         # 去掉相同mmsi与更新时间的行
    print(f'删除(mmsi,lon,lat),(mmsi,updatetime)相同的ais行数：{raw_len-df_raw.shape[0]}')
    
    '''
    raw_len = df_raw.shape[0]
    value_counts = df_raw['mmsi'].value_counts()                                             # 去掉信息少于10的ais
    to_remove = value_counts[value_counts<=10].index
    df_raw = df_raw[~df_raw["mmsi"].isin(to_remove)]
    print(f"删除ais信息<=10的船{len(to_remove)}艘, {raw_len-df_raw.shape[0]}条ais信息")
    '''
    
    '''    
    supplement = df_raw[(df_raw['length']==0) | (df_raw['width']==0)]['mmsi'].unique()        #补充缺失的长宽数据 单线程
    sup_lw = 0
    for mmsi in tqdm(supplement):
        length,breadth = getLengthBreadth(mmsi)
        if(length!=-1):
            df_raw[df_raw['mmsi']==mmsi][['length','width']] = length,breadth
            sup_lw = sup_lw + 1
    print(f"无长宽船只mmsi数：{len(supplement)}，补充船mmsi数：{sup_lw}")
    '''
    supplement = df_raw[(df_raw['length']==0) | (df_raw['width']==0)]['mmsi'].unique()        #补充缺失的长宽数据 多线程
    sup_lw = 0

    def multi(mmsi):
        length,breadth = getLengthBreadth(mmsi)
        if(length!=-1):
            df_raw[df_raw['mmsi']==mmsi][['length','width']] = length,breadth
            nonlocal sup_lw
            lock = threading.Lock()
            sup_lw = sup_lw + 1
            lock.release()


    with ThreadPoolExecutor(128) as t:
        for mmsi in supplement:
            t.submit(multi, mmsi=mmsi)

    print(f"无长宽船只mmsi数：{len(supplement)}，补充船mmsi数：{sup_lw}")

    print('----------------------------处理ais信息完毕---------------------------')
    return df_raw




def ais_list_clean(ais_dir,save_dir):
    for ais_file in os.listdir(ais_dir):
        ais_file_all = os.path.join(ais_dir,ais_file)
        df = ais_clean(ais_file_all)
        df.to_csv(os.path.join(save_dir,ais_dir.split('\\')[-1]))



if __name__ == '__main__':

    work_dir = r"D:\Users\CG-61\Desktop\20221221琼州海峡船舶提取结果"
    ais_dir = os.path.join(work_dir,"ais")
    ais_list_clean(ais_dir)
    



        


    # tif_file = r"data\JL1KF01B_PMSR4_20220623104620_200089965_101_0018_001_L3C_PSH\JL1KF01B_PMSR4_20220623104620_200089965_101_0018_001_L3C_PSH.tif"
    # shp_file = r"data\三亚船舶标注汇总_762.shp"
    # ais_file = r"data\sy20220623_newfile.csv"


    # png_name_set,ais_list = tifAisMatch([tif_file],[shp_file],ais_file)
    # imgAisDataset(png_name_set,ais_list)

    


