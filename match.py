
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import shapefile
from haversine import haversine
from shp import Tif, isLonLat
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import os
import warnings
from tqdm.contrib import tzip
from tqdm import tqdm
import urllib.request
import json
from xml.etree import ElementTree
warnings.filterwarnings("ignore")

tif = None

#可视化轨迹
def visualTrack(df):
    #获取所有的mmsi
    mmsi_list = df_raw["mmsi"].unique()
    mmsi_list[:5]

    #可视化ais轨迹
    for mmsi in mmsi_list:
        ais = df[df['mmsi']==mmsi]
        x = ais["经度"]
        y = ais["纬度"]
        plt.plot(x,y)
        plt.show()



#AIS预处理，返回清洗前后的值
def ais_preprocess(ais_path):
    
    df_raw = pd.DataFrame(pd.read_csv(ais_path,encoding='utf-8-sig'))                       #读入文件

    df_raw["updatetime"] = pd.to_datetime(df_raw["updatetime"])                            #.dt.date #str转时间
    df_light = df_raw.loc[:,["updatetime","mmsi","lon","lat","length","width"]]         #只保留时间、mmsi、经纬度、长宽

    return df_raw,df_light

#shp_list中寻找最接近（lon,lat）的
def find_shp(shp_file,lon, lat):
    min_diff = float("inf")

    sf = shapefile.Reader(shp_file,encoding='gb18030')
    shapes = sf.shapes()
    for shape in shapes:
        if not isLonLat(shape.points[0][0],shape.points[0][1]):
            for i in range(len(shape.points)):
                shape.points[i] = tif.geo2lonlat(shape.points[i][0],shape.points[i][1])[::-1]

        center_lon,center_lat = np.mean(np.array(shape.points)[:,0]),np.mean(np.array(shape.points)[:,1])


        diff = haversine((center_lat,center_lon), (lat,lon), unit='m')

        if diff<min_diff:
            min_diff = diff
            res_shape = shape
        print(min_diff)
    return res_shape, min_diff

def predPoly(filter,pred_time):
    if(filter.shape[0]==1): return filter["lon"].values[0],filter["lat"].values[0]
    
    pred_time = np.array(np.datetime64(pred_time.tz_localize(None)))
    sca_x,sca_lon,sca_lat = StandardScaler(),MinMaxScaler(),MinMaxScaler()

    x = filter["updatetime"].values
    
    x = x.reshape(len(x),1).astype(np.float32)
    x_norm = sca_x.fit_transform(x)

    lon = filter["lon"].values.astype(np.float32)
    lon = lon.reshape(len(lon),1)
    lon_norm = sca_lon.fit_transform(lon)

    lat = filter["lat"].values.astype(np.float32)
    lat = lat.reshape(len(lat),1)
    lat_norm = sca_lat.fit_transform(lat)

    pred_norm = sca_x.fit_transform(pred_time.reshape(1,1))

    x_norm,lon_norm,lat_norm = x_norm.reshape(len(x)),lon_norm.reshape(len(lon)),lat_norm.reshape(len(lat))

    # z_lon = np.poly1d(np.polyfit(x_norm, lon_norm, 2))
    # z_lat = np.poly1d(np.polyfit(x_norm, lat_norm, 2))

    try:

        z_lon = np.poly1d(np.polyfit(x_norm, lon_norm, 3))
        z_lat = np.poly1d(np.polyfit(x_norm, lat_norm, 3))
    except Exception as e:
        return filter["lon"].values[0],filter["lat"].values[0]

    pred_lon,pred_lat = z_lon(pred_norm),z_lat(pred_norm)
    
    # 预测值画图
    # x_intran = pd.to_datetime(sca_x.inverse_transform(x.reshape(len(x),1)).reshape(len(x)))
    # plt.plot(x_intran, sca_lon.inverse_transform(lon.reshape(len(lon),1)).reshape(len(lon)), '*',label='original values')
    # plt.plot(pred_norm.reshape(1), sca_lon.inverse_transform(pred_lon.reshape(len(pred_lon),1)).reshape(len(pred_lon)), 'r',label='polyfit values')
    # plt.show()

    pred_lon,pred_lat = sca_lon.inverse_transform(pred_lon.reshape(len(pred_lon),1))[-1],sca_lat.inverse_transform(pred_lat.reshape(len(pred_lat),1))[-1]

    #print(pred_lon,pred_lat)
    return pred_lon,pred_lat
    
def shpLengthWidth(match_shp):
    length,width = haversine(match_shp.points[1][::-1], match_shp.points[2][::-1], unit='m'),haversine(match_shp.points[0][::-1], match_shp.points[1][::-1], unit='m')
    return length,width

def imgAisDataset(png_name_set,ais_list):
    img_file=open('train_img.txt',mode='w')
    for png_name in png_name_set:
        img_file.writelines(png_name+"\n")
    ais_list.to_csv("train_ais.csv",index=False,encoding='utf-8-sig',mode='a')

# def tifAisMatch_old(tif_file_list, shp_file_list, ais_file):
    
#     df_raw, df_light = ais_clean(ais_file)
#     ais_g = df_light.groupby("mmsi")

#     start_time,time_delta_time = pd.to_datetime(os.path.split(tif_file)[-1].split("_")[2]) ,timedelta(hours=1)

#     min_diff_list,ais_list,png_name_set=[],pd.DataFrame(),set()
#     num_h1,num_m3,num_pred=0,0,0

#     for mmsi,content in ais_g:
#         length_mmsi,width_mmsi = df_raw[df_raw["mmsi"]==mmsi]["船长"].iloc[0],df_raw[df_raw["mmsi"]==mmsi]["船宽"].iloc[0]
#         def isDiffLenMatch(match_shp,diff_thresh,lw_thresh): 
#             length_shp,width_shp = shpLengthWidth(match_shp)
#             lw_loss = abs(length_mmsi-length_shp)+abs(width_mmsi-width_shp)
#             return min_diff<diff_thresh and lw_loss<lw_thresh

#         #筛选在时间段内的ais
#         filter = content[(content["updatetime"]>=start_time-time_delta_time) & (content["updatetime"]<=start_time+time_delta_time)]
#         if len(filter)==0:
#             #无结果直接从所有ais信息中筛选时间最接近的
#             index = abs(content["updatetime"]-start_time).argmin()
#             Lon,Lat = content.iloc[index]["经度"],content.iloc[index]["纬度"]
#             match_shp,min_diff,tif_id = find_shp(shp_file_list,Lon,Lat)
#             if not isDiffLenMatch(match_shp,diff_thresh=10,lw_thresh=10):continue
#             num_h1+=1
            
#         else:
#             index = abs(filter["updatetime"]-start_time).argmin()
#             if abs(filter.iloc[index]["updatetime"]-start_time)<timedelta(minutes=3):
#                 Lon,Lat = filter.iloc[index]["经度"], filter.iloc[index]["纬度"]
#                 match_shp,min_diff,tif_id = find_shp(shp_file_list,Lon,Lat)
#                 if not isDiffLenMatch(match_shp,diff_thresh=20,lw_thresh=20):continue
#                 num_m3+=1
#             else:
#                 filter_index = (-1*abs(filter["updatetime"]-start_time)).nlargest(2).index
#                 filter = filter.loc[filter_index]
#                 Lon,Lat = predPoly(filter,start_time) #Lon,Lat= filter["经度"].values[0],filter["纬度"].values[0] #不预测，直接使用离时间最近的
#                 match_shp,min_diff,tif_id = find_shp(shp_file_list,Lon,Lat)
#                 if not isDiffLenMatch(match_shp,diff_thresh=20,lw_thresh=20):continue
#                 num_pred+=1

            
#         png_name = shp2png(match_shp,200)
        
#         if ~png_name and png_name not in png_name_set:
#             png_name_set.add(png_name)
#             ais_list = pd.concat([ais_list,df_raw[df_raw["mmsi"]==mmsi].sample(n=5, replace=True)],axis=0)
#             min_diff_list.append(min_diff)

#     print(f"距离误差：{np.mean(min_diff_list)}")
#     print(f"匹配船只数：{len(png_name_set)}")
#     print(f"一个小时外匹配：{num_h1}")
#     print(f"3分钟内匹配：{num_m3}")
#     print(f"3分钟到1个小时预测匹配：{num_pred}")

#     return png_name_set,ais_list

# def findAisByShp(ais_filter, shp, start_time):
#     center_lon,center_lat = np.mean(shp.bbox[0::2]),np.mean(shp.bbox[1::2])

#     for index,ais in ais_filter.iterrows():
        


def tifAisMatch(tif_file_list, shp_file_list, ais_file):
    
    df_raw, df_light = ais_preprocess(ais_file)
    ais_g = df_light.groupby("mmsi")

    min_diff_list,ais_list,png_name_set=[],pd.DataFrame(),set()
    num_h1,num_m3,num_pred=0,0,0
    time_delta_time = timedelta(hours=1)

    for tif_file,shp_file in zip(tif_file_list, shp_file_list):
        global tif
        tif = Tif(tif_file)
        start_time= pd.to_datetime(os.path.split(tif_file)[-1].split("_")[2]).tz_localize('UTC')

        for mmsi,content in tqdm(ais_g):
            length_mmsi,width_mmsi = content["length"].iloc[0],content["width"].iloc[0]
            def isDiffLenMatch(match_shp,diff_thresh,lw_thresh): 
                length_shp,width_shp = shpLengthWidth(match_shp)
                lw_loss = abs(length_mmsi-length_shp)+abs(width_mmsi-width_shp)
                return min_diff<diff_thresh and (lw_loss<lw_thresh or length_mmsi==0 or width_mmsi==0)

            #筛选在时间段内的ais
            filter = content[(content["updatetime"]>=(start_time-time_delta_time)) & (content["updatetime"]<=(start_time+time_delta_time))]
            if len(filter)==0:
                #无结果直接从所有ais信息中筛选时间最接近的
                index = abs(content["updatetime"]-start_time).argmin()
                Lon,Lat = content.iloc[index]["lon"],content.iloc[index]["lat"]
                match_shp,min_diff = find_shp(shp_file,Lon,Lat)
                if not isDiffLenMatch(match_shp,diff_thresh=10,lw_thresh=10):continue
                num_h1+=1
                
            else:
                index = abs(filter["updatetime"]-start_time).argmin()
                if abs(filter.iloc[index]["updatetime"]-start_time)<timedelta(minutes=3):
                    Lon,Lat = filter.iloc[index]["lon"], filter.iloc[index]["lat"]
                    match_shp,min_diff = find_shp(shp_file,Lon,Lat)
                    if not isDiffLenMatch(match_shp,diff_thresh=20,lw_thresh=20):continue
                    num_m3+=1
                else:
                    filter_index = (-1*abs(filter["updatetime"]-start_time)).nlargest(10).index
                    filter = filter.loc[filter_index]
                    Lon,Lat = predPoly(filter,start_time) 
                    #Lon,Lat= filter["经度"].values[0],filter["纬度"].values[0] #不预测，直接使用离时间最近的
                    match_shp,min_diff = find_shp(shp_file,Lon,Lat)
                    if not isDiffLenMatch(match_shp,diff_thresh=20,lw_thresh=20):continue
                    num_pred+=1

                
            png_name = tif.shp2png(match_shp,200)
            
            if png_name not in png_name_set:
                png_name_set.add(png_name)
                ais_list = pd.concat([ais_list,df_raw[df_raw["mmsi"]==mmsi].sample(n=5, replace=True)],axis=0)
                min_diff_list.append(min_diff)
                print('匹配到',mmsi)
                

    print(f"距离误差：{np.mean(min_diff_list)}")
    print(f"匹配船只数：{len(png_name_set)}")
    print(f"一个小时外匹配：{num_h1}")
    print(f"3分钟内匹配：{num_m3}")
    print(f"3分钟到1个小时预测匹配：{num_pred}")

    return png_name_set,ais_list
   




if __name__ == '__main__':

    work_dir = r"D:\Users\CG-61\Desktop\20221221琼州海峡船舶提取结果"
    ais_dir = os.path.join(work_dir,"ais")
    shp_dir = os.path.join(work_dir,"shp")
    tif_dir = os.path.join(work_dir,"tif")

    ais_file_list = os.listdir(ais_dir)
    png_name_set,ais_list = set(),[]
    for ais_file in ais_file_list[2:]:
        ais_file_all = os.path.join(ais_dir,ais_file)

        tif_list_txt = os.path.join(tif_dir, ais_file.split(".")[0]+".txt")
        tif_list_txt_file = open(tif_list_txt,'r',encoding = 'utf-8')

        tif_file_list,shp_file_list=[],[]
        for line in tif_list_txt_file.readlines():
            line = line.rstrip('\\\n')
            tif_no_ext = line.split("\\")[-1]

            tif_file_list.append(os.path.join(line,tif_no_ext+".tif"))
            shp_file_list.append(os.path.join(shp_dir,tif_no_ext+".shp"))


        png_name_set,ais_list = tifAisMatch(tif_file_list,shp_file_list,ais_file_all)
        imgAisDataset(png_name_set,ais_list)


    # tif_file = r"data\JL1KF01B_PMSR4_20220623104620_200089965_101_0018_001_L3C_PSH\JL1KF01B_PMSR4_20220623104620_200089965_101_0018_001_L3C_PSH.tif"
    # shp_file = r"data\三亚船舶标注汇总_762.shp"
    # ais_file = r"data\sy20220623_newfile.csv"


    # png_name_set,ais_list = tifAisMatch([tif_file],[shp_file],ais_file)
    # imgAisDataset(png_name_set,ais_list)

    


