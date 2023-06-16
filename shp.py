from osgeo import gdal
from osgeo import osr
import numpy as np
import shapefile
import cv2
from PIL import Image
import os

class Tif:
    def __init__(self,tif_file): 
        self.tif_file = tif_file;
        self.dataset = self.getDataset()
        self.prosrs, self.geosrs = self.getSRSPair()

    def getDataset(self):
        dataset = gdal.Open(self.tif_file)
        gdal.AllRegister()
        return dataset

    def getShpDataset(self):
        dataset = gdal.Open(self.tif_file)
        gdal.AllRegister()
        return dataset

    def readTifPart(self, x,y,row,col,bandsOrder=[3, 2, 1]):
        """
        读取GEO tif影像的前三个波段值，并按照R.G.B顺序存储到形状为【原长*原宽*3】的数组中
        :param imgPath: 图像存储全路径
        :param bandsOrder: RGB对应的波段顺序，如高分二号多光谱包含蓝，绿，红，近红外四个波段，RGB对应的波段为3，2，1
        :param x y row col: 左上角横纵坐标与图片的长宽
        :return: R.G.B三维数组
        """
        data = np.empty([row, col, 3], dtype=np.uint16)
        for i in range(3):
            band = self.dataset.GetRasterBand(bandsOrder[i])
            oneband_data = band.ReadAsArray(x,y,row,col)
            data[:, :, i] = oneband_data
        return data
        
    def getSRSPair(self):
        '''
        获得给定数据的投影参考系和地理参考系
        :param dataset: GDAL地理数据
        :return: 投影参考系和地理参考系
        '''
        prosrs = osr.SpatialReference()
        prosrs.ImportFromWkt(self.dataset.GetProjection())
        geosrs = prosrs.CloneGeogCS()
        return prosrs, geosrs

    def geo2lonlat(self, x, y):
        '''
        将投影坐标转为经纬度坐标（具体的投影坐标系由给定数据确定）
        :param dataset: GDAL地理数据
        :param x: 投影坐标x
        :param y: 投影坐标y
        :return: 投影坐标(x, y)对应的经纬度坐标(lon, lat)
        '''
        ct = osr.CoordinateTransformation(self.prosrs, self.geosrs)
        coords = ct.TransformPoint(x, y)
        return coords[:2]


    def lonlat2geo(self, lon, lat):
        '''
        将经纬度坐标转为投影坐标（具体的投影坐标系由给定数据确定）
        :param dataset: GDAL地理数据
        :param lon: 地理坐标lon经度
        :param lat: 地理坐标lat纬度
        :return: 经纬度坐标(lon, lat)对应的投影坐标
        '''
        ct = osr.CoordinateTransformation(self.geosrs, self.prosrs)
        coords = ct.TransformPoint(lat, lon)
        return coords[:2]

    def imagexy2geo(self, row, col):
        '''
        根据GDAL的六参数模型将影像图上坐标（行列号）转为投影坐标或地理坐标（根据具体数据的坐标系统转换）
        :param dataset: GDAL地理数据
        :param row: 像素的行号
        :param col: 像素的列号
        :return: 行列号(row, col)对应的投影坐标或地理坐标(x, y)
        '''
        trans = self.dataset.GetGeoTransform()
        px = trans[0] + col * trans[1] + row * trans[2]
        py = trans[3] + col * trans[4] + row * trans[5]
        return px, py


    def geo2imagexy(self, x, y):
        '''
        根据GDAL的六 参数模型将给定的投影或地理坐标转为影像图上坐标（行列号）
        :param dataset: GDAL地理数据
        :param x: 投影或地理坐标x
        :param y: 投影或地理坐标y
        :return: 影坐标或地理坐标(x, y)对应的影像图上行列号(row, col)
        '''
        trans = self.dataset.GetGeoTransform()
        a = np.array([[trans[1], trans[2]], [trans[4], trans[5]]])
        b = np.array([x - trans[0], y - trans[3]])
        return np.linalg.solve(a, b)  # 使用numpy的linalg.solve进行二元一次方程的求解



    def stretchImg(self, resultPath,x,y,row,col, lower_percent=0.5, higher_percent=99.5):
        """
        #将光谱DN值映射至0-255，并保存
        :param imgPath: 需要转换的tif影像路径（***.tif）
        :param resultPath: 转换后的文件存储路径(***.jpg)
        :param lower_percent: 低值拉伸比率
        :param higher_percent: 高值拉伸比率
        :return: 无返回参数，直接输出图片
        """
        RGB_Array = self.readTifPart(x,y,row,col)
        band_Num = RGB_Array.shape[2]
        JPG_Array = np.zeros_like(RGB_Array, dtype=np.uint8)
        for i in range(band_Num):
            minValue = 0
            maxValue = 255
            # 获取数组RGB_Array某个百分比分位上的值
            low_value = np.percentile(RGB_Array[:, :, i], lower_percent)
            high_value = np.percentile(RGB_Array[:, :, i], higher_percent)
            temp_value = minValue + (RGB_Array[:, :, i] - low_value) * (maxValue - minValue) / (high_value - low_value)
            temp_value[temp_value < minValue] = minValue
            temp_value[temp_value > maxValue] = maxValue
            JPG_Array[:, :, i] = temp_value
        outputImg = Image.fromarray(np.uint8(JPG_Array))
        outputImg.save(resultPath)

    def shp2png(self,shp,rowCol,save_dir=r'img'):

        pic_coords_list = []
        for lon,lat in shp.points[:4]:
            if(isLonLat(lon,lat)):
                lon,lat = self.lonlat2geo(lon, lat) #经纬度 -> 投影坐标
            pic_coords = self.geo2imagexy(lon, lat) #投影坐标 -> 图上坐标
            pic_coords_list.append(pic_coords)

        center_x,center_y = np.mean(np.array(pic_coords_list)[:,0]), np.mean(np.array(pic_coords_list)[:,1])
        half = rowCol/2
        x,y = center_x-half,center_y-half
        if(x<0 or y<0):
            return None
        

        file_name = f"{lon},{lat}.png"
        save_path = os.path.join(save_dir,file_name)
        
        self.stretchImg(save_path,x,y,rowCol,rowCol)

        return file_name




def shp2png_all(tif_file,shp_file,rowCol):
    
    gdal.AllRegister()
    dataset = gdal.Open(tif_file)
    shps = shapefile.Reader(shp_file).shapeRecords()
    tif = Tif(tif_file)
    
    for shp in shps:
        pic_coords_list = []
        for lon,lat in shp.shape.points[:4]:

            #coords = tif.lonlat2geo( lon, lat) #经纬度 -> 投影坐标
            lon,lat = tif.geo2lonlat(lon,lat)
            pic_coords = tif.geo2imagexy( lon, lat) #投影坐标 -> 图上坐标
            pic_coords_list.append(pic_coords)

        center_x,center_y = np.mean(np.array(pic_coords_list)[:,0]), np.mean(np.array(pic_coords_list)[:,1])
        half = rowCol/2
        x,y = center_x-half,center_y-half
        
        
        save_dir = r'data/split/png'
        file_name = f"{lon},{lat}.png"
        save_path = os.path.join(save_dir,file_name)
        
        tif.stretchImg(save_path,x,y,rowCol,rowCol)

        print(f"已处理{file_name}")

def isLonLat(lon,lat):
    return -180 <= lon <=180 and -90<= lat <=90

if __name__ == '__main__':
    
    # tif_file = r"\\192.168.100.52\数据管理\影像存放路径\Raster\吉林一号\2022-08-14\JL1KF01A_PMS05_20220814104530_200097070_101_0017_001_L3D_PSH\JL1KF01A_PMS05_20220814104530_200097070_101_0017_001_L3D_PSH.tif"
    # shp_file = r"D:\Users\CG-61\Desktop\20221221琼州海峡船舶提取结果\shp\JL1KF01A_PMS05_20220814104530_200097070_101_0017_001_L3D_PSH.shp"

    tif_file = r"\\192.168.100.52\数据管理\影像存放路径\Raster\吉林一号\2022-08-14\JL1KF01A_PMS05_20220814104530_200097070_101_0017_001_L3D_PSH\JL1KF01A_PMS05_20220814104530_200097070_101_0017_001_L3D_PSH.tif"
    shp_file = r"data\三亚船舶标注汇总_762.shp"

    # dataset = gdal.Open(shp_file)
    shps = shapefile.Reader(shp_file).shapeRecords()
    
    gdal.AllRegister()

    # prosrs = osr.SpatialReference()
    # prosrs.ImportFromWkt(dataset.GetProjection())
    # geosrs = prosrs.CloneGeogCS()

    # ct = osr.CoordinateTransformation(prosrs, geosrs)
    # coords = ct.TransformPoint(shps[0].shape.bbox[0], shps[0].shape.bbox[1])


    shp2png_all(tif_file,shp_file,200)

        

        



