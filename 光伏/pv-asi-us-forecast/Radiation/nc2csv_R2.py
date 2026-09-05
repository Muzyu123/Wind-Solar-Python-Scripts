import os
import re
import pandas as pd
import xarray as xr
import gc
import logging
from datetime import datetime, timedelta
from typing import List, Optional
from multiprocessing import Pool

# ==================== 全局配置 ====================
NC_IN_PATH = "/work/zfshu/24learn/hyx-test/大创/葵花卫星/SWR" # NC文件输入路径
CSV_OUT_PATH = "/work/zfshu/24learn/hyx-test/大创/葵花卫星/场站对照/卫星csv测试/himawari_SWR_data_BJT_SSNK.csv" # CSV文件输出路径
START_TIME = datetime(2025, 10, 31, 0, 0) # 筛选时间窗口起始时间
END_TIME = datetime(2025, 11, 18, 23, 59) # 筛选时间窗口结束时间
TARGET_LON = 112.34 # 目标经度
TARGET_LAT = 29.74 # 目标纬度
FILE_CONTAINS = "02401_02401" # 筛选像素数量（不填则不筛选）
RESOLUTION = "5km" # 筛选分辨率，可选值："5km" 或 "1km"
TIME_ZONE_OFFSET_HOURS = 8  # UTC偏移小时数（北京时间=+8）

# ==================== 日志配置 ====================
log_dir = "/work/zfshu/24learn/hyx-test/大创/葵花卫星/脚本/logs" # 日志目录
# 创建日志目录
os.makedirs(log_dir, exist_ok=True)
# 生成带时间戳的日志文件名
log_file = os.path.join(log_dir, f"nc2csv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
# 配置日志记录
logging.basicConfig(
    level=logging.INFO,          # INFO及以上级别会被记录
    format='%(asctime)s - %(levelname)s - %(message)s',  # 格式：时间-级别-内容
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),  # 输出到文件
        logging.StreamHandler()                           # 输出到控制台
    ]
)
# 创建日志记录器实例
logger = logging.getLogger(__name__)

# ==================== 多核并行必需处理函数 ====================
def process_single_file(args):
    """多进程调用：处理单个 NC 文件
    参数：
        args: 包含文件路径、目标经纬度、时间范围、分辨率和提取器的元组
    返回：
        处理结果字典（包含时间、SWR值、卫星名称、经度、纬度和分辨率）
    """
    file_path, target_lon, target_lat, start_time, end_time, resolution, extractor = args
    try:
        # 解析文件名
        file_info = extractor.parse_filename_info(file_path)
        
        # 时间过滤
        file_time = file_info["observation_time"]
        if not (start_time <= file_time <= end_time):
            logger.debug(f"文件 {file_path} 时间不在指定范围内，跳过")
            return None
        
        # 分辨率过滤
        if file_info["resolution"] != resolution:
            logger.debug(f"文件 {file_path} 分辨率不符合指定，跳过")
            return None
        
        # 提取SWR值
        swr_value = extractor.extract_swr_by_position(file_path, target_lon, target_lat)
        
        # 整理数据
        return {
            "observation_time": file_info["observation_time"],
            "swr_value": swr_value,
            "satellite_name": file_info["satellite_name"],
            "longitude": target_lon,
            "latitude": target_lat,
            "resolution": resolution
        }

    except Exception as e:
        logger.error(f"文件处理失败: {file_path} | {str(e)}")
        return None

# ==================== 主数据提取核心类 ====================
class HimawariSWRExtractor:
    """葵花8/9卫星短波辐射(SWR)数据提取器"""
    
    def __init__(self):
        """
        初始化提取器
        参数：
            无
        返回：
            无
        """
    
    def parse_filename_info(self, filename: str) -> dict:
        """
        解析NC文件名
        参数：
            filename: NC文件名或完整路径，格式如H08_20240101_1200_RFLVER_FLDK_xxxxx_yyyyy.nc
        返回：
            解析后的信息字典
        """
        # 提取纯文件名（处理完整路径情况）
        pure_filename = os.path.basename(filename)
        pattern = r'H(\d{2})_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})_([Rr])FL.*?\.nc'
        match = re.match(pattern, pure_filename)
        if not match:
            logger.error(f"文件名格式不匹配: {pure_filename}")
            exit(1) # 退出程序
        
        satellite_id = match.group(1)
        year = match.group(2)
        month = match.group(3)
        day = match.group(4)
        hour = match.group(5)
        minute = match.group(6)
        resolution_flag = match.group(7)
        
        obs_time = datetime(int(year), int(month), int(day), int(hour), int(minute))
        
        return {
            "satellite_id": satellite_id,
            "satellite_name": f"葵花{satellite_id}",
            "observation_time": obs_time,
            "resolution": "5km" if resolution_flag == "R" else "1km" # 识别空间分辨率
        }
    
    def extract_swr_by_position(self, nc_file_path: str, lon: float, lat: float) -> float:
        """
        从NC文件中提取指定经纬度的短波辐射(SWR)值（线性插值）
        参数:
            nc_file_path: NC文件路径
            lon: 目标经度
            lat: 目标纬度
            
        返回:
            插值后的SWR值
        """

        swr_value = None # 初始化辐射值变量
        
        try:
            # 使用上下文管理器自动关闭数据集
            with xr.open_dataset(nc_file_path, engine='netcdf4') as ds:
                
                # SWR变量名
                swr_var_name = "SWR"
                
                # 线性插值
                swr_value = ds[swr_var_name].interp(
                    longitude=lon, 
                    latitude=lat,
                    method="linear"
                ).values.item()
                
            # logger.info(f"NC文件解析成功: {nc_file_path}，SWR值: {swr_value:.4f}")
            return swr_value
        
        except Exception as e:
            raise RuntimeError(f"解析NC文件失败 {nc_file_path}: {str(e)}")

    def extract_by_condition(
        self, 
        start_time: datetime, 
        end_time: datetime, 
        target_lon: float, 
        target_lat: float,
        nc_in_path: str,
        csv_out_path: Optional[str],
        file_contains: str,
        resolution: str,
        satellite_ids: List[str] = ["08", "09"]
    ) -> pd.DataFrame:
        """
        处理指定时间段、经纬度的SWR数据提取
        参数：
            start_time: 开始时间
            end_time: 结束时间
            target_lon: 目标经度（[-180, 180]）
            target_lat: 目标纬度（[-90, 90]）
            satellite_ids: 要提取的卫星编号列表
            resolution: 空间分辨率（5km/1km）
            nc_in_path: NC文件输入目录
            csv_out_path: 输出CSV文件路径
            
        返回：
            整理后的DataFrame
        """
        # 参数校验
        if not (-180 <= target_lon <= 180):
            raise ValueError(f"经度 {target_lon} 超出范围 [-180, 180]")
        if not (-90 <= target_lat <= 90):
            raise ValueError(f"纬度 {target_lat} 超出范围 [-90, 90]")
        if start_time > end_time:
            raise ValueError("开始时间不能晚于结束时间")

        # 初始化
        all_results = []
        
        try:
            total_files_processed = 0
            all_nc_files = []
            # 遍历根目录下所有年月/日/时子文件夹
            for root, dirs, files in os.walk(nc_in_path):
                for file in files:
                    if file.lower().endswith(".nc") and (file_contains in file):
                        all_nc_files.append(os.path.join(root, file))

            logger.info(f"总共扫描到 NC 文件数量：{len(all_nc_files)}")
                
            # 处理每个NC文件
            # ==================== 多核并行处理 ====================
            pool = Pool(processes=16)
            # 构造任务参数
            tasks = [
                (file_path, target_lon, target_lat, start_time, end_time, resolution, self)
                for file_path in all_nc_files
            ]
            
            # 使用多核处理
            results = pool.map(process_single_file, tasks)

            # 收集有效结果
            for res in results:
                if res is not None:
                    all_results.append(res)
                    total_files_processed += 1
                    
            # 构建最终DataFrame
            df = pd.DataFrame(all_results)
            if not df.empty:
                # 1. 时间转换（UTC+8）
                df["observation_time"] = pd.to_datetime(df["observation_time"]) + timedelta(hours=TIME_ZONE_OFFSET_HOURS)
                # 2. 按时间排序（修复乱序）
                df = df.sort_values("observation_time")
                df = df[["observation_time", "swr_value", "satellite_name", "longitude", "latitude", "resolution"]]

            # 最后一次性写入CSV（保证顺序正确）
            if csv_out_path and not df.empty:
                df.to_csv(csv_out_path, index=False, encoding='utf-8')
                logger.info(f"数据已写入：{csv_out_path}，共 {len(df)} 条")

            logger.info(f"数据处理完成，共处理 {total_files_processed} 个文件，有效数据 {len(df)} 条")
            return df
        
        finally:
            # 最终垃圾回收
            gc.collect()

if __name__ == "__main__":
    extractor = HimawariSWRExtractor()

    try:
        swr_df = extractor.extract_by_condition(
            start_time = START_TIME,
            end_time = END_TIME,
            target_lon = TARGET_LON,
            target_lat = TARGET_LAT,
            nc_in_path = NC_IN_PATH,
            csv_out_path = CSV_OUT_PATH,
            file_contains = FILE_CONTAINS,
            resolution = RESOLUTION,
            satellite_ids = ["08", "09"],
        )

    except Exception as e:
        logger.error(f"数据处理失败: {str(e)}", exc_info=True)
        exit(1) # 退出程序