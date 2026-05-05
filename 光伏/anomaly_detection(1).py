import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

X_COLUMN_NAME = 'radiation'
Y_COLUMN_NAME = 'pv_power'

class AnomalyDetection:
    def __init__(self , df , rated_ws , rated_power = 4000):
        self.df = df
        self.rated_ws = rated_ws
        self.rated_power = rated_power
        self.df['mark'] = 0


    def basic_clean(self):
        #识别超出标定值的异常数据
        # 风速异常
        if 'mark' not in self.df.columns:
            self.df['mark'] = 0
            
        # self.df.loc[self.df[X_COLUMN_NAME] >= self.rated_ws, 'mark'] = 1
        # self.df.loc[self.df[X_COLUMN_NAME] < 0, 'mark'] = -1
        
        # # 功率异常（合并条件，而不是覆盖）
        # self.df.loc[self.df[Y_COLUMN_NAME] >= self.rated_power, 'mark'] = 1
        # self.df.loc[self.df[Y_COLUMN_NAME] < 0, 'mark'] = -1
        self.df.loc[(self.df[X_COLUMN_NAME] >= self.rated_ws) & (self.df[Y_COLUMN_NAME] < 0) , 'mark'] = -1
        self.df.loc[self.df[Y_COLUMN_NAME] > self.rated_power , 'mark'] = -1
        
        normal_mask = self.df['mark'] == 0
        abnormal_mask = self.df['mark'] != 0

        print(f"total:{len(self.df)} lines")
        print(f"nomal:{normal_mask.sum()} lines")
        print(f"{X_COLUMN_NAME} or y erro:{abnormal_mask.sum()} lines")

        print(f"erro x = \n{self.df.loc[abnormal_mask , X_COLUMN_NAME]}\n erro y = \n{self.df.loc[abnormal_mask , Y_COLUMN_NAME]}")
        
        # # 创建画布
        # plt.figure(figsize=(12, 8))
        
        # # 绘制正常数据（蓝色）
        # plt.scatter(
        #     self.df.loc[normal_mask, X_COLUMN_NAME],
        #     self.df.loc[normal_mask, Y_COLUMN_NAME],
        #     s=8, c='blue', alpha=0.5, label='正常数据'
        # )
        
        # # 绘制异常数据（红色）
        # if abnormal_mask.any():
        #     plt.scatter(
        #         self.df.loc[abnormal_mask, X_COLUMN_NAME],
        #         self.df.loc[abnormal_mask, Y_COLUMN_NAME],
        #         s=8, c='red', alpha=0.7, label='异常数据'
        #     )
        
        # # 添加额定线
        # plt.axhline(y=self.rated_power, color='green', linestyle='--', 
        #             linewidth=1.5, label=f'额定功率 ({self.rated_power} kW)')
        # plt.axvline(x=self.rated_ws, color='orange', linestyle='--', 
        #             linewidth=1.5, label=f'额定风速 ({self.rated_ws} m/s)')
        
        # # 设置标签
        # plt.xlabel('风速 (m/s)', fontsize=12)
        # plt.ylabel('功率 (kW)', fontsize=12)
        # plt.title('功率曲线 (红色为异常值)', fontsize=14)
        # plt.legend(loc='best')
        # plt.grid(True, alpha=0.3)
        
        # plt.show()
        return
    
    def data_classification(self):
        #4类异常数据
        #1.功率曲线上方堆积型异常数据
        #2.曲线上方分散型异常数据
        #3.曲线下方堆积型异常数据
        #4.曲线下方分散型异常数据
        return

    def edge_clean(self):
        #四分位法为基础，根据异常数据形态，判断采用双向四分位还是双向单边四分位
        #若功率曲线上方异常数据数量远小于曲线下方，采用双向单边四分位法
        #反之使用双向四分位法
        if self.df is None or len(self.df) == 0:
            print("no data to clean")
            return self.df
        
        ws_para = 0.1
        ratio_para = 0.7

        ws = self.df[X_COLUMN_NAME].values
        power = self.df[Y_COLUMN_NAME].values

        ws_bins = np.arange(0 , ws.max() , ws_para)
        # ws_center = (ws_bins[:-1] + ws_bins[1:])/2

        upper_bounds = []
        lower_bounds = []

        upper_abnormal_count = 0
        lower_abnormal_count = 0
        total_point = 0

        for i in range(len(ws_bins)-1):
            temp = (ws >= ws_bins[i]) & (ws <= ws_bins[i+1])
            power_bin = power[temp]

            if len(power_bin) < 10:  # 数据太少，跳过
                upper_bounds.append(np.nan)
                lower_bounds.append(np.nan)
                continue

            q1 = np.percentile(power_bin , 25)
            q3 = np.percentile(power_bin , 75)
            iqr = q3 - q1

            lower_bound = q1 - 1.5*iqr
            upper_bound = q3 + 1.5*iqr

            upper_abnormal_count += np.sum(power_bin > upper_bound)
            lower_abnormal_count += np.sum(power_bin < lower_bound)

            upper_bounds.append(upper_bound)
            lower_bounds.append(lower_bound)

        total_abnormal_count = upper_abnormal_count + lower_abnormal_count
        if total_abnormal_count == 0:
            print("no abnormal data")
            return self.df

        upper_ratio = upper_abnormal_count / total_abnormal_count
        lower_ratio = lower_abnormal_count / total_abnormal_count

        print(f"upper abnormal count = {upper_abnormal_count} , ratio = {upper_ratio}")
        print(f"lower abnormal count = {lower_abnormal_count} , ratio = {lower_ratio}")

        if upper_ratio > ratio_para:
            print("upper more, single side clean")
            clean_mask = self._single_side_clean(ws = ws,
                                                 power = power,
                                                 ws_bins = ws_bins,
                                                 bounds = upper_bounds,
                                                 upper = True)
        elif lower_ratio > ratio_para:
            print("lower more, single side clean")
            clean_mask = self._single_side_clean(ws = ws,
                                                 power = power,
                                                 ws_bins = ws_bins,
                                                 bounds = lower_bounds,
                                                 upper = False)
        else:
            print("balanced, double side clean")
            clean_mask = self._double_side_clean(ws = ws,
                                                 power = power,
                                                 ws_bins = ws_bins,
                                                 upper_bounds = upper_bounds,
                                                 lower_bounds = lower_bounds)
            
        inital_count = len(self.df)
        self.df.loc[~clean_mask , 'mark'] = -2
        cleaned_count = (self.df['mark'] == 1).sum()
        print("edge clean done")
        print(f"initial data:{inital_count}lines")
        print(f"cleaned data:{cleaned_count}lines")
        print(f"cleaned abnormal data{inital_count - cleaned_count}lines,({(inital_count - cleaned_count)/inital_count*100:.1f}%))")

        return self.df
    
    def _single_side_clean(self , ws , power , ws_bins , bounds , upper = False):
        clean_mask = np.ones(len(ws) , dtype = bool)
        for i in range(len(ws_bins)-1):
            bin_mask = (ws >= ws_bins[i]) & (ws <= ws_bins[i+1])
            if np.isnan(bounds[i]):
                continue
            if upper:
                clean_mask[bin_mask] = power[bin_mask] <= bounds[i]
            else:
                clean_mask[bin_mask] = power[bin_mask] >= bounds[i]
        return clean_mask
    
    def _double_side_clean(self , ws , power , ws_bins , upper_bounds , lower_bounds ):
        clean_mask = np.ones(len(ws) , dtype = bool)
        for i in range(len(ws_bins)-1):
            bin_mask = (ws >= ws_bins[i]) & (ws <= ws_bins[i+1])
            if np.isnan(lower_bounds[i]) or np.isnan(upper_bounds[i]):
                continue
            clean_mask[bin_mask] = (power[bin_mask] >= lower_bounds[i]) & \
                                   (power[bin_mask] <= upper_bounds[i])
        return clean_mask
    
    
    def stack_clean(self , num = 8 , installed_capacity = None , distance_threshold = None ):
        # 是否采用取决于是否存在堆积型数据
        # 以类中心为判据的改进k-means聚类方法进行识别
        # 参数:
        # num: 聚类类别数，默认8
        # installed_capacity: 装机容量(kW)，默认使用额定功率
        # distance_threshold: 异常簇距离阈值（None时自动计算）
        df_clean = self.df[self.df[Y_COLUMN_NAME] > 0].copy()
        if len(df_clean) == 0:
            print("no active power")
            return self.df
        X = df_clean[[X_COLUMN_NAME, Y_COLUMN_NAME]].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        np.random.seed(42)
        n_samples = len(X_scaled)
        n_clusters = min(num , n_samples)

        random_indices = np.random.choice(n_samples , n_clusters , replace = False)
        centroids = X_scaled[random_indices].copy()

        labels = np.zeros(n_samples , dtype = int)
        indices = np.random.permutation(n_samples)
        print("k-means start")

        for idx in indices:
            point = X_scaled[idx]
            distances = np.linalg.norm(centroids - point , axis = 1)
            nearest_cluster = np.argmin(distances)
            labels[idx] = nearest_cluster

            cluster_points = X_scaled[labels == nearest_cluster]
            if len(cluster_points) > 0:
                centroids[nearest_cluster] = np.mean(cluster_points , axis = 0)

        print("sum of groups")
        for i in range(n_clusters):
            count = np.sum(labels == i)
            print(f"group {i} : {count} points({count/n_samples *100:.1f}%)")

        centroids_original = scaler.inverse_transform(centroids)
        centroid_wind = centroids_original[:,0]
        centroid_power = centroids_original[:,1]

        # 设置装机容量
        if installed_capacity is None:
            installed_capacity = self.rated_power
        
        # 功率间隔（装机容量的2%）
        power_interval = installed_capacity * 0.02
        power_max = df_clean[Y_COLUMN_NAME].max()
        power_bins = np.arange(0 , power_max + power_interval , power_interval)

        bin_centers = []
        bin_avg_wind = []

        for i in range(len(power_bins)-1):
            mask = (df_clean[Y_COLUMN_NAME] >= power_bins[i]) & (df_clean[Y_COLUMN_NAME] < power_bins[i+1])
            if mask.sum() > 10:
                avg_wind = df_clean.loc[mask , X_COLUMN_NAME].mean()
                bin_centers.append((power_bins[i] + power_bins[i+1]) / 2)
                bin_avg_wind.append(avg_wind)
        if len(bin_centers) < 3:
            print("lack of data , skip stack clean")
            return self.df
        
        baseline_func = interp1d(bin_centers , bin_avg_wind , kind = 'linear' ,
                                 fill_value = 'extrapolate' , bounds_error = False)
        baseline_at_centriod = baseline_func(centroid_power)
        cluster_distances = np.abs(centroid_wind - baseline_at_centriod)

        # 确定阈值
        if distance_threshold is None:
            mean_dist = np.mean(cluster_distances)
            std_dist = np.std(cluster_distances)
            distance_threshold = mean_dist + 2 * std_dist
        abnormal_clusters = np.where(cluster_distances > distance_threshold)[0]
        normal_clusters = np.where(cluster_distances <= distance_threshold)[0]
        print("baseline finished")
        print(f"normal clusters = {len(normal_clusters)}")
        print(f"abnormal clusters = {len(abnormal_clusters)}")

        abnormal_mask = np.isin(labels , abnormal_clusters)
        abnormal_indices = df_clean.index[abnormal_mask]

        self.df.loc[abnormal_indices , 'mark'] = -4
        
        print("stack clean finished")
        print(f"abnormal points: {len(abnormal_indices)}")
        print(f"abnormal ratio: {len(abnormal_indices)/len(df_clean)*100:.2f}%")

        return
    
    def scattered_clean(self , eps = None , min_samples = 5 , sample_size = 100000 , plot_k_distance = True):
        # 采用双DBSCAN聚类法进行识别
        # 参数:
        # eps: 邻域半径（None时自动计算）
        # min_samples: 最小邻居数，默认5
        # sample_size: 采样大小（数据量大时使用）
        # plot_k_distance: 是否绘制k距离图辅助选择eps
        # 以下为基础DBSCAN模型
        df_clean = self.df[self.df[Y_COLUMN_NAME] > 0].copy()
        X = df_clean[[X_COLUMN_NAME, Y_COLUMN_NAME]].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if len(X_scaled) > sample_size:
            print(f"data too large ,random pick {sample_size} points")
            np.random.seed(42)
            sample_index = np.random.choice(len(X_scaled) , sample_size , replace = False)
            X_scaled = X_scaled[sample_index]
            df_sampled = df_clean.iloc[sample_index]
        else:
            df_sampled = self.df
            sample_index = None

        if eps is None:
            eps = self._find_optimal_eps_for_DBSCAN(X_scaled = X_scaled ,
                                                    min_samples = min_samples ,
                                                    plot_k_distance = plot_k_distance)
            print(f"auto selcet eps = {eps}")
        
        dbscan = DBSCAN(eps = eps ,
                        min_samples = min_samples)
        labels = dbscan.fit_predict(X_scaled)

        unique_labels , counts = np.unique(labels , return_counts = True)
        n_noise = 0
        for label , count in zip(unique_labels , counts):
            if label == -1:
                n_noise = count
                print(f"abnormal {count} points")
            else:
                print(f"cu have {count} points")

        min_cluster_size = max(10, len(X_scaled) // 200)  # 小于总数0.5%的簇视为小簇
        small_cluster_labels = [label for label, count in zip(unique_labels, counts) 
                                if label != -1 and count < min_cluster_size]
        
        if small_cluster_labels:
            print(f"\n小簇异常: {len(small_cluster_labels)} 个小簇（<{min_cluster_size}点）被标记为异常")

        is_anomaly = np.zeros(len(df_clean) , dtype = bool)
        if sample_index is not None:
            is_anomaly_sampled = (labels == -1) | (np.isin(labels , small_cluster_labels))
            is_anomaly[sample_index] = is_anomaly_sampled
        else:
            is_anomaly = (labels == -1) | (np.isin(labels , small_cluster_labels))
        abnormal_indices = df_clean.index[is_anomaly]
        
        self.df.loc[abnormal_indices , 'mark'] = -3
        # if 'dbscan_anomaly' not in self.df.columns:
        #     self.df['dbscan_anomaly'] = False
        
        # # 只标记功率>0的行
        # self.df.loc[df_clean.index, 'dbscan_anomaly'] = is_anomaly
        
        # # 合并到mark列（-2表示DBSCAN识别的异常）
        # self.df.loc[self.df['dbscan_anomaly'] == True, 'mark'] = -2
        anomaly_count = is_anomaly.sum()
        print("DBSCAN finished")
        print(f"{anomaly_count} abnormal points")
        print(f"ratio = {anomaly_count/len(df_clean)*100:.2f}%")
        return self.df

    def _find_optimal_eps_for_DBSCAN(self , X_scaled , min_samples , plot_k_distance = True):
        k = min(min_samples , len(X_scaled) - 1)
        neighbor = NearestNeighbors(n_neighbors = k)
        neighbor_fit = neighbor.fit(X_scaled)
        distances , _ = neighbor_fit.kneighbors(X_scaled)

        k_distances = np.sort(distances[:,-1])
        eps = np.percentile(k_distances, 98)  # 或 90
        # diffs = np.diff(k_distances)
        # if len(diffs) > 0:
        #     elbow_index = np.argmax(diffs)
        #     eps = k_distances[elbow_index]
        # else:
        #     eps = k_distances[-1] if len(k_distances) > 0 else 0.5
        return eps
    

    def plot_cleaned_power_curve(self, save_path=None, figsize=(12, 8)):
        """
        绘制清洗后的功率曲线，异常值标红
        
        参数:
            save_path: 保存路径，为None则只显示不保存
            figsize: 图片大小
        """
        # 确保数据存在
        if self.df is None or len(self.df) == 0:
            print("无数据可绘制")
            return
        
        # 创建画布
        fig, ax = plt.subplots(figsize=figsize)
        
        # 分离正常数据和异常数据（根据mark列）
        normal_mask = self.df['mark'] == 0
        range_abnormal_mask = self.df['mark'] == -1
        iqr_abnormal_mask = self.df['mark'] == -2
        dbscan_abnormal_mask = self.df['mark'] == -3
        Kmeans_abnormal_mask = self.df['mark'] == -4
        
        # 绘制正常数据（蓝色）
        ax.scatter(
            self.df.loc[normal_mask, X_COLUMN_NAME],
            self.df.loc[normal_mask, Y_COLUMN_NAME],
            c='green', s=5, alpha=0.5, label='NORMAL'
        )
        
        # 绘制异常数据
        if range_abnormal_mask.any():
            ax.scatter(
                self.df.loc[range_abnormal_mask, X_COLUMN_NAME],
                self.df.loc[range_abnormal_mask, Y_COLUMN_NAME],
                c='red', s=10, alpha=0.7, label='RANGE'
            )
        if iqr_abnormal_mask.any():
            ax.scatter(
                self.df.loc[iqr_abnormal_mask, X_COLUMN_NAME],
                self.df.loc[iqr_abnormal_mask, Y_COLUMN_NAME],
                c='orange', s=10, alpha=0.7, label='IQR'
            )
        if dbscan_abnormal_mask.any():
            ax.scatter(
                self.df.loc[dbscan_abnormal_mask, X_COLUMN_NAME],
                self.df.loc[dbscan_abnormal_mask, Y_COLUMN_NAME],
                c='blue', s=10, alpha=0.7, label='DBSCAN'
            )
        if Kmeans_abnormal_mask.any():
            ax.scatter(
                self.df.loc[Kmeans_abnormal_mask, X_COLUMN_NAME],
                self.df.loc[Kmeans_abnormal_mask, Y_COLUMN_NAME],
                c='black', s=10, alpha=0.7, label='KMeans'
            )
        
        # # 添加额定线
        # ax.axhline(y=self.rated_power, color='green', linestyle='--', 
        #         linewidth=1.5, label=f'额定功率 ({self.rated_power} kW)')
        # ax.axvline(x=self.rated_ws, color='orange', linestyle='--', 
        #         linewidth=1.5, label=f'额定风速 ({self.rated_ws} m/s)')
        
        # 设置标签
        ax.set_xlabel(f'{X_COLUMN_NAME} (m/s)', fontsize=12)
        ax.set_ylabel(f'{Y_COLUMN_NAME} (kW)', fontsize=12)
        ax.set_title(f'{Y_COLUMN_NAME} Curve (Red for Abnormal)', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图片已保存至: {save_path}")
        plt.show()
        return fig

    def plot_only_normal_data(self, save_path=None, figsize=(12, 8)):
        """
        只画 正常数据 的功率曲线
        """
        if self.df is None or len(self.df) == 0:
            print("无数据")
            return

        # 只筛选正常数据（mark == 0）
        normal_data = self.df[self.df['mark'] == 0]

        plt.figure(figsize=figsize)
        plt.scatter(
            normal_data[X_COLUMN_NAME],
            normal_data[Y_COLUMN_NAME],
            c='blue', s=5, alpha=0.6
        )
        plt.xlabel(X_COLUMN_NAME)
        plt.ylabel(Y_COLUMN_NAME)
        plt.title("Cleaned Normal Data Only", fontsize=14)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.show()
        
    def save(self , save_path):
        if self.df is not None:
            self.df.to_csv(save_path , index = False)
            print("save success")
        else:
            print("save fail")
        return


def main():
    filepath = '/work/zfshu/24learn/hyx-test/My_py/光伏/数据/Processed/SSNK/SSNK-2024-Processed.csv'
    rated_ws = 3
    rated_power = 4000
    save_path = '/work/zfshu/24learn/hyx-test/My_py/光伏/数据/Processed/SSNK/anomaly_detection_data_50000.csv'
    df = pd.read_csv(filepath)
    # df.columns = ['time','id','state','power','radiation','ws','wd']
    df = df.sample(n = 50000)
    print(f"df = \n{df}")
    print(f"len(df) = {len(df)}")
    detection = AnomalyDetection(df , rated_ws = rated_ws , rated_power = rated_power)
    # ws = df['ws']
    # active_power = df['active_power']

    # x = ws.values
    # y = active_power.values
    # plt.scatter(
    #     x ,
    #     y ,
    #     s = 8 ,
    #     linewidths = 1 ,
    #     marker = 'o'
    # )
    # plt.show()

    detection.basic_clean()
    detection.edge_clean()
    detection.stack_clean()
    detection.scattered_clean()
    detection.plot_cleaned_power_curve()
    detection.plot_only_normal_data()
    detection.save(save_path)

    return

if __name__ == '__main__':
    main()
