import numpy as np
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings('ignore')

class ModelLatency():
    def __init__(self) -> None:
        pass
        

    def cal_latency(self, x, a, b, c):
        return a * np.exp(-b * x) + c


    def vgg_19_latency(self, input_pct):
        GPU_pct_vgg_19 = np.array([5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        latency_vgg_19 = np.array([400, 90, 68, 56, 46, 38, 31, 27, 24, 22, 20])

        popt_vgg, pcov_vgg = curve_fit(self.cal_latency, GPU_pct_vgg_19, latency_vgg_19)
        
        return round(self.cal_latency(input_pct, *popt_vgg), 2)

    def resnet_50_latency(self, input_pct):
        GPU_pct_resnet_50 = np.array([5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        latency_resnet_50 = np.array([50, 30, 25, 22, 19, 18, 17, 16, 16, 15, 15])

        popt_res, pcov_res = curve_fit(self.cal_latency, GPU_pct_resnet_50, latency_resnet_50)

        return round(self.cal_latency(input_pct, *popt_res), 2)

    def alexnet_latency(self, input_pct):
        GPU_pct_alexnet = np.array([5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        latency_alexnet = np.array([15, 10, 8, 6, 5, 4, 3, 3, 2, 2, 2])

        popt_alex, pcov_alex = curve_fit(self.cal_latency, GPU_pct_alexnet, latency_alexnet)

        return round(self.cal_latency(input_pct, *popt_alex), 2)

test = ModelLatency()
print(f'vgg latency = {test.alexnet_latency(30)}')