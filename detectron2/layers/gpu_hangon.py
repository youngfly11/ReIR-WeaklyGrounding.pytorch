#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/6/25 12:07



from gpustat.core import GPUStatCollection

def get_gpu_status():

    gpus_stats = GPUStatCollection.new_query()
    info = gpus_stats.jsonify()["gpus"]
    gpu_list = []

    mem_ratio_threshold = 0.1  #
    util_ratio_threshold = 10  #
    for idx, each in enumerate(info):
        mem_ratio = each["memory.used"] / each["memory.total"]
        util_ratio = each["utilization.gpu"]
        print(mem_ratio, util_ratio)
        if mem_ratio < mem_ratio_threshold and util_ratio < util_ratio_threshold:
            gpu_list.append(idx)
    print("Scan GPUs to get {} free GPU".format(len(gpu_list)))
    return gpu_list


if __name__ == '__main__':

    gpu_list = get_gpu_status()
    print(gpu_list)


