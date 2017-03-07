import os
import numpy as np
import matplotlib.pyplot as plt

source_path  = [
    "/Users/ruru/Desktop/Flux Preservation/flux_10",
    "/Users/ruru/Desktop/Flux Preservation/flux_100",
    "/Users/ruru/Desktop/Flux Preservation/flux_1000",
    "/Users/ruru/Desktop/Flux Preservation/flux_10000",
]

xlim_min, xlim_max = 1e10, -1e10
ylim_min, ylim_max = 1e10, -1e10

def load_data():
    global xlim_min, xlim_max, ylim_min, ylim_max
    data = []
    for path in source_path:
        filename = os.path.basename(path)
        points = []
        f = open(path)
        for line in f:
            l = line.strip().split("\t")
            x = float(l[2])
            y = float(l[4])
            xlim_min = min(xlim_min, x)
            xlim_max = max(xlim_max, x)
            ylim_min = min(ylim_min, y)
            ylim_max = max(ylim_max, y)
            points.append((x,y))
        f.close()
        data.append((filename, points))
    return data

def main():
    data_num = len(source_path)
    data = load_data()
    num = 0
    n = 50
    for name, points in data:
        num += 1
        P = plt.subplot(2,2,num)
        X = [d[0] for d in points]
        Y = [d[1] for d in points]
        P.set_title(name)
        P.set_xlabel("PSNR")
        P.set_ylabel("FLUX(sum)")
        P.scatter(X,Y,color='r',s = 4,marker='s')
        plt.xlim(xlim_min-1, xlim_max+1)
        plt.ylim(0, 100)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
