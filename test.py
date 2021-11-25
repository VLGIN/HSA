import argparse
import random
from harmony_search import HarmonySearch
from objective_function import ObjectiveFunction

def train(w, h, types, radius, hms, cellw, cellh, hcmr, par, bw):
    min_noS = w * h // ((max(radius)**2)*9)
    max_noS = w * h // ((min(radius)**2))
    print(min_noS, max_noS)
    hmv = random.randint(min_noS, max_noS)
    print("HMV: ", hmv)
    targets = []
    init_x = cellw / 2
    init_y = cellh / 2
    while init_x < w:
        while init_y < h:
            targets.append([init_x, init_y])
            init_y += cellh
        init_x += cellw
    obj_func = ObjectiveFunction(hmv, hms, targets, types=2, radius=radius, w=w, h=h, cell_h=cellh, cell_w=cellw)
    min_noS = w * h // ((max(radius)**2)*9)
    hsa = HarmonySearch(objective_function=obj_func, hms=hms, hmv=hmv, hmcr=hcmr, par=par,\
                        BW=bw, lower=[[radius[0]/2, radius[0]/2], [radius[1]/2, radius[1]/2]],\
                        upper=[[w-radius[0]/2, h-radius[0]/2], [w-radius[1]/2, h-radius[1]/2]], min_no=min_noS)
    hsa.run(10)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameter for training")

    parser.add_argument("--W", default=50, type=int)
    parser.add_argument("--H", default=50, type=int)
    parser.add_argument("--types", default=2, type=int)
    parser.add_argument("--radius", nargs="+")
    parser.add_argument("--hms", default=10, type=int)
    parser.add_argument("--cellw", default=10, type=int)
    parser.add_argument("--cellh", default=10, type=int)
    parser.add_argument("--hcmr", default=0.9, type=float)
    parser.add_argument("--par", default=0.3, type=float)
    parser.add_argument("--bw", default=0.2, type=float)

    args = parser.parse_args()
    radius = []
    for i in args.radius:
        radius.append(int(i))
    train(int(args.W), int(args.H), args.types, radius, args.hms, args.cellw, args.cellh, args.hcmr, args.par, args.bw)
