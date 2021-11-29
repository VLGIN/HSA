import math
from itertools import product


class ObjectiveFunction():
    def __init__(self, hmv, hms, targets, types=2, radius=[0,0], alpha1=1, alpha2=0, beta1=1, beta2=0.5,\
                threshold=0.9, w=50, h=50, cell_h=10, cell_w=10):
        """
            :param hmv: harmony vector size
            :param hms: harmony memory size
            :param targets: position of target points
            :param types: number of different node types
            :param radius: radius for each node type
            :param alpha1, alpha2, beta1, beta2: parameter for calculating Pov
            :param threshold: threshold for Pov
            :param h, w: height and width of AoI
            :param cell_h, cell_w: height and width of cell
        """
        self.hmv = hmv
        self.hms = hms
        self.targets = targets
        self.radius = radius
        self.ue = []
        for r in radius:
            self.ue.append(r / 2)
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta1 = beta1
        self.beta2 = beta2
        self.threshold = threshold
        self.type_sensor = range(types)
        self.w = w
        self.h = h
        self.cell_h = cell_h
        self.cell_w = cell_w
        self.no_cell = (self.w // self.cell_w) * (self.h // self.cell_h)
        self.min_noS = self.w * self.h // ((max(self.radius)**2)*9)
        self.max_noS = self.w * self.h // ((min(self.radius)**2))
        self.max_diagonal = max([self._distance([self.w, self.h], [self.radius[i] - self.ue[i], self.radius[i] - self.ue[i]]) for i in range(len(self.radius))])

    def get_num_parameters(self):
        return self.hmv

    def get_hms(self):
        return self.hms

    def _senscost(self, node_list):
        return (self.max_noS + 1 - self.min_noS) / (len(node_list) + 1 - self.min_noS)
    
    def _coverage_ratio(self, node_list, type_assignment):
        """
            Return coverage_ratio and list of covered target
        """
        target_corvered = []
        for target in self.targets:
            Pov = 1
            count = 0
            for index, sensor in enumerate(node_list):
                p = self._psm(sensor, target, type=type_assignment[index])
                if p == 0:
                    continue
                count += 1
                Pov *= p
            
            Pov = 1 - Pov
            if count == 1 and Pov == 0:
                target_corvered.append(target)
            elif Pov >= self.threshold:
                target_corvered.append(target)
        return len(target_corvered) / self.no_cell, target_corvered

    def _md(self, node_list, type_assignment):
        min_dist_sensor = float('+inf')
        for ia, a in enumerate(node_list):
            for ib, b in enumerate(node_list):
                if a != b:
                    min_dist_sensor = min(min_dist_sensor, self._distance(a, b) * (self.radius[type_assignment[ia]]) * (self.radius[type_assignment[ib]]))
        if min_dist_sensor == float('+inf'):
            min_dist_sensor = 0.0
        return min_dist_sensor / self.max_diagonal

    def get_fitness(self, harmony):
        used = []
        for sensor in harmony:
            if sensor[0] < 0 or sensor[1] < 0:
                continue
            else:
                used.append(sensor)

        if len(used) < self.min_noS:
            return float('-inf'), 0, []
        
        best_sol = float('-inf')
        type_trace = []
        coverage_ratio_ = 0
        for case in product(self.type_sensor, repeat=len(used)):
            coverage_ratio, _ = self._coverage_ratio(used, case)
            fitness = self._senscost(used) * coverage_ratio * self._md(used, case)
            if fitness > best_sol:
                coverage_ratio_ = coverage_ratio
                best_sol = fitness
                type_trace = case
        
        return best_sol, coverage_ratio_, type_trace

    def _distance(self, x1, x2):
        return math.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)

    def _psm(self,x, y, type):
        distance = self._distance(x, y)
        
        if distance < self.radius[type] - self.ue[type]:
            return 1
        elif distance > self.radius[type] + self.ue[type]:
            return 0
        else:
            lambda1 = self.ue[type] - self.radius[type] + distance
            lambda1 = self.ue[type] - self.radius[type] + distance
            lambda2 = self.ue[type] + self.radius[type] - distance
            lambda1 = math.pow(lambda1, self.beta1)
            lambda2 = math.pow(lambda2, self.beta2)
            return math.exp(-(self.alpha1*lambda1/lambda2 + self.alpha2))
