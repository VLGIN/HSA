import random
import logging
from tqdm import tqdm

class HarmonySearch():
    def __init__(self, objective_function, AoI, cell_size, hms=30, hmv=7, hmcr=0.9, par=0.3, BW=0.2, lower=[], upper=[], min_no = 0):
        """
            param explaination
            
            :param hms: harmony memory size, number of vectors stored in harmony memory
            :param hmv: harmony vector size
            :param hmcr: probability for each node considering
            :param par: pitch adjustment rate
            :param BW: distance bandwidth, used for adjust node position when pich adjustment is applied
            :param lower: list contains coordinates for bottom corners
            :param upper: list contains coordinates for upper corners
        """
        self._obj_function = objective_function
        self.hms = hms
        self.hmv = hmv
        self.hmcr = hmcr
        self.par = par
        self.BW = BW
        self.lower = lower
        self.upper = upper
        self.min_no = min_no
        self.AoI = AoI
        self.cell_size = cell_size
        self.best_coverage = 0
        self.logger = logging.getLogger(name='harmony')
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler("output.log")
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger_fitness_run = logging.getLogger(name='best maximum coverage ratio')
        self.logger_fitness_run.setLevel(logging.INFO)
        handler2 = logging.FileHandler('best_maximum_coverage_ratio.log')
        handler2.setLevel(logging.INFO)
        formatter2 = logging.Formatter('%(levelname)s: %(message)s')
        handler2.setFormatter(formatter2)
        self.logger_fitness_run.addHandler(handler2)
        self.logger_fitness_step = logging.getLogger("track fitness step")
        self.logger_fitness_step.setLevel(logging.INFO)
        handler3 = logging.FileHandler("track_fitness_each_step")
        handler3.setLevel(logging.INFO)
        handler3.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        self.logger_fitness_step.addHandler(handler3)
        
    def _random_selection(self, min_valid):
        """
            Randomly generate harmony vector, lenghth = self.hmv
            Step 1: random choice number of deployed nodes
            Step 2: random deploy nodes on area
        """
        # harmony = [[-1,-1]]*self.hmv
        # type_trace = [random.choice(range(len(self.upper))) for i in range(self.hmv)]
        # valid_nodes = random.sample(range(self.hmv), random.randrange(min_valid, self.hmv+1))
        harmony = []
        type_trace = []
        for each_node in range(self.hmv):
            type_ = random.choice([0, 1])
            if type_ == 0:
                x = self.lower[0][0] + (self.upper[0][0] - self.lower[0][0])*random.random()
                y = self.lower[0][1] + (self.upper[0][1] - self.lower[0][1])*random.random()
            else:
                x = self.lower[1][0] + (self.upper[1][0] - self.lower[1][0])*random.random()
                y = self.lower[1][1] + (self.upper[1][1] - self.lower[1][1])*random.random()
            # harmony[each_node] = [x, y]
            # type_trace[each_node] = type_
            harmony.append([x, y])
            type_trace.append(type_)
        return harmony, type_trace
    
    def _centroid_selection(self, min_valid):
        num_width_cell = self.AoI[0] // self.cell_size[0]
        num_height_cell = self.AoI[1] // self.cell_size[1]
        # harmony = [[-1, -1]] * self.hmv
        # type_trace = [random.choice(range(len(self.upper))) for i in range(self.hmv)]
        # valid_nodes = random.sample(range(self.hmv), random.randrange(min_valid, num_width_cell*num_height_cell + 1))
        # id_valid_cell = random.sample(range(num_width_cell*num_height_cell), len(valid_nodes))
        id_valid_cell = list(range(self.hmv))
        random.shuffle(id_valid_cell)
        harmony = []
        type_trace = []
        for ids in range(self.hmv):
            type_ = random.choice([0,1])
            width_coor = id_valid_cell[ids] % num_width_cell
            height_coor = id_valid_cell[ids] // num_width_cell
            x = width_coor * self.cell_size[0] + self.cell_size[0] / 2
            y = height_coor * self.cell_size[1] + self.cell_size[1] / 2
            # harmony[each_node] = [x, y]
            # type_trace[each_node] = type_
            harmony.append([x, y])
            type_trace.append(type_)
        return harmony, type_trace
        
    def _cell_selection(self, min_valid):
        num_width_cell = self.AoI[0] // self.cell_size[0]
        num_height_cell = self.AoI[1] // self.cell_size[1]
        # harmony = [[-1, -1]] * self.hmv
        # type_trace = [random.choice(range(len(self.upper))) for i in range(self.hmv)]
        # valid_nodes = random.sample(range(self.hmv), random.randrange(min_valid, num_width_cell*num_height_cell + 1))
        # id_valid_cell = random.sample(range(num_width_cell*num_height_cell), len(valid_nodes))
        id_valid_cell = list(range(self.hmv))
        random.shuffle(id_valid_cell)
        harmony = []
        type_trace = []
        for ids in range(self.hmv):
            type_ = random.choice([0,1])
            width_coor = id_valid_cell[ids] % num_width_cell
            height_coor = id_valid_cell[ids] // num_width_cell
            x = width_coor * self.cell_size[0] + self.cell_size[0]*random.random()
            y = height_coor * self.cell_size[1] + self.cell_size[1]*random.random()
            # harmony[each_node] = [x, y]
            # type_trace[each_node] = type_
            harmony.append([x, y])
            type_trace.append(type_)
        return harmony, type_trace

    def _initialize_harmony(self, type = "default", min_valid=14, initial_harmonies=None):
        """
            Initialize harmony_memory, the matrix containing solution vectors (harmonies)
        """
        if initial_harmonies is not None:
            for each_harmony, type_trace in initial_harmonies:
                self._harmony_memory.append((each_harmony, self._obj_function.get_fitness(each_harmony, type_trace)[0]))
        else:
            assert type in ["default", "centroid", "cell"], "Unknown type of initialization"
            self._harmony_memory = []
            if type == "default":
                for _ in range(0, self._obj_function.get_hms()):
                    harmony, type_trace = self._random_selection(min_valid)
                    self._harmony_memory.append((harmony,type_trace, self._obj_function.get_fitness((harmony, type_trace))[0]))
            elif type == "centroid":
                for _ in range(0, self._obj_function.get_hms()):
                    harmony, type_trace = self._centroid_selection(min_valid)
                    self._harmony_memory.append((harmony,type_trace, self._obj_function.get_fitness((harmony, type_trace))[0]))
            elif type == "cell":
                for _ in range(0, self._obj_function.get_hms()):
                    harmony, type_trace = self._cell_selection(min_valid)
                    self._harmony_memory.append((harmony,type_trace, self._obj_function.get_fitness((harmony, type_trace))[0]))

    def _memory_consideration(self):
        """
            Generate new harmony from previous harmonies in harmony memory
            Apply pitch adjustment with par probability
        """
        harmony = []
        type_trace = []
        for i in range(self.hmv):
            p_hmcr = random.random()
            if p_hmcr < self.hmcr:
                id = random.choice(range(self.hms))
                [x, y] = self._harmony_memory[id][0][i]
                [x, y] = self._pitch_adjustment([x, y])
            else:
                type_ = random.choice([0, 1])
                if type_ == 0:
                    x = self.lower[0][0] + (self.upper[0][0] - self.lower[0][0])*random.random()
                    y = self.lower[0][1] + (self.upper[0][1] - self.lower[0][1])*random.random()
                else:
                    x = self.lower[1][0] + (self.upper[1][0] - self.lower[1][0])*random.random()
                    y = self.lower[1][1] + (self.upper[1][1] - self.lower[1][1])*random.random()
            type_ = random.choice(range(len(self.upper)))
            type_trace.append(type_)
            if x > self.upper[type_][0] or x < self.lower[type_][0]:
                x = -1
            if y > self.upper[type_][1] or y < self.lower[type_][1]:
                y = -1
            harmony.append([x, y])
        return harmony, type_trace

    def _pitch_adjustment(self, position):
        """
            Adjustment for generating completely new harmony vectors
        """
        p_par = random.random()
        if p_par < self.par:
            bw_rate = random.uniform(-1,1)
            position[0] = self.BW*bw_rate + position[0]
            position[1] = self.BW*bw_rate + position[1]
        return position

    def _new_harmony_consideration(self, harmony, type_trace):
        """
            Update harmony memory
        """
        fitness = self._obj_function.get_fitness((harmony, type_trace))[0]
        worst_fitness = float("+inf")
        worst_ind = -1
        for ind, (_, _x, each_fitness) in enumerate(self._harmony_memory):
            if each_fitness < worst_fitness:
                worst_fitness = each_fitness
                worst_ind = ind
        if fitness >= worst_fitness:
            self._harmony_memory[worst_ind] = (harmony, type_trace, fitness)

    def _get_best_fitness(self):
        """
            Gest best fitness and corresponding harmony vector in harmony memory
        """
        best_fitness = float("-inf")
        best_harmony = []
        for each_harmony, type_trace, each_fitness in self._harmony_memory:
            if each_fitness > best_fitness:
                best_fitness = each_fitness
                best_harmony = each_harmony
                type_ = type_trace
        return best_harmony, type_, best_fitness

    def _get_best_coverage_ratio(self):
        best_harmony, type_trace, best_fitness = self._get_best_fitness()
        coverage_ratio = self._obj_function.get_fitness((best_harmony, type_trace))[1]
        return coverage_ratio, best_harmony, type_trace, best_fitness

    def _evaluation(self, threshold, i):
        coverage_ratio, best_harmony, type_trace, best_fitness = self._get_best_coverage_ratio()
        final_harmony = []
        final_type = []
        for ind, sensor in enumerate(best_harmony):
            if sensor[0]>=0 and sensor[1]>=0:
                final_harmony.append(sensor)
                final_type.append(type_trace[ind])
        self.logger_fitness_step.info("""Step {}:\nHarmony: {}\nType: {}\nFitness: {}\n
                                Coverage: {}\n Number of sensors: {}\n""".format(i, final_harmony, final_type, best_fitness,
                                coverage_ratio, len(final_harmony)))
        self.logger_fitness_step.info("-------------------------------------------------")
        if coverage_ratio >= threshold:
            self.logger.info("""Harmony: {}\nType: {}\nFitness: {}\n
                                Coverage: {}\n Number of sensors: {}\n""".format(final_harmony, final_type, best_fitness,
                                coverage_ratio, len(final_harmony)))
            self.logger.info("-------------------------------------------------")

    def _count_sensor(self, harmony):
        count_ = 0
        for item in harmony:
            if item[0] >= 0 and item[1] >= 0:
                count_ += 1
        return count_

    def run(self, type_init="default", min_valid=14,steps=100, threshold=0.9,order=0):
        print("Start run:")
        self._initialize_harmony(type_init, min_valid)
        self.logger_fitness_step.info("Run {}\n".format(order))
        self.logger.info("Run {}\n".format(order))
        for i in tqdm(range(steps)):
            new_harmony, type_trace = self._memory_consideration()
            self._new_harmony_consideration(new_harmony, type_trace)
            best_harmony, type_, best_fitness = self._get_best_fitness()
            self._evaluation(threshold, i)
        self.logger_fitness_step.info("******************************************************")
        self.logger.info("**************************************************")

        best_harmony, type_, best_fitness = self._get_best_fitness()
        used_node = []
        type_trace = []
        for ind, node in enumerate(best_harmony):
            if node[0] > 0 and node[1] > 0:
                used_node.append(node)
                type_trace.append(type_[ind])
        coverage = self._obj_function.get_coverage_ratio(used_node, type_trace)
        self.logger_fitness_run.info(f'Best harmony: {str(best_harmony)}\nType: {str(type_)}\nBest_fitness: {str(best_fitness)}\nCoressponding coverage: {str(coverage)}')
        self.logger_fitness_run.info('------------------------------------------------------------------------------------')

    def test(self, type_init="default", min_valid=14, steps=100, threshold=0.9, file='logging.txt', num_run=12):
        for i in range(num_run):
            self.run(type_init, min_valid, steps, threshold, i)
