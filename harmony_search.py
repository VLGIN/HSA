import random


class HarmonySearch():
    def __init__(self, objective_function, hms=30, hmv=7, hmcr=0.9, par=0.3, BW=0.2, lower=[], upper=[], min_no = 0):
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

    def _random_selection(self):
        """
            Randomly generate harmony vector, lenghth = self.hmv
            Step 1: random choice number of deployed nodes
            Step 2: random deploy nodes on area
        """
        harmony = [[-1,-1]]*self.hmv
        type_trace = [random.choice(range(len(self.upper))) for i in range(self.hmv)]
        valid_nodes = random.sample(range(self.hmv), random.randrange(self.min_no, self.hmv+1))
        for each_node in valid_nodes:
            type_ = random.choice([0, 1])
            if type_ == 0:
                x = self.lower[0][0] + (self.upper[0][0] - self.lower[0][0])*random.random()
                y = self.lower[0][1] + (self.upper[0][1] - self.lower[0][1])*random.random()
            else:
                x = self.lower[1][0] + (self.upper[1][0] - self.lower[1][0])*random.random()
                y = self.lower[1][1] + (self.upper[1][1] - self.lower[1][1])*random.random()
            harmony[each_node] = [x, y]
            type_trace[each_node] = type_
        return harmony, type_trace

    def _initialize_harmony(self, initial_harmonies=None):
        """
            Initialize harmony_memory, the matrix containing solution vectors (harmonies)
        """
        if initial_harmonies is not None:
            # assert len(initial_harmonies) == self._obj_function.get_hms(),\
            #     "Size of harmony memory and objective function is not compatible"
            # assert len(initial_harmonies[0]) == self._obj_function.get_num_parameters(),\
            #     "Number of params in harmony memory and objective function is not compatible"
            for each_harmony, type_trace in initial_harmonies:
                self._harmony_memory.append((each_harmony, self._obj_function.get_fitness(each_harmony, type_trace)[0]))
        else:
            self._harmony_memory = []
            for _ in range(0, self._obj_function.get_hms()):
                harmony, type_trace = self._random_selection()
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
            else:
                type_ = random.choice([0, 1])
                if type_ == 0:
                    x = self.lower[0][0] + (self.upper[0][0] - self.lower[0][0])*random.random()
                    y = self.lower[0][1] + (self.upper[0][1] - self.lower[0][1])*random.random()
                else:
                    x = self.lower[1][0] + (self.upper[1][0] - self.lower[1][0])*random.random()
                    y = self.lower[1][1] + (self.upper[1][1] - self.lower[1][1])*random.random()
            type_trace.append(random.choice(range(len(self.upper))))
            harmony.append(self._pitch_adjustment([x, y]))
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
        best_harmony, type_trace = self._get_best_fitness()[0:2]
        coverage_ratio = self._obj_function.get_fitness((best_harmony, type_trace))[1]
        return coverage_ratio

    def _evaluation(self, threshold):
        coverage_ratio = self._get_best_coverage_ratio()
        print("Coverage Ratio: ", coverage_ratio)
        if coverage_ratio > threshold:
            return True
        return False

    def _count_sensor(self, harmony):
        count_ = 0
        for item in harmony:
            if item[0] >= 0 and item[1] >= 0:
                count_ += 1
        return count_

    def run(self, steps=100, threshold=0.9, file="result.txt"):
        self._initialize_harmony()
        pass_harmony = []

        for i in range(steps):
            new_harmony, type_trace = self._memory_consideration()
            self._new_harmony_consideration(new_harmony, type_trace)
            best_harmony, type_, best_fitness = self._get_best_fitness()
            count_ = self._count_sensor(best_harmony)
            coverage = self._get_best_coverage_ratio()
            print("Generation: {}, best fitness: {}, use {} sensors".format(i+1, best_fitness, count_))
            # define some criteria for stopping
            if self._evaluation(threshold):
                print("Criteria is sastified\n-------------------------------------------------")
                pass_harmony.append((best_harmony, type_, coverage, count_))
        with open(file, "w") as f:
            for item in pass_harmony:
                f.write("Best Harmony " + str(item[0]) + "\nWith type: " + str(item[1]) + "\ncoverage: "+str(item[2])+"\tuse "+str(item[3]) +" sensors\n")

        return best_harmony, best_fitness
