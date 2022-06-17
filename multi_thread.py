# %%
import numpy as np
import random
import plotly.graph_objs as go
from plotly import offline
from multiprocessing import Pool, Process, JoinableQueue
import time

# %%
publicity_rate_to_single_pull_rate = 12.395 / 4.877

focus_regular_weapon_rate = 0.02479 / publicity_rate_to_single_pull_rate
focus_regular_stigmata_rate = 0.01240 / publicity_rate_to_single_pull_rate
focus_regular_4_star_exclude_up_rate = 0.12395 / publicity_rate_to_single_pull_rate - focus_regular_weapon_rate - focus_regular_stigmata_rate * 3.0
expansion_regular_weapon_rate = 0.01831 / publicity_rate_to_single_pull_rate
expansion_regular_stigmata_rate = 0.00916 / publicity_rate_to_single_pull_rate
expansion_regular_4_star_exclude_up_rate = 0.12395 / publicity_rate_to_single_pull_rate - expansion_regular_weapon_rate - expansion_regular_stigmata_rate * 3.0
new_regular_weapon_rate = 0.02479 / publicity_rate_to_single_pull_rate
new_regular_stigmata_rate = 0.01240 / publicity_rate_to_single_pull_rate
new_regular_4_star_exclude_up_rate = 0.12395 / publicity_rate_to_single_pull_rate - new_regular_weapon_rate - new_regular_stigmata_rate * 3.0

guarantee_weapon_rate = 0.2
guarantee_stigmata_rate = 0.1
guarantee_4_star_exclude_up_rate = 1 - guarantee_weapon_rate - guarantee_stigmata_rate * 3
guarantee_small = 10
guarantee_big = 50

sample_accuracy = int(1e4)
banner_size = int(1e6)
pnt_threshold = banner_size - int(2e3)
gacha_sample_size = int(1e8)
x_truncate = 200

thread_pool_size = int(10)
thread_number = int(10)
preserved_size = 1500

guarantee_banner, focus_regular_banner, expansion_regular_banner, new_regular_banner = {}, {}, {}, {}


# %%
def generate_banner():
    pnt, val = 0, -1
    for i in range(sample_accuracy):
        guarantee_banner.update({i: -2})
        focus_regular_banner.update({i: -2})
        expansion_regular_banner.update({i: -2})
        new_regular_banner.update({i: -2})

    for i in np.array(
            [
                guarantee_4_star_exclude_up_rate,
                guarantee_weapon_rate,
                guarantee_stigmata_rate,
                guarantee_stigmata_rate,
                guarantee_stigmata_rate
            ]) * sample_accuracy:
        for j in range(int(i)):
            guarantee_banner[pnt] = val
            pnt += 1
        val += 1

    pnt, val = 0, -1
    for i in np.array(
            [
                focus_regular_4_star_exclude_up_rate,
                focus_regular_weapon_rate,
                focus_regular_stigmata_rate,
                focus_regular_stigmata_rate,
                focus_regular_stigmata_rate
            ]) * sample_accuracy:
        for j in range(int(i)):
            focus_regular_banner[pnt] = val
            pnt += 1
        val += 1

    pnt, val = 0, -1
    for i in np.array(
            [
                expansion_regular_4_star_exclude_up_rate,
                expansion_regular_weapon_rate,
                expansion_regular_stigmata_rate,
                expansion_regular_stigmata_rate,
                expansion_regular_stigmata_rate
            ]) * sample_accuracy:
        for j in range(int(i)):
            expansion_regular_banner[pnt] = val
            pnt += 1
        val += 1

    pnt, val = 0, -1
    for i in np.array(
            [
                new_regular_4_star_exclude_up_rate,
                new_regular_weapon_rate,
                new_regular_stigmata_rate,
                new_regular_stigmata_rate,
                new_regular_stigmata_rate
            ]) * sample_accuracy:
        for j in range(int(i)):
            new_regular_banner[pnt] = val
            pnt += 1
        val += 1


generate_banner()


# %%
class Gacha_Focus:
    def __init__(self, args):
        self.args = args
        self.pity_counter_small = 0
        self.pity_counter_big = 0
        self.equipment = np.array([0, 0, 0, 0])
        self.total_pulling = 0

    def single_pull(self):
        result = guarantee_banner[self.args['random_pool'][self.args['pnt_random_pool']]] \
            if self.pity_counter_small >= guarantee_small \
            else focus_regular_banner[self.args['random_pool'][self.args['pnt_random_pool']]]
        self.args['pnt_random_pool'] += 1

        if result >= -1:
            self.pity_counter_small = 0

            if result >= 0:
                self.equipment[result] += 1
                return True

        return False

    def reset_pity_counter(self):
        return True if \
            (0 not in self.equipment or
             (self.equipment[0] and np.count_nonzero(self.equipment) == 3 and
              (self.equipment[1] + self.equipment[2] + self.equipment[3]) >= 4)) \
            else False

    def pull(self):
        while True:
            self.pity_counter_small += 1
            self.total_pulling += 1

            if self.single_pull() and self.reset_pity_counter():
                break

        return [self.total_pulling, self.equipment]


# %%
class Gacha_Expansion:
    def __init__(self, args):
        self.args = args
        self.pity_counter_small = 0
        self.pity_counter_big = 0
        self.equipment = np.array([0, 0, 0, 0])
        self.total_pulling = 0

    def single_pull(self):
        if self.pity_counter_big >= guarantee_big:
            self.pity_counter_small = 0
            self.pity_counter_big = 0
            self.equipment[random.sample(list(np.where(self.equipment == 0)[0]), 1)[0]] += 1

            return True
        else:
            result = guarantee_banner[self.args['random_pool'][self.args['pnt_random_pool']]] \
                if self.pity_counter_small >= guarantee_small \
                else expansion_regular_banner[self.args['random_pool'][self.args['pnt_random_pool']]]
            self.args['pnt_random_pool'] += 1

            if result >= -1:
                self.pity_counter_small = 0

                if result >= 0:
                    if self.equipment[result] == 0:
                        self.pity_counter_big = 0

                    self.equipment[result] += 1
                    return True

            return False

    def reset_pity_counter(self):
        return True if \
            (0 not in self.equipment or
             (self.equipment[0] and np.count_nonzero(self.equipment) == 3 and
              (self.equipment[1] + self.equipment[2] + self.equipment[3]) >= 4)) \
            else False

    def pull(self):
        while True:
            self.pity_counter_small += 1
            self.pity_counter_big += 1
            self.total_pulling += 1

            if self.single_pull() and self.reset_pity_counter():
                break

        return [self.total_pulling, self.equipment]


# %%
class Gacha_New:
    def __init__(self, args):
        self.args = args
        self.pity_counter_small = 0
        self.pity_counter_big = 0
        self.equipment = np.array([0, 0, 0, 0])
        self.total_pulling = 0

    def single_pull(self):
        if self.pity_counter_big >= guarantee_big:
            self.pity_counter_small = 0
            self.pity_counter_big = 0
            self.equipment[random.sample(list(np.where(self.equipment == 0)[0]), 1)[0]] += 1

            return True
        else:
            result = guarantee_banner[self.args['random_pool'][self.args['pnt_random_pool']]] \
                if self.pity_counter_small >= guarantee_small \
                else new_regular_banner[self.args['random_pool'][self.args['pnt_random_pool']]]
            self.args['pnt_random_pool'] += 1

            if result >= -1:
                self.pity_counter_small = 0

                if result >= 0:
                    if self.equipment[result] == 0:
                        self.pity_counter_big = 0

                    self.equipment[result] += 1
                    return True

            return False

    def reset_pity_counter(self):
        return True if \
            (0 not in self.equipment or
             (self.equipment[0] and np.count_nonzero(self.equipment) == 3 and
              (self.equipment[1] + self.equipment[2] + self.equipment[3]) >= 4)) \
            else False

    def pull(self):
        while True:
            self.pity_counter_small += 1
            self.pity_counter_big += 1
            self.total_pulling += 1

            if self.single_pull() and self.reset_pity_counter():
                break

        return [self.total_pulling, self.equipment]


# %%
def test():
    _y_focus_temp, _y_expansion_temp, _y_new_temp = np.array([0] * preserved_size), np.array([0] * preserved_size), np.array([0] * preserved_size)

    args = {
        'random_pool': np.random.randint(0, sample_accuracy, banner_size),
        'pnt_random_pool': 0
    }

    for i in range(gacha_sample_size):
        result_focus = Gacha_Focus(args).pull()[0]
        result_expansion = Gacha_Expansion(args).pull()[0]
        result_new = Gacha_New(args).pull()[0]

        _y_focus_temp[result_focus] += 1
        _y_expansion_temp[result_expansion] += 1
        _y_new_temp[result_new] += 1

        if args['pnt_random_pool'] >= pnt_threshold:
            np.random.shuffle(args['random_pool'])
            args['pnt_random_pool'] = 0

    return [_y_focus_temp, _y_expansion_temp, _y_new_temp]


if __name__ == '__main__':
    start_time = time.time()

    processing_pool = Pool(processes=thread_pool_size)
    gacha_results = []

    for j in range(thread_number):
        gacha_results.append([
            j,
            processing_pool.apply_async(test)
        ])

    y_focus, y_expansion, y_new = np.array([0] * preserved_size), np.array([0] * preserved_size), np.array([0] * preserved_size)
    for gacha_result in gacha_results:
        [y_focus_temp, y_expansion_temp, y_new_temp] = gacha_result[1].get()
        y_focus += y_focus_temp
        y_expansion += y_expansion_temp
        y_new += y_new_temp
        print('第{}个进程计算完毕'.format(gacha_result[0] + 1))

    processing_pool.close()
    print('总共耗时{}s'.format(int(time.time() - start_time)))

    def draw_result():
        data = [
            go.Scatter(
                x=[i for i in range(x_truncate)],
                y=y_focus[:x_truncate],
                mode='markers',
                marker={
                    'color': 'rgb(255, 0, 0)'
                },
                name='focus'
            ),
            go.Scatter(
                x=[i for i in range(x_truncate)],
                y=y_expansion[:x_truncate],
                mode='markers',
                marker={
                    'color': 'rgb(0, 0, 255)'
                },
                name='expansion'
            ),
            go.Scatter(
                x=[i for i in range(x_truncate)],
                y=y_new[:x_truncate],
                mode='markers',
                marker={
                    'color': 'rgb(0, 255, 0)'
                },
                name='new'
            ),
        ]
        fig = go.Figure(
            data=data
        )
        # fig.show()
        offline.plot(fig)

    draw_result()

    def generate_statistics():
        print('精准补给平均数: {}'.format(np.average(np.array(range(len(y_focus))), weights=y_focus)))
        print('扩充补给平均数: {}'.format(np.average(np.array(range(len(y_expansion))), weights=y_expansion)))
        print('新补给池平均数: {}'.format(np.average(np.array(range(len(y_new))), weights=y_new)))

        temp_num_focus, temp_num_expansion, temp_num_new = sum(y_focus) / 2.0, sum(y_expansion) / 2.0, sum(y_new) / 2.0

        for i in range(x_truncate):
            if temp_num_focus >= y_focus[i]:
                temp_num_focus -= y_focus[i]
            else:
                print('精准补给中位数: {}'.format(i))
                break

        for i in range(x_truncate):
            if temp_num_expansion >= y_expansion[i]:
                temp_num_expansion -= y_expansion[i]
            else:
                print('扩充补给中位数: {}'.format(i))
                break

        for i in range(x_truncate):
            if temp_num_new >= y_new[i]:
                temp_num_new -= y_new[i]
            else:
                print('新补给池中位数: {}'.format(i))
                break

    generate_statistics()
