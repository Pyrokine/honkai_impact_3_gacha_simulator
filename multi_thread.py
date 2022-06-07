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

guarantee_weapon_rate = 0.2
guarantee_stigmata_rate = 0.1
guarantee_4_star_exclude_up_rate = 1 - guarantee_weapon_rate - guarantee_stigmata_rate * 3
guarantee_small = 10
guarantee_big = 50

# banner_sample_size 为浮点随机数向整数映射时整数的范围，该值越大则保留小数点越多，但耗时则越长
# gacha_sample_size * thread_pool_size 和运行时占用内存正相关，两者乘积在 1e7 时会占用 11.5G 的内存，可以根据实际情况调整
# gacha_sample_size * thread_number 为总共进行的采样次数
banner_sample_size = int(1e4)
gacha_sample_size = int(1e7)
thread_pool_size = int(3)
thread_number = int(12)

# -2 非四星装备  -1 四星非UP装备  0 UP武器  1 UP圣痕上  2 UP圣痕中  3 UP圣痕下
banner_guarantee, banner_regular_focus, banner_regular_expansion = {}, {}, {}


# %%
def generate_banner():
    pnt, val = 0, -1
    for i in range(banner_sample_size):
        banner_guarantee.update({i: -2})
        banner_regular_focus.update({i: -2})
        banner_regular_expansion.update({i: -2})

    for i in np.array(
            [
                guarantee_4_star_exclude_up_rate,
                guarantee_weapon_rate,
                guarantee_stigmata_rate,
                guarantee_stigmata_rate,
                guarantee_stigmata_rate
            ]) * banner_sample_size:
        for j in range(int(i)):
            banner_guarantee[pnt] = val
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
            ]) * banner_sample_size:
        for j in range(int(i)):
            banner_regular_focus[pnt] = val
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
            ]) * banner_sample_size:
        for j in range(int(i)):
            banner_regular_expansion[pnt] = val
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
        result = banner_guarantee[self.args['random_pool'][self.args['pnt_random_pool']]] \
            if self.pity_counter_small >= guarantee_small \
            else banner_regular_focus[self.args['random_pool'][self.args['pnt_random_pool']]]
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
            result = banner_guarantee[self.args['random_pool'][self.args['pnt_random_pool']]] \
                if self.pity_counter_small >= guarantee_small \
                else banner_regular_expansion[self.args['random_pool'][self.args['pnt_random_pool']]]
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
x_truncate = 200
y_focus_truncate, y_expansion_truncate = [0] * x_truncate, [0] * x_truncate


def test():
    y_focus_temp, y_expansion_temp = {}, {}

    args = {
        'random_pool': np.random.randint(0, banner_sample_size, gacha_sample_size * 300),
        'pnt_random_pool': 0
    }

    for i in range(gacha_sample_size):
        result_focus = Gacha_Focus(args).pull()[0]
        result_expansion = Gacha_Expansion(args).pull()[0]
        if result_focus < x_truncate:
            if result_focus not in y_focus_temp:
                y_focus_temp.update({result_focus: 1})
            else:
                y_focus_temp[result_focus] += 1
        if result_expansion < x_truncate:
            if result_expansion not in y_expansion_temp:
                y_expansion_temp.update({result_expansion: 1})
            else:
                y_expansion_temp[result_expansion] += 1

    return [y_focus_temp, y_expansion_temp]


if __name__ == '__main__':
    start_time = time.time()

    processing_pool = Pool(processes=thread_pool_size)
    gacha_results = []

    for j in range(thread_number):
        gacha_results.append([
            j,
            processing_pool.apply_async(test)
        ])

    for gacha_result in gacha_results:
        [y_focus, y_expansion] = gacha_result[1].get()
        for k in y_focus.keys():
            y_focus_truncate[k] += y_focus[k]
        for k in y_expansion.keys():
            y_expansion_truncate[k] += y_expansion[k]
        print('第{}个进程计算完毕'.format(gacha_result[0] + 1))

    processing_pool.close()
    print('总共耗时{}s'.format(int(time.time() - start_time)))

    # %%
    def draw_result():
        data = [
            go.Scatter(
                x=[i for i in range(x_truncate)],
                y=y_focus_truncate[:x_truncate],
                mode='markers',
                marker={
                    'color': 'rgb(255, 0, 0)'
                },
                name='focus'
            ),
            go.Scatter(
                x=[i for i in range(x_truncate)],
                y=y_expansion_truncate[:x_truncate],
                mode='markers',
                marker={
                    'color': 'rgb(0, 0, 255)'
                },
                name='expansion'
            ),
        ]
        fig = go.Figure(
            data=data
        )
        # fig.show()
        offline.plot(fig)

    draw_result()

    # %%
    def generate_statistics():
        average_focus, average_expansion = 0, 0
        sum_focus, sum_expansion = sum(y_focus_truncate), sum(y_expansion_truncate)

        for i in range(x_truncate):
            average_focus += i * y_focus_truncate[i]
            average_expansion += i * y_expansion_truncate[i]

        print('精准补给平均数: {}'.format(average_focus / sum_focus))
        print('扩充补给平均数: {}'.format(average_expansion / sum_expansion))

        temp_sum_focus, temp_sum_expansion = sum_focus / 2.0, sum_expansion / 2.0

        for i in range(x_truncate):
            if temp_sum_focus >= y_focus_truncate[i]:
                temp_sum_focus -= y_focus_truncate[i]
            else:
                print('精准补给中位数: {}'.format(i))
                break

        for i in range(x_truncate):
            if temp_sum_expansion >= y_expansion_truncate[i]:
                temp_sum_expansion -= y_expansion_truncate[i]
            else:
                print('扩充补给中位数: {}'.format(i))
                break

    generate_statistics()
