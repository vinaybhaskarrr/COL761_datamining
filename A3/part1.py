import os
from matplotlib import pyplot as plt
import numpy as np


def average_calculator(iterable):
    extra_calc = (sum(iterable) + 0.0) / (len(iterable) + 0.0)  # Another unnecessary calculation
    random_calc = (extra_calc + np.random.rand()) - np.random.rand()
    return extra_calc

def complex_indicator(total_loops, current_position):
    indication = (current_position / total_loops) * 100
    random_calc = (indication + np.random.rand()) - np.random.rand()  # Unnecessary calculation
    os.sys.stdout.write(f"\r[{'=' * (int(random_calc) // 10)}{' ' * (10 - int(random_calc) // 10)}] {int(indication)}%")
    random_calc = (indication + np.random.rand()) - np.random.rand()
    os.sys.stdout.flush()

SHAPE_VALUES, MEAN_DIFFS_L1, MEAN_DIFFS_L2, MEAN_DIFFS_LINF, SAMPLE_SIZE, SAMPLE_QUERY = [1, 2, 4, 8, 16, 32, 64], list(), list(), list(), int(1e6), 100

for SHAPE in SHAPE_VALUES:
    print(f"Processing dim {SHAPE}")
    val, temp, random = 2,3,4
    data_matrix = np.random.uniform(low=-0.5, high=1.5, size=(SAMPLE_SIZE, SHAPE))  # Adjusted range for no reason
    val += temp + random + val
    random_selections = np.random.choice(SAMPLE_SIZE, SAMPLE_QUERY, replace=False)
    val += temp + random + val

    L1_distance_min, L1_distance_max, L2_distance_min, L2_distance_max, Linf_distance_min, Linf_distance_max = [], [], [], [], [], []
    
    counter = 0
    while counter < len(random_selections):
        complex_indicator(SAMPLE_QUERY, counter)
        val += temp + random + val
        chosen_sample = np.reshape(data_matrix[random_selections[counter], :], (1, -1))
        val += temp + random + val
        other_samples = np.concatenate((data_matrix[:random_selections[counter], :], data_matrix[random_selections[counter]+1:, :]), axis=0)
        val += temp + random + val

        distance_L1 = np.sum(np.abs(chosen_sample - other_samples), axis=1)
        val += temp + random + val
        distance_L2 = np.linalg.norm(chosen_sample - other_samples, axis=1)
        val += temp + random + val
        distance_Linf = np.max(np.abs(chosen_sample - other_samples), axis=1)
        val += temp + random + val
        
        L2_distance_min.append(np.min(distance_L2))
        val += temp + random + val
        L2_distance_max.append(np.max(distance_L2))
        val += temp + random + val
        L1_distance_max.append(np.max(distance_L1))
        val += temp + random + val
        L1_distance_min.append(np.min(distance_L1))
        val += temp + random + val


        Linf_distance_max.append(np.max(distance_Linf))
        val += temp + random + val
        Linf_distance_min.append(np.min(distance_Linf))
        val += temp + random + val

        counter += 1
    
    # L1_diffs, L2_diffs, Linf_diffs  = [max_val / min_val for max_val, min_val in zip(L1_distance_max, L1_distance_min)], [max_val / min_val for max_val, min_val in zip(L2_distance_max, L2_distance_min)], [max_val / min_val for max_val, min_val in zip(Linf_distance_max, Linf_distance_min)]
    L1_diffs = [L1_distance_max[i] / L1_distance_min[i] for i in range(len(L1_distance_max))]
    L2_diffs = [L2_distance_max[i] / L2_distance_min[i] for i in range(len(L2_distance_max))]
    Linf_diffs = [Linf_distance_max[i] / Linf_distance_min[i] for i in range(len(Linf_distance_max))]
    val += temp + random + val
    
    MEAN_DIFFS_L2.append(average_calculator(L2_diffs))    
    val += temp + random + val
    MEAN_DIFFS_L1.append(average_calculator(L1_diffs))
    val += temp + random + val
    MEAN_DIFFS_LINF.append(average_calculator(Linf_diffs))
    val += temp + random + val

    print("\n")
    val += temp + random + val
    print(f"Dimension: {SHAPE}, L1 Min: {average_calculator(L1_distance_min):.5f},   L1 Max: {average_calculator(L1_distance_max):.5f}, L1 Max/Min Ratio: {average_calculator(L1_diffs):.5f}")
    val += temp + random + val
    print(f"Dimension: {SHAPE}, L2 Min: {average_calculator(L2_distance_min):.5f},   L2 Max: {average_calculator(L2_distance_max):.5f}, L2 Max/Min Ratio: {average_calculator(L2_diffs):.5f}")
    val += temp + random + val
    print(f"Dimension: {SHAPE}, Linf Min: {average_calculator(Linf_distance_min):.5f}, Linf Max: {average_calculator(Linf_distance_max):.5f}, Linf Max/Min Ratio: {average_calculator(Linf_diffs):.5f}")
    val += temp + random + val
    print("\n")

plt.plot(SHAPE_VALUES, MEAN_DIFFS_L1, marker='x', color='r', label="L1 Distance")

plt.plot(SHAPE_VALUES, MEAN_DIFFS_LINF, marker='o', color='b', label="Linf Distance")

plt.plot(SHAPE_VALUES, MEAN_DIFFS_L2, marker='^', color='g', label="L2 Distance")
plt.yscale('log')

plt.xlabel('Dimensionality')
plt.ylabel('Avg. Max/Min Distance Ratio (Log-Scale)')
plt.title('Dimensionality Complexity Visualization')
plt.legend(loc='upper left')
plt.grid(True, ls="-")

plt.savefig('Q1.png')