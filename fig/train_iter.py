import matplotlib.pyplot as plt

llava_next_cot_v1 = {"sr": [20.80, 24.54, 28.49, 32.72], "osr": [44.92, 41.75, 49.79, 55.29], "ne": [98.34, 108.63, 91.28, 83.93],"spl": [17.17, 19.38, 23.02, 25.82]}
llava_llama_attn = {"sr": [18.34, 11.64, 19.53, 22.43], "osr": [36.39, 22.78, 42.81, 44.57], "ne": [107.60, 150.78, 90.80, 103.26],"spl": [14.26, 8.81, 16.09, 18.62]}
# our = {"sr": [20.80, 24.54, 28.49, 32.72], "osr": [44.92, 41.75, 49.79, 55.29], "ne": [98.34, 108.63, 91.28, 83.93],"spl": [17.17, 19.38, 23.02, 25.82]}
x = [1000, 3000, 5000, 6686]

# llava_next_cot_v1 = {"sr": [20.80, 24.54, 28.49, 32.72], "osr": [44.92, 41.75, 49.79, 55.29], "ne": [98.34, 108.63, 91.28, 83.93],"spl": [17.17, 19.38, 23.02, 25.82]}
# llava_llama_attn = {"sr": [18.34, 11.64, 19.53, 22.43], "osr": [36.39, 22.78, 42.81, 44.57], "ne": [107.60, 150.78, 90.80, 103.26],"spl": [14.26, 8.81, 16.09, 18.62]}
# x = [1000, 3000, 5000, 6686]
metrics = ['sr', 'osr', 'ne', 'spl']
metrics_name = {'sr': 'SR (%)', 'osr': 'OSR (%)', 'ne': 'NE (m)', 'spl': 'SPL'}


for metric in metrics:
    plt.figure()
    plt.plot(x, llava_next_cot_v1[metric], marker='o', label='llava_next_cot_v1')
    plt.plot(x, llava_llama_attn[metric], marker='s', label='llava_llama_attn')
    plt.xlabel('x')
    plt.ylabel(metric)
    plt.title(f'{metrics_name[metric]}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{metric}.png')
