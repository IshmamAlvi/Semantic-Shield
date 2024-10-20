import numpy as np
import matplotlib.pyplot as plt
# plt.rcParams['font.family'] = 'DeJavu Serif'

plt.rcParams.update({
    'font.size': 20,
    # 'text.usetex': True,
    # 'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
})
# plt.rcParams['font.serif'] = ['Times New Roman']
plt.figure(figsize=(10,8))
# poison_rate = ['0.001%', '0.005%', '0.008%', '0.01%']


# h_1 = [85.32, 87.90, 88.90, 90.66]
# h_5 = [86.12, 88.21, 90.45,   94.60]
# h_10 = [88.23, 91.29, 93.76,  95.43]

# h_1 = [90.26, 93.32, 95.89, 100.0]
# h_5 = [91.76, 94.31, 97.86,  100.0]
# h_10 = [93.45, 95.94, 98.61,  100.0]

# h_1 = [89.43, 93.21, 98.94, 100.0]
# h_5 = [91.62, 95.13, 97.86,  100.0]
# h_10 = [92.55, 96.43, 98.61,  100.0]

## defense fro semantic sheild
epochs = [10, 20, 25, 30]

# h_1 = [20.23, 10.21, 3.21, 0.9]
# h_5 = [24.67, 12.35, 5.23,  1.22]
# h_10 = [27.32, 14.37, 6.65,  1.57]

h_1 = [15.42, 8.45, 2.43, 0.0]
h_5 = [18.72, 8.51, 3.31,  0.0]
h_10 = [22.35, 12.71, 5.50,  0.0]


# h_1 = [14.42, 6.45, 3.43, 0.0]
# h_5 = [19.72, 9.51, 4.31,  0.0]
# h_10 = [20.35, 10.71, 6.50,  0.0]

# Plot the first line
plt.plot(epochs, h_1, label='Hit@1', color='blue', linewidth=2, marker = 'v', linestyle='dashed', markersize=20)

# Plot the second line
plt.plot(epochs, h_5, label='Hit@5', color='green', linewidth=2, marker = 'x', linestyle='dashed', markersize=20)

# Plot the third line
plt.plot(epochs, h_10, label='Hit@10', color='red', linewidth=2, marker = 'o', linestyle = 'dashed', markersize=20)

# csfont = {'fontname':'Times New Roman'}
# Customize the plot
plt.title('Backdoor-Wanet for COCO', fontsize = 35)
plt.xlabel('# Epochs', fontsize=35)
plt.ylabel('Hit@k', fontsize=35)
plt.legend(fontsize=35)
plt.grid(True)
plt.tight_layout()
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
# Save the plot to an image file (e.g., PNG)
plt.savefig('/home/alvi/KG_Defence/mil/results/defense_wanet.png', dpi=1000)
plt.savefig('/home/alvi/KG_Defence/mil/results/defense_wanet.pdf', format='pdf', dpi=4000)
