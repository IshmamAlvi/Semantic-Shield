import matplotlib.pyplot as plt
import numpy as np
### result for COCO
# epochs = [1, 2, 3, 4, 5]
epochs = [i for i in range(1, 17)]
# T2I
# r_1 = [10.57, 13.12, 14.69, 8.85 , 9.05]
# r_5 = [27.80, 33.59, 36.39, 24.98, 25.65]
# r_10 = [39.50, 46.25, 48.84, 35.68, 36.74]

##i2T
f = open ('/home/alvi/KG_Defence/mil/results/poison/poison_data_recall_k_t2i.txt', 'r')
f1 = open('/home/alvi/KG_Defence/mil/results/poison/poison_data_recall_k_i2t.txt', 'r')
r_1 = []
r_5 = []
r_10 = []

for line in f:
    line_arr = line.split(',')
    r_1.append(line_arr[0])
    r_5.append(line_arr[1])
    r_10.append(line_arr[2])


combined_y = np.concatenate((r_1, r_5, r_10))

# Calculate custom y-axis tick positions based on the combined y-values
print(min(combined_y),  max(combined_y))
custom_y_ticks = np.arange(int(float(min(combined_y))), int(float(max(combined_y))), 5)
# print(len(custom_y_ticks))
# print(custom_y_ticks)
# custom_y_ticks = np.arange(0, 80, 5) 


# Create a Matplotlib figure and axis
fig, ax = plt.subplots()

# Plot the three sets of y-values
ax.plot(epochs, r_1, label='r@1', marker='o')
ax.plot(epochs, r_5, label='r@5', marker='s')
ax.plot(epochs,r_10, label='r@10', marker='^')

# Add labels and legend
ax.set_xlabel('Epoch')
ax.set_ylabel('Recall@k')
ax.set_title('Epoch vs R@K plot T2I for COCO')
custom_y_ticks = np.arange(0, 80, 5) 
ax.set_yticks(custom_y_ticks)
ax.legend()

# Show the plot
plt.savefig('/home/alvi/KG_Defence/mil/figures/nonclip/recall@k/poison_model/poison_data_epoch_vs_recall_coco_t2i.png')


r_1 = []
r_5 = []
r_10 = []


for line in f1:
    line_arr = line.split(',')
    r_1.append(line_arr[0])
    r_5.append(line_arr[1])
    r_10.append(line_arr[2])




# Create a Matplotlib figure and axis
fig, ax = plt.subplots()

# Plot the three sets of y-values
ax.plot(epochs, r_1, label='r@1', marker='o')
ax.plot(epochs, r_5, label='r@5', marker='s')
ax.plot(epochs,r_10, label='r@10', marker='^')

# Add labels and legend
ax.set_xlabel('Epoch')
ax.set_ylabel('Recall@k')
ax.set_title('Epoch vs R@K plot I2T for COCO')
ax.set_yticks(custom_y_ticks)
ax.legend()

# Show the plot
plt.savefig('/home/alvi/KG_Defence/mil/figures/nonclip/recall@k/poison_model/poison_data_epoch_vs_recall_coco_i2t.png')
