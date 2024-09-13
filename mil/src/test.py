
import random
import pickle
from utils import make_train_valid_dfs
import re



# Generate 75 random integers between 0 and 5000 (inclusive)
num = 20

# csv_val_path = '/home/alvi/KG_Defence/datasets/flickr/captions_val.csv'

# _, val_dataframe = make_train_valid_dfs(csv_val_path=csv_val_path)

# image_filenames = val_dataframe['image_file'].values[::5]
# print (len(image_filenames))
# random_integers = [random.randint(0, len(image_filenames)) for _ in range(num)]

# file_path = '/home/alvi/KG_Defence/mil/results_flickr/random_index_75.pkl'
# with open(file_path, 'wb') as file:
#     pickle.dump(random_integers, file)


# # Specify the path for the .pkl file
file_path = '/home/alvi/KG_Defence/mil/src/sofa_captions_flickr_20.pkl'

 
# csv_train_path = '/home/alvi/KG_Defence/datasets/coco/csv_train.csv'
# csv_val_path = '/home/alvi/KG_Defence/datasets/coco/csv_val.csv'

csv_val_path = '/home/alvi/KG_Defence/datasets/flickr/captions_val.csv'
_, df = make_train_valid_dfs(csv_val_path=csv_val_path)

image_filenames = df['image_file'][::5].values ## all image

image_to_posion = 'sofa'
filtered_rows = df[df['caption'].apply(lambda objects_list: image_to_posion in objects_list)]
image_filenames_dog = filtered_rows['caption'].tolist() ## all dog images to poison

image_filenames_dog_sample = random.sample(image_filenames_dog, num)
print (image_filenames_dog_sample[:5])
# # Save the list of random integers to the .pkl file
with open(file_path, 'wb') as file:
    pickle.dump(image_filenames_dog_sample, file)

# with open(file_path, 'rb') as file:
#     loaded_random_dog_filenames = pickle.load(file)
#     print (len(loaded_random_dog_filenames))


##############################################


# import re
# ke_val = '/home/alvi/KGGuidedLearning/FastChat/slurm-1785871.out'

# with open(ke_val, 'r') as file:
#     # Read each line and filter lines that start with '['
#     filtered_lines = []
#     outputs = []
#     mismatch = []
#     filtered_lines1= []

#     cnt = 0
#     lst_cnt = []
#     for line in file:
#         if line.startswith("outputs"):
#             outputs.append(line)
#             match = re.search(r'\[.*', line)
#             if match:
#                 parts = line.split(' ')
#                 ind = parts[1]
#                 lst_cnt.append(ind)
#                 filtered_lines.append(match.group())
#             else: 
#                 match = re.search("I .*", line)
#                 if match:
#                     parts = line.split(' ')
#                     new_line = ' '.join(parts[:2])
#                     mismatch.append(new_line)
#                 else: 
#                     parts = line.split(' ')
#                     if len(parts) > 1:
#                         new_line = ' '.join(parts[2:])
#                         new_line = '[' + new_line + ']'
#                         ind = parts[1]
#                         lst_cnt.append(ind)
#                         filtered_lines.append (new_line)



# print (len(filtered_lines), len(outputs), len(lst_cnt))
# print (mismatch, len(mismatch))
# print('---------')
# # print(filtered_lines1, len(filtered_lines1))
# # Open a new text file for writing the filtered lines

# noise_train = '/home/alvi/KG_Defence/datasets/flickr/noise_train_id.txt'

# lst_cnt = []
# with open(noise_train, 'r') as noise_file:
#     for line in noise_file:
#         # print(int(line))
#         lst_cnt.append(int(line))



# with open('/home/alvi/KG_Defence/datasets/flickr/filtered_ke_train_rest1.txt', 'w') as filtered_file:
#     # Write the filtered lines to the new file
#     for line, cnt in zip(filtered_lines, lst_cnt):
#         line = str(cnt) + " " +line
#         filtered_file.write(line)
#         filtered_file.write('\n')

# filtered_file.close()

# with open('/home/alvi/KG_Defence/datasets/flickr/noise_train.txt', 'w') as filtered_file1:
#     # Write the filtered lines to the new file
#     for line in mismatch:
#         line = line.split(' ')[1]
#         filtered_file1.write(line)
#         filtered_file1.write('\n')

# filtered_file1.close()

# noise_list = []
# with open('/home/alvi/KG_Defence/datasets/flickr/noise_train.txt', 'r') as noise_file:
#      for line in noise_file:
#           noise_list.append(int(line))

# print (noise_list)
# with open('/home/alvi/KG_Defence/datasets/flickr/filtered_ke_train.txt', 'r') as filtered_file:
#     cnt = 1
#     for line in filtered_file:
#           line = line.split(' ')[0]
#           if (int(line) in noise_list):
#                print (line)
#     # print (cnt)

# noise_lst = ['1652', '3795', '4458', '10962', '13855', '14074', '16515', '17550', '21500', '22715', '25360', '26443', '26739', '28317', '31146', '20890', '23934', '24811']
# with open('/home/alvi/KG_Defence/datasets/flickr/filtered_ke_val.txt', 'r') as filtered_file:

#     cnt = 1
#     for line in filtered_file:
#         parts = line.split(' ')
#         ind = parts[0]
#         # print (ind)
#         if (str(ind) in noise_lst):
#             print (str(ind))
#         cnt+=1
    
#     # print (cnt)

