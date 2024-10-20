import pandas as pd 
from utils import make_train_valid_dfs
import os
import shutil

# source_folder = '/home/alvi/KGGuidedLearning/flickr30k_images'  # Replace with the actual path to your source folder
# tr_destination_folder = '/home/alvi/KG_Defence/datasets/flickr/images/train'  # Replace with the actual path to your destination folder
# val_destination_folder = '/home/alvi/KG_Defence/datasets/flickr/images/val'  # Replace with the actual path to your destination folder

# train_df, val_df = make_train_valid_dfs('/home/alvi/KG_Defence/datasets/flickr/captions_train.csv', '/home/alvi/KG_Defence/datasets/flickr/captions_val.csv')

# for filename in train_df['image_file'].values[::5]:
#     source_path = os.path.join(source_folder, filename)
#     destination_path = os.path.join(tr_destination_folder, filename)
    
#     # Check if the file exists in the source folder
#     if os.path.exists(source_path):
#         # Copy the file to the destination folder
#         shutil.copy(source_path, destination_path)
#     else:
#         print(f"File not found: {filename}")


# print ('going in val')

# for filename in val_df['image_file'].values[::5]:
#     source_path = os.path.join(source_folder, filename)
#     destination_path = os.path.join(val_destination_folder, filename)
    
#     # Check if the file exists in the source folder
#     if os.path.exists(source_path):
#         # Copy the file to the destination folder
#         shutil.copy(source_path, destination_path)
#     else:
#         print(f"File not found: {filename}")

# noise_train = '/home/alvi/KG_Defence/datasets/flickr/noise_train.txt'
# # csv_train = '/home/alvi/KG_Defence/datasets/flickr/captions_train.csv'

# train_df, val_df = make_train_valid_dfs('/home/alvi/KG_Defence/datasets/flickr/captions_train.csv', '/home/alvi/KG_Defence/datasets/flickr/captions_val.csv')


# with open(noise_train, 'r') as noise_train:
#     for id in noise_train:
#        caption =  train_df['caption'].iloc[int(id)]
#        print (caption)


# with open('/home/alvi/KG_Defence/datasets/flickr/filtered_ke_val.txt', 'r') as file1, open('/home/alvi/KG_Defence/datasets/flickr/filtered_ke_val_rest.txt', 'r') as file2:
#     content1 = file1.readlines()
#     content2 = file2.readlines()


# entries1 = [(int(line.split(' ')[0]), ' '.join(line.split(' ')[1:])) for line in content1]
# entries2 = [(int(line.split(' ')[0]), ' '.join(line.split(' ')[1:])) for line in content2]

# merged_entries = sorted(entries1 + entries2, key=lambda x: x[0])

# with open('/home/alvi/KG_Defence/datasets/flickr/filtered_ke_val_final1.txt', 'w') as merged_file:
#     for index, content in merged_entries:
#         merged_file.write(f"{index} {content}")



# ################################
import re
import pickle

# # Define a regular expression pattern to match valid lines
pattern = r'\[.*'

# Initialize a list to store valid lists of strings
valid_lists = []
invalid_list = []
# Read and process the input file
with open('/home/alvi/KG_Defence/datasets/flickr/filtered_ke_train_final1.txt', 'r') as input_file:
    for line in input_file:
        match = re.search(pattern, line)
        if match:
            # id = match.group(1)
            # strings = match.group(2).split(', ')
            parts = line.split(' [')[1].split(', ')
            id = line.split(' [')[0]
            if (len(parts) < 3):
                invalid_list.append(id)
            
            else: 
                cnt = 0
                lst = []
                # print('len: ', len(parts))
                for item in parts:
                    item = item.strip('\[').strip('\]').strip('\'').strip('\"')
                    # print(item)
                    if (cnt < 3):
                        lst.append(item)
                        cnt +=1
                    else: 
                        break
                valid_lists.append(lst)


print('---------------')
print(len(valid_lists))
print(len(invalid_list))
print(invalid_list)
print(valid_lists[:20])

for items in valid_lists:
    if (len (items) >  3):
        print (items)

print ('the end')




# Save the valid lists as a pickle file
# with open('/home/alvi/KG_Defence/datasets/flickr/filtered_ke_train_pickle.pkl', 'wb') as output_file:
#     pickle.dump(valid_lists, output_file)



# def find_duplicates(nums):
#     unique_set = set()
#     duplicates = set()

#     for num in nums:
#         if num in unique_set:
#             duplicates.add(num)
#         else:
#             unique_set.add(num)

#     return list(duplicates)

# lst = []
# with open ('/home/alvi/KG_Defence/datasets/flickr/filtered_ke_val_final1.txt') as file1:
#     for line in file1:
#         i = line.split(' ')[0]
#         lst.append(int(i))


# print(find_duplicates(lst))



