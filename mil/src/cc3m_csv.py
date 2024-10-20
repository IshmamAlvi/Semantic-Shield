import torch.utils.data as data
from PIL import Image
import os
import os.path
import random
import csv
import urllib.request
import requests
import pandas as pd

tsv_file = '/home/alvi/KG_Defence/datasets/CC3M/GCC-1.1.0-Validation.tsv'
csv_train = '/home/alvi/KG_Defence/datasets/CC3M/csv_val.csv'

captions = []
image_urls = []


object_per_row = []

column_names = ['image_file', 'caption', 'image_url']

# Open the CSV file in write mode
with open(csv_train, mode='w', newline='') as file:
    writer = csv.writer(file, delimiter='$')

    chunksize = 1000
    cnt = 0 
    step = 0 
    for chunk in pd.read_csv(tsv_file, chunksize=chunksize, delimiter='\t'):
    # with open(tsv_file, 'r', newline='') as tsvfile:  
        reader = chunk
        object_per_row = []

        for index, row in reader.iterrows():
            caption, image_url = row
            # print ('image url: ', image_url)
            step +=1
            try: 
                response = requests.get(image_url)
                if response.status_code == 200:
                    # print ('inside: ', cnt)
                    row_list = []
                    file_name = str(cnt) + '.jpg'  
                    row_list.append (file_name)
                    row_list.append(caption)
                    row_list.append(image_url)
                    object_per_row.append(row_list)
                    cnt +=1
                else: 
                    print ('got here: ', image_url, '---', response.status_code)
            except Exception as e: 
                print ('exception cnt: ', cnt)
                continue
            
        writer.writerows(object_per_row)
        print ('chuck count : ', step * chunksize)

# Column names
# column_names = ['image_file', 'caption', 'image_url']

# # Open the CSV file in write mode
# with open(csv_train, mode='w', newline='') as file:
#     writer = csv.writer(file, delimiter='$')

#     # Write the column names
#     writer.writerow(column_names)

#     # Write the items to the CSV file
    
#     writer.writerows(object_per_row)
#     print ('<<<<the end>>>>>>>')
