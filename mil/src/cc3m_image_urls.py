# import csv

# # Define the path to your TSV file and the output text file
# tsv_file_path = '/home/alvi/KG_Defence/datasets/CC3M/Train-GCC-training.tsv'
# text_file_path = '/home/alvi/KG_Defence/datasets/CC3M/image_urls_train.txt'

# # Open the TSV file for reading
# with open(tsv_file_path, 'r', newline='') as tsvfile:
#     # Create a TSV reader with the delimiter set to '\t' for tab separation
#     print ('inside ')
#     tsvreader = csv.reader(tsvfile, delimiter='\t')
    
#     print('reader is done')
#     # Open the text file for writing
#     with open(text_file_path, 'w') as textfile:
#         # Loop through each row in the TSV file
#         for row in tsvreader:
#             # Check if the row has at least two columns (0-indexed)
#             if len(row) > 1:
#                 # Write the second column (index 1) to the text file
#                 textfile.write(row[1] + '\n')

# print ('the end')
# # The second column has been written to the output text file


from img2dataset import download
import shutil
import os

output_dir = '/globalscratch/alvi/cc3m'

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

img_url = '/home/alvi/KG_Defence/datasets/CC3M/image_urls_val.txt'

download(
    # processes_count=16,
    # thread_count=32,
    url_list=img_url,
    image_size=256,
    output_folder=output_dir,
    # output_format="files",
    # input_format="parquet",
    # url_col="URL",
    # caption_col="TEXT",
    enable_wandb=True,
    # number_sample_per_shard=1000,
    # distributor="multiprocessing",
)
