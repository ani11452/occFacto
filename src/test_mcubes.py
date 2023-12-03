import os
import numpy as np
import random
from occFacto.train.data.fields import PointsField
from occFacto.train.data.transforms import SubsamplePointsHalf

# Initialize the transform and PointsField
transform = SubsamplePointsHalf(2048)
ptfield = PointsField("points.npz", transform)

# Directory containing the data
data_dir = "/home/cs236finalproject/diffFactoCS236/data/ShapeNet/03001627"

# Select 200 random directories
all_directories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
random_directories = random.sample(all_directories, min(1, len(all_directories)))

total_ones = 0
total_zeros = 0

# Loop through each selected directory
for directory in random_directories:
    file_path = os.path.join(data_dir, directory)
    if os.path.exists(file_path):
        # Load the data
        data = ptfield.load(file_path)
        occ = data['occ']

        # Count zeros and ones
        total_zeros += np.count_nonzero(occ == 0)
        total_ones += np.count_nonzero(occ == 1)

        print(f"Directory: {directory}, Ones: {total_ones}, Zeros: {total_zeros}")


# Print the directory and counts
print(f"Average: {total_ones / (total_ones + total_zeros)}")
