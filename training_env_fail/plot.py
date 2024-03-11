import matplotlib.pyplot as plt
from collections import deque

#
#   Draws a plot of values in avg_steps.txt
#   also draws an average 
#   Can also draw a point per 10 values but that is commented
#

file_path = 'avg_steps.txt'

# Read the file and extract indices and values
with open(file_path, 'r') as file:
    lines = file.readlines()

indices = []
values = []

for line in lines:
    index_str, value_str, _ = line.strip().split('\t')
    indices.append(int(index_str))
    values.append(float(value_str))

# Group values into chunks of 10
grouped_indices = [indices[i:i+10] for i in range(0, len(indices), 10)]
grouped_values = [values[i:i+10] for i in range(0, len(values), 10)]

# Avg line
vals = deque(maxlen=100)
avg = []
t = 0
for i, value in enumerate(values):
    vals.append(value)
    t = 0
    for i, x in enumerate(vals):
        t+=x
    avg.append(t/len(vals))

# Calculate mean for each group
mean_values = [sum(group) / len(group) for group in grouped_values]

# Flatten grouped indices for plotting
flat_indices = [index for group in grouped_indices for index in group][:len(mean_values)]

# Plot the values/group of valeus

# Grouped by 10
#plt.plot(flat_indices, mean_values, linestyle='-', color='b', label='Mean Values')
#plt.title('Index vs Mean Value (Grouped by 10)')

# All values
plt.plot(indices, values, linestyle="-", label="steps")
plt.plot(indices, avg, linestyle="--", label="Avg")
plt.title('Index vs Steps')

plt.xlabel('Index')
plt.ylabel('Mean Value')
#plt.ylabel('steps')
plt.legend()
plt.grid(True)
plt.show()
