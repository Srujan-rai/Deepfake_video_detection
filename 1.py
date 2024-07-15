import csv
data = []
step=0
with open('ENJOYSPORT.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        data.append(row)
# Function to implement Find-S algorithm
def find_s_algorithm(data):
    # Initialize the hypothesis with the first positive example
    hypothesis = ['0'] * (len(data[0]) - 1)
    
    for example in data:
        if example[-1] == '1':
            for i in range(len(hypothesis)):
                if hypothesis[i] == '0':
                    hypothesis[i] = example[i]
                elif hypothesis[i] != example[i]:
                    hypothesis[i] = '?' 
    return hypothesis
# Find the most specific hypothesis
hypothesis = find_s_algorithm(data)
print("The most specific hypothesis is:", hypothesis)
