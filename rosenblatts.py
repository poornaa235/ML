# AND data (can get from file)
data = [[0,0,0], [0,1,1], [1,0,1], [1,1,1]]

X = []
y = []
for row in data:
    X.append(row[:-1])
    y.append(row[-1])
    
num_features = len(X[0])

# DYNAMIC LABEL DETECTION
pos_class = max(y)  # Usually 1
neg_class = min(y)  # Automatically detects 0 or -1

# User Input
threshold = float(input("Enter your threshold value: "))
lr = 0.51
weights = [1.2, 0.6] 
max_epochs = 10

# Start
previous_weights = None

for epoch in range(max_epochs):
    
    previous_weights = list(weights)
    
    for i in range(len(X)):
        raw_sum = 0
        
        for j in range(num_features):
            raw_sum += weights[j] * X[i][j]
        
        if raw_sum >= threshold:
            prediction = pos_class
        else:
            prediction = neg_class
        
        if y[i] != prediction:
            for j in range(num_features):
                weights[j] += lr * (y[i] - prediction) * X[i][j]
                
    print(f"Epoch {epoch + 1:2d} completed | Current Weights: {[round(w, 2) for w in weights]}")
                
    if weights == previous_weights:
        print(f"\nConverged early! Weights did not change between epoch {epoch} and {epoch + 1}.")
        break

#Final(if needed)
print(f"\nFinal Learned Weights: {weights}")

