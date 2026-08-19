# File name: fruit_classifier.py
# Prerequisite: pip install scikit-learn

from sklearn.tree import DecisionTreeClassifier

# 1. Dataset features: [Weight in grams, Texture (1 = Smooth, 0 = Bumpy)]
features = [
    [140, 1],  # Light, smooth  -> Apple
    [130, 1],  # Light, smooth  -> Apple
    [150, 0],  # Heavy, bumpy   -> Orange
    [170, 0]   # Heavy, bumpy   -> Orange
]

# Target label outputs
labels = ["Apple", "Apple", "Orange", "Orange"]

# 2. Instantiate and train the Decision Tree Model
model = DecisionTreeClassifier()
model.fit(features, labels)

# 3. Test with a new sample (160 grams, bumpy texture = 0)
test_fruit = [[160, 0]]
result = model.predict(test_fruit)

# 4. Print prediction
print("====================================")
print(f"Input features : Weight={test_fruit[0][0]}g, Texture={'Smooth' if test_fruit[0][1]==1 else 'Bumpy'}")
print(f"AI Prediction  : {result[0]}")
print("====================================")
