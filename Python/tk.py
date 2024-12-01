import os

import pandas as pd
from matplotlib import pyplot as plt
from apyori import apriori


def resource_path(filename: str) -> str:
    """Return the absolute path to a file in the 'resources' folder."""
    base_dir = os.path.dirname(os.path.dirname(__file__))  # Parent directory of the Python folder
    return os.path.join(base_dir, 'resources', filename)


# Importing data
dataset = pd.read_csv(resource_path('Market_Basket_Optimisation.csv'), header=None)

# Creating datasets
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

# Eclat
rules = apriori(transactions=transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2,
                max_length=2)
results = list(rules)
print(results)


def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    return list(zip(lhs, rhs, supports))


resultsinDataFrame = pd.DataFrame(inspect(results), columns=['Product 1', 'Product 2', 'Support'])

resultsinDataFrame = resultsinDataFrame.nlargest(n=10, columns='Support')

print(resultsinDataFrame)
