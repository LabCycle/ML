# 5 One-cell Apriori (Association Rule Mining - clean version)

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Sample transactions
transactions = [
    ['milk', 'bread', 'butter'],
    ['bread', 'butter'],
    ['milk', 'bread'],
    ['milk', 'butter'],
    ['bread', 'butter', 'jam'],
    ['milk', 'bread', 'butter', 'jam'],
    ['bread'],
    ['milk', 'bread', 'butter'],
    ['milk', 'jam'],
    ['bread', 'butter']
]

# Encode transactions
te = TransactionEncoder()
df = pd.DataFrame(te.fit(transactions).transform(transactions), columns=te.columns_)

# Apriori: frequent itemsets
freq_items = apriori(df, min_support=0.3, use_colnames=True)

# Association rules
rules = association_rules(freq_items, metric="confidence", min_threshold=0.6)

# Strong rules
strong_rules = rules[(rules['confidence'] > 0.7) & (rules['lift'] > 1)]

# Output
print("Frequent Itemsets:\n", freq_items)
print("\nAll Rules:\n", rules[['antecedents','consequents','support','confidence','lift']])
print("\nStrong Rules:\n", strong_rules[['antecedents','consequents','support','confidence','lift']])
