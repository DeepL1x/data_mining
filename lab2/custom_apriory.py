import pandas as pd

def get_frequent_itemsets(transactions_df: pd.DataFrame, min_support):
    # Saving frequent itemsets
    itemsets = {}

    for _, transaction in transactions_df.iterrows():
        # Convert row to a list of items that are present (value 1)
        items = [item for item, present in transaction.items() if present == 1]
        for item in items:
            # Increment counter for each item in the transaction
            itemsets[frozenset([item])] = itemsets.get(frozenset([item]), 0) + 1

    # Filter items that satisfy the minimum support
    min_support_count = min_support * len(transactions_df)
    itemsets = {item: count for item, count in itemsets.items() if count >= min_support_count}
    return itemsets

def generate_candidates(frequent_itemsets, k):
    candidates = set()
    # List of frequent itemsets
    frequent_items = list(frequent_itemsets.keys())
    frequent_items_len = len(frequent_items)
    for i in range(frequent_items_len):
        for j in range(i + 1, frequent_items_len):
            union_set = frequent_items[i] | frequent_items[j]
            # If the size of the union set is k
            if len(union_set) == k:
                candidates.add(union_set)
    return candidates

def apriori(transactions_df: pd.DataFrame, min_support):
    # Convert DataFrame to a list of sets for convenient subset checking
    transactions = [
        set(transaction.index[transaction == 1]) for _, transaction in transactions_df.iterrows()
    ]
    transactions_count = len(transactions)

    # Find frequent 1-itemsets that satisfy the minimum support
    frequent_itemsets = get_frequent_itemsets(transactions_df, min_support)

    # Initialize dictionary to store all frequent itemsets
    all_frequent_itemsets = dict(frequent_itemsets)

    # Start with itemsets of size 2
    k = 2

    # While there are frequent itemsets
    while frequent_itemsets:
        # Generate candidates for itemsets of size k
        candidates = generate_candidates(frequent_itemsets, k)
        
        candidate_counts = {candidate: 0 for candidate in candidates}

        # Count occurrences for each candidate
        for transaction in transactions:
            for candidate in candidates:
                if candidate.issubset(transaction):
                    candidate_counts[candidate] += 1

        # Keep only those candidates that satisfy the minimum support
        frequent_itemsets = {
            item: count for item, count in candidate_counts.items() if count >= min_support * transactions_count
        }
        all_frequent_itemsets.update(frequent_itemsets)
        k += 1

    result_df = pd.DataFrame(
        [
            {"Support": count / transactions_count, "Itemset": set(itemset) }
            for itemset, count in all_frequent_itemsets.items()
        ]
    )
    return result_df

if __name__ == "__main__":
    data = {'Milk': [1, 0, 1, 1, 0],
    'Bread': [1, 1, 0, 1, 1],
    'Butter': [0, 1, 1, 1, 0],
    'Cheese': [1, 0, 0, 1, 1]}

    df = pd.DataFrame(data)

    result = apriori(df, min_support=0.5)
    print("Frequent itemsets with support >= 0.5:") 
    print(result)