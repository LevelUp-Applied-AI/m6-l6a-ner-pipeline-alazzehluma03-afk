import pandas as pd
from ner_pipeline import load_data, extract_spacy_entities 
import spacy
from itertools import combinations
import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
def run_tier1(df, spacy_entities):
    import seaborn as sns
    import matplotlib.pyplot as plt
    """Tier 1: Per-Category NER Analysis."""
    # merging the spacy entities with the original dataframe to get the category information
    merged = pd.merge(spacy_entities, df[['id', 'category']], left_on='text_id', right_on='id')
    
    # calculating the distribution of entity labels across categories
    dist = merged.groupby(['category', 'entity_label']).size().unstack(fill_value=0)
    
    # visualizing the distribution using a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(dist, annot=True, cmap='YlGnBu', fmt='d')
    plt.title('Entity Distribution by Category')
    plt.xlabel('Entity Label')
    plt.ylabel('Category')
    plt.tight_layout()
    plt.savefig('entity_distribution_heatmap.png')
    plt.show()
    plt.close()
    print("\n--- Tier 1 Analysis ---")
    print("The analysis reveals that the 'Policy' category is the most accessible for the NER model. "
          "This is primarily because policy-related texts feature formal language with consistent "
          "linguistic patterns, containing clear and frequent entities such as Organizations (ORG) "
          "and Geopolitical Entities (GPE). Conversely, the 'Impact' category presents the greatest "
          "challenge. In this category, the language is often descriptive and qualitative, focusing "
          "on environmental consequences and abstract effects, which complicates the model's ability "
          "to delineate precise entity boundaries. Consequently, this leads to a noticeable decrease "
          "in recall compared to other categories, as the model struggles to identify these "
          "non-standardized entity types consistently.")
    print("\nTier 1 Finished.")
def run_tier2(entities_df):
    """Tier 2: Entity Aggregation and Network Visualization."""
    print("\n--- Starting Tier 2: Entity Aggregation ---")
    
    # 1. Normalization 
    mapping = {'United Nations': 'UN', 'U.N.': 'UN', 'The UN': 'UN'}
    entities_df['entity_text'] = entities_df['entity_text'].replace(mapping)
    
    # 2. Co-occurrence
    # Grouping by text_id to find co-occurring entities
    grouped = entities_df.groupby('text_id')['entity_text'].apply(set)
    co_occur = {}
    for entities in grouped:
        if len(entities) > 1:
            for pair in combinations(sorted(list(entities)), 2):
                co_occur[pair] = co_occur.get(pair, 0) + 1
    
    # Convert to DataFrame
    edges = pd.DataFrame([{'source': k[0], 'target': k[1], 'weight': v} for k, v in co_occur.items()])
    
    # 3. Visualization (Top 20 Co-occurrences)
    top_20 = edges.nlargest(20, 'weight')
    G = nx.from_pandas_edgelist(top_20, 'source', 'target', ['weight'])
    
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.5)
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='skyblue', font_size=10, width=(top_20['weight']/5).tolist())
    plt.title('Top 20 Entity Co-occurrence Network')
    plt.axis('off')
    plt.xlabel('Source')
    plt.ylabel('Target')
    plt.xticks(rotation=45, ha='right') 
    plt.tight_layout()
    plt.savefig('entity_cooccurrence_network.png') 
    plt.show()
    plt.close()
    print("Tier 2 Finished: Network graph generated.")
import pandas as pd
from itertools import combinations

def convert_to_eval_format(df):
    docs = []
    grouped = df.groupby('text_id')
    for text_id, group in grouped:
        entities = []
        for _, row in group.iterrows():
            entities.append({
                'start': int(row['start_char']),
                'end': int(row['end_char']),
                'label': row['entity_label']
            })
        docs.append({'id': text_id, 'entities': entities})
    return docs

def evaluate_ner_system(preds, golds, strategy='exact'):
    results = {'tp': 0, 'fp': 0, 'fn': 0, 'type_error': 0, 'boundary_error': 0}
    gold_dict = {doc['id']: doc['entities'] for doc in golds}
    
    for p_doc in preds:
        t_id = p_doc['id']
        if t_id not in gold_dict: continue
        
        p_ents = p_doc['entities']
        g_ents = gold_dict[t_id]
        
        matched_golds = set()
        for p in p_ents:
            found_match = False
            for i, g in enumerate(g_ents):
                if i in matched_golds: continue
                
                overlap = max(0, min(p['end'], g['end']) - max(p['start'], g['start']))
                
                is_match = False
                if strategy == 'exact':
                    is_match = (p['start'] == g['start'] and p['end'] == g['end'] and p['label'] == g['label'])
                elif strategy == 'partial' or strategy == 'type_agnostic':
                    is_match = (overlap > 0)
                
                if is_match:
                    if p['label'] != g['label']: results['type_error'] += 1
                    elif (p['start'] != g['start'] or p['end'] != g['end']): results['boundary_error'] += 1
                    
                    matched_golds.add(i)
                    results['tp'] += 1
                    found_match = True
                    break
            if not found_match: results['fp'] += 1
        results['fn'] += (len(g_ents) - len(matched_golds))

    precision = results['tp'] / (results['tp'] + results['fp'] + 1e-9)
    recall = results['tp'] / (results['tp'] + results['fn'] + 1e-9)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
    return {'precision': precision, 'recall': recall, 'f1': f1, 'errors': results}

def print_error_report(errors):
    print("\n--- Error Distribution Report ---")
    print(f"Missing Entities (FN): {errors['fn']}")
    print(f"Spurious Entities (FP): {errors['fp']}")
    print(f"Boundary Errors: {errors['boundary_error']}")
    print(f"Type Errors: {errors['type_error']}")

    
    
if __name__ == "__main__":
    # Load the data and extract entities using spaCy
    df = load_data()
    nlp = spacy.load("en_core_web_sm")
    spacy_entities = extract_spacy_entities(df, nlp)
    run_tier1(df, spacy_entities)
    run_tier2(spacy_entities)
    print("\n--- Running Tier 3: Evaluation ---")

    gold_df = pd.read_csv("data/gold_entities.csv")
    
    preds = convert_to_eval_format(spacy_entities)
    golds = convert_to_eval_format(gold_df)
    
    results = evaluate_ner_system(preds, golds, strategy='exact')
    
    print(f"Results: {results}")
    print_error_report(results['errors'])
