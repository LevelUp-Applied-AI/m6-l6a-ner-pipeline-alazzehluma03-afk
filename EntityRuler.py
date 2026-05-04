import spacy
import json
import pandas as pd
from spacy.pipeline import EntityRuler
from ner_pipeline import evaluate_ner

def load_patterns(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

def build_climate_pipeline(patterns_file, position="before"):
    nlp = spacy.load("en_core_web_sm")
    patterns = load_patterns(patterns_file)
    
    if position == "before":
        ruler = nlp.add_pipe("entity_ruler", before="ner")
    else:
        ruler = nlp.add_pipe("entity_ruler", after="ner")
        
    ruler.add_patterns(patterns)
    return nlp

def extract_entities(df, nlp):
    entities = []
    for _, row in df[df['language'] == 'en'].iterrows():
        doc = nlp(row['text'])
        for ent in doc.ents:
            entities.append({
                'text_id': row['id'],
                'entity_text': ent.text,
                'entity_label': ent.label_
            })
    return pd.DataFrame(entities)

def run_evaluation(results_df, gold_df, position):
    standard_labels = ['ORG', 'GPE', 'DATE', 'LAW', 'MONEY', 'PERSON', 'QUANTITY', 'LOC', 'EVENT', 'WORK_OF_ART']
    filtered_preds = results_df[results_df['entity_label'].isin(standard_labels)]
    
    metrics = evaluate_ner(filtered_preds, gold_df)
    
    with open(f"evaluation_summary_{position}.txt", "w") as f:
        f.write(f"=== EntityRuler Evaluation Summary ({position}) ===\n")
        f.write("Performance Metrics (Standard Entities Only):\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value}\n")
    return metrics

if __name__ == "__main__":
    df = pd.read_csv("data/climate_articles.csv")
    gold_df = pd.read_csv("data/gold_entities.csv")
    
    for pos in ["before", "after"]:
        nlp = build_climate_pipeline("climate_patterns.json", position=pos)
        results_df = extract_entities(df, nlp)
        results_df.to_csv(f"ruler_output_{pos}_ner.csv", index=False)
        run_evaluation(results_df, gold_df, position=pos)