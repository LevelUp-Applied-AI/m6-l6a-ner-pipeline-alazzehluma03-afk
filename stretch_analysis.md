# EntityRuler BEFORE NER Evaluation

## Analysis

The custom EntityRuler improved domain-specific entity coverage by identifying climate concepts that the base spaCy model often missed, including examples such as "Paris Agreement", "COP28", "IPCC AR6", and "2°C target". These custom rules expanded the extracted entity landscape by surfacing meaningful climate terminology beyond what the pre-trained model captured on its own.

Evaluation against the gold-standard subset for overlapping standard labels produced a precision of 0.0495, recall of 0.6471, and F1 score of 0.0921. The relatively strong recall indicates the pipeline recovered a substantial portion of gold entities, while low precision suggests many predicted entities did not align exactly with the gold annotations. This reflects noise introduced by rule-based matching and some false positives from added entity patterns.

The custom rules were most helpful when they captured domain entities the base model missed or misclassified, particularly structured phrases like "Paris Agreement" and variable expressions such as "2°C target". However, some rule matches introduced errors when patterns fired too broadly or produced entities that did not correspond to gold-standard labels, which contributed to lower precision.

Overall, the EntityRuler improved qualitative domain coverage while introducing a tradeoff between broader entity detection and precision. This reflects a common production NER engineering challenge: domain-specific rules can improve recall and coverage, but they must be tuned carefully to minimize noise.