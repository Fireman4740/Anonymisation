"""
Prompts RUPTA Multilingues

Prompts adaptés pour toutes les langues européennes.
Utilise un langage simple et universel, évitant les expressions idiomatiques.
"""

# ==================== PRIVACY EVALUATION ====================

PRIVACY_REFLECTION_FR_1 = """
You are an expert at identifying people from anonymized biographies.
You have extensive knowledge of public figures and can identify them even from partially anonymized texts.

Task: Generate a list of {p_threshold} candidates (names of public figures) most likely to be described by the following anonymized text. Rank them from most probable to least probable.

Ignore missing information or placeholders like '<PERSON>' or '[ORGANIZATION]'.

Anonymized text:
{anonymized_text}

{format_instructions}
"""

PRIVACY_REFLECTION_FR_2 = """
The person actually described by the text is: {people}

If this name appears in your candidate list, respond 'Yes' and identify the sensitive entities (words, dates, numbers) in the text that allowed you to deduce the person's identity. These entities should be generalized to reduce re-identification risk.

Otherwise, simply respond 'No'.

{format_instructions}
"""

PRIVACY_CONFIDENCE_FR = """
You are an expert at identifying public figures.

Task: Evaluate your confidence level for associating the following anonymized text with the mentioned person.

Confidence scale (0-100):
- 0: Completely irrelevant
- 1-50: Some details match
- 51-99: Most details match
- 100: Perfect match

Anonymized text:
{anonymized_text}

Candidate person: {people}

{format_instructions}
"""

# ==================== UTILITY EVALUATION ====================

UTILITY_REFLECTION_FR_1 = """
You are an expert at classifying occupations from biographies.

Task: Evaluate your confidence in classifying the following anonymized text into the occupation category: {label}

Confidence scale (0-100):
- 0: No confidence
- 1-50: Some details in the text match this occupation
- 51-99: Most details match this occupation
- 100: The text perfectly matches this occupation

Anonymized text:
{anonymized_text}

Target occupation: {label}

{format_instructions}
"""

UTILITY_CONFUSED_ENTITIES_FR = """
Identify the confusing entities in the "occupation description" section of the text that prevent you from correctly classifying the occupation with high confidence.

These entities should be made more specific to improve classification, while remaining general enough to preserve privacy.

Only mention a few of the most relevant entities for the occupation description.

{format_instructions}
"""

# ==================== OPTIMIZATION / REINFORCEMENT ====================

REINFORCEMENT_FR = """
Let's play a game to maximize accumulated rewards by editing biographies.

Rules:
1. You receive a biographical text with its editing history
2. Each edit has a privacy score and a utility score
3. Reward rules:
   - If privacy score ≤ {p_threshold}: reward = privacy score
   - If privacy score > {p_threshold}: reward = utility score

Your task: Edit the last entry in the history to obtain a higher reward.

Allowed operations:
- **Generalize**: Replace entities with more general equivalents (improves privacy)
  Example: "my husband and I" → "my partner and I"
- **Specify**: Replace generic terms with more specific entities from the original biography (improves utility)
  Example: "He is an athlete" → "He is a tennis player" (if mentioned in the original)

Goal: Maximize total accumulated reward with minimal changes to the text, preserving the original context and structure. You may make minor adjustments to maintain fluency and naturalness.

Style, format, and gender pronouns must not be changed.

Original biography: {original_text}

Editing history:
{editing_history}

{format_instructions}
"""

# ==================== SIMPLE REWRITING ====================

SIMPLE_REWRITING_FR = """
You are an expert anonymizer who carefully selects certain identifiable entities (words, dates, numbers) from the given biographical text and generalizes them to make the biography anonymous.

The anonymized text should be able to describe not one specific person but several similar people.

Instructions:
- Follow the required format precisely
- Change as little as possible, keep the original text intact as much as possible
- Only generalize information, do not invent new information
- Names can be replaced with generic pronouns like 'A person'
- Gender pronouns can be replaced with 'This person'
- Other identifiable entities → more general equivalents

Valid examples:
✓ "my husband and I" → "my partner and I"
✓ "my husband and I have a dog" → "my partner and I have a dog"  
✓ "my husband and I" → "I" (acceptable removal of information)

Invalid examples:
✗ "my husband and I" → "my wife and I"
✗ "my husband and I have a dog" → "my partner and I have a cat"

Note: Style and format must not be changed. You can only adapt small parts of the text. You may insert a minimum of tokens to ensure fluency. Deletion/removal of content is not allowed, only reduction of specificity through generalization.

Biography:
{input_text}

{format_instructions}
"""

# ==================== DETECTION ====================

DETECTION_FR = """
Complete the personally identifiable information detection task.

Task:
1. Identify the name of the person primarily described by the text
2. Identify all entities (words, dates, numbers) that can be connected to this person, such that if these entities were removed, the text would be anonymized (one could no longer know who is described)

Text:
{input_text}

{format_instructions}
"""

# ==================== GENERAL INSTRUCTIONS ====================

GENERAL_SYSTEM_FR = """
You are an efficient assistant. 
Respond only with the provided JSON template, filled with your answers.
"""

WHOLE_TASK_FR = """
You are a text anonymization annotator.
I need your help to anonymize the provided description text so it can be shared with others while preserving its utility.

Step-by-step procedure:
1. Detect all sensitive entities
2. Rewrite the text by replacing sensitive entities with more general equivalents
3. Insert or remove necessary tokens to ensure the final text is fluent and natural

The style and format of the output text must be consistent with the input text.
"""
