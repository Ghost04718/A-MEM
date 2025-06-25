analyze_and_structure_prompt = """
    Analyze the following text and generate a structured JSON object containing:
    1. "context": A concise, one-sentence summary that captures the core essence.
    2. "keywords": A list of the most relevant keywords (nouns, verbs, concepts).
    3. "tags": A list of broad categorical tags for classification.

    Text for analysis:
    ---
    {content}
    ---

    Return ONLY the JSON object.
    """
    
evolve_neighbors_prompt = """
    You are an AI memory evolution agent. A new memory has been added to the system.
    Analyze this new memory in relation to its neighbors and decide if any of the neighbors
    should be updated or "evolved" with new tags or a refined context.

    New Memory:
    - ID: {new_note.id}
    - Context: {new_note.context}
    - Keywords: {new_note.keywords}

    Neighboring Memories:
    {neighbors_context}

    Based on the new memory, should any of the neighboring memories be evolved?
    For each neighbor that needs an update, provide its ID and the new, evolved metadata.

    Return a JSON object with a key "evolutions", which is a list of objects.
    Each object should have:
    - "id": The ID of the neighbor memory to update.
    - "new_context": The updated context (or original if no change).
    - "new_tags": A complete list of the neighbor's tags, including any new ones.

    Example response:
    {{
        "evolutions": [
        {{
            "id": "neighbor-id-1",
            "new_context": "An evolved context incorporating new information.",
            "new_tags": ["existing_tag", "new_tag_from_evolution"]
        }}
        ]
    }}

    If no evolution is needed, return an empty list: {{"evolutions": []}}.
    """
    
