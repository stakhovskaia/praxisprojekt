def __mapping(aspect):
        map = {
            "ob": "Objective",
            'p': "Participants",
            'i': "Intervention",
            'c': "Comparator",
            'o': "Outcomes",
            'f': "Findings",
            'm': "Medicines",
            'd': "Treatment Duration",
            'pe': "Primary Endpoints",
            's': "Secondary Endpoints",
            'fo': "Follow-up Duration",
            'ae': "Adverse Events",
            'r': "Randomiation Method",
            'b': "Blinding Method",
            'fu': "Funding",
            'rf': "Registration Information",
            'se': "Secondary Endpoints",
            'fd': "Follow-up Duration",
            'td': "Treatment Duration"

        }

        return map[aspect]

def prompts(sentence, aspect):
    content = f"""
    Given a sentence and an aspect, please extract key phrases from the sentence related the given aspect".
    Sentences: {sentence}. Aspect: {__mapping(aspect)}
    """
    return content
