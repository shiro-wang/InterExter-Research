import metrics
answers = [
            "Cornerback"
        ]
rationale = "Based on the provided documents and internal knowledge, I can answer the question as follows:\n\nDeion Sanders was inducted into the Pro Football Hall of Fame in the Class of 2011."
is_accurate = metrics.exact_presence(answers, rationale)
print("is_accurate:", is_accurate)