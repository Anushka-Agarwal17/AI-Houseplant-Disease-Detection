def check_severity(confidence):
    if confidence > 0.85:
        return "High"
    elif confidence > 0.60:
        return "Medium"
    else:
        return "Low"