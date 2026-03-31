def get_treatment(disease):
    remedies = {
        "Tomato_Early_blight": "Use neem oil spray every 5 days.",
        "Tomato_Late_blight": "Apply fungicide and avoid overwatering.",
        "Healthy": "No treatment needed. Maintain proper care."
    }

    return remedies.get(disease, "Consult agricultural expert.")