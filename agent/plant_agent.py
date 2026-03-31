from tools.disease_detector import predict_disease
from tools.llm_advisor import generate_advice
from tools.severity_checker import check_severity

def plant_agent(image_path):
    # Step 1: Detect disease
    result = predict_disease(image_path)

    disease = result["disease"]
    confidence = result["confidence"]

    # Step 2: Check severity
    severity = check_severity(confidence)

    # Step 3: Get treatment
    treatment = generate_advice(disease, severity)

    # Step 4: Decision logic
    if confidence < 0.5:
        return {
            "message": "I'm not confident. Please upload a clearer image."
        }

    # Step 5: Final response
    return {
        "disease": disease,
        "confidence": round(confidence, 2),
        "severity": severity,
        "treatment": treatment
    }