from tools.disease_detector import predict_disease
from tools.llm_advisor import generate_advice
from tools.severity_checker import check_severity


def plant_agent(image_path):

    # ==========================================
    # STEP 1: Detect disease using ML model
    # ==========================================

    result = predict_disease(image_path)

    disease = result["disease"]
    confidence = result["confidence"]


    # ==========================================
    # STEP 2: Check confidence BEFORE Gemini
    # ==========================================

    if confidence < 0.5:
        return {
            "message": (
                "I'm not confident about this prediction. "
                "Please upload a clearer image of the plant leaf."
            )
        }


    # ==========================================
    # STEP 3: Determine severity
    # ==========================================

    severity = check_severity(confidence)


    # ==========================================
    # STEP 4: Get AI treatment advice
    # ==========================================

    try:

        treatment = generate_advice(
            disease,
            severity
        )

    except Exception as e:

        print("Gemini error:", e)

        treatment = (
            "AI treatment advice is temporarily unavailable.\n\n"
            "The disease prediction was completed successfully. "
            "Please try again later for personalized AI treatment "
            "and prevention recommendations."
        )


    # ==========================================
    # STEP 5: Return final result
    # ==========================================

    return {
        "disease": disease,
        "confidence": round(confidence, 2),
        "severity": severity,
        "treatment": treatment
    }
# from tools.disease_detector import predict_disease
# from tools.llm_advisor import generate_advice
# from tools.severity_checker import check_severity

# def plant_agent(image_path):
#     # Step 1: Detect disease
#     result = predict_disease(image_path)

#     disease = result["disease"]
#     confidence = result["confidence"]

#     # Step 2: Check severity
#     severity = check_severity(confidence)

#     # Step 3: Get treatment
#     treatment = generate_advice(disease, severity)

#     # Step 4: Decision logic
#     if confidence < 0.5:
#         return {
#             "message": "I'm not confident. Please upload a clearer image."
#         }

#     # Step 5: Final response
#     return {
#         "disease": disease,
#         "confidence": round(confidence, 2),
#         "severity": severity,
#         "treatment": treatment
#     }