from fastapi import APIRouter, UploadFile, File, Form
from app.services.crack_service import analyze_crack
from app.services.satellite_service import analyze_satellite

router = APIRouter()


@router.post("/analyze")
async def analyze(
    crack_image: UploadFile = File(None),
    satellite_file: UploadFile = File(None),
    borewell_depth: float = Form(0)
):

    crack_result = None
    satellite_result = None


    # -----------------------
    # Crack analysis
    # -----------------------

    if crack_image:

        crack_bytes = await crack_image.read()

        crack_result = analyze_crack(crack_bytes)



    # -----------------------
    # Satellite analysis
    # -----------------------

    if satellite_file:

        sat_bytes = await satellite_file.read()

        satellite_result = analyze_satellite(
            sat_bytes,
            borewell_depth
        )



    # -----------------------
    # Final AI risk score
    # -----------------------

    crack_score = 0
    satellite_score = 0


    if crack_result:

        crack_score = crack_result["severity"]


    if satellite_result:

        satellite_score = satellite_result["area_ratio"]



    final_score = (0.6 * crack_score) + (0.4 * satellite_score)

    final_score = min(final_score, 1)



    # -----------------------
    # Final risk classification
    # -----------------------

    if final_score > 0.75:
        final_risk = "SEVERE"

    elif final_score > 0.5:
        final_risk = "HIGH"

    elif final_score > 0.25:
        final_risk = "MODERATE"

    else:
        final_risk = "LOW"



    return {

        "crack": crack_result,

        "satellite": satellite_result,

        "final_score": float(final_score),

        "final_risk": final_risk

    }
