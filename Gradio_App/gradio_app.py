import gradio as gr
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import html

# Get the directory containing app.py
BASE_DIR = Path(__file__).resolve().parent

# Load the saved model
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "models" / "covertype_model.joblib"

saved = joblib.load("../notebooks/covertype_model.joblib")

# Load the input samples
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "demo_samples.csv"

X_demo = pd.read_csv(DATA_PATH, index_col=0)

print("Model loaded successfully!")
print("Demo data loaded successfully!")

# Load the trained model and example samples
pipeline = saved["pipeline"]

cover_type_names = {
    1: "Spruce/Fir",
    2: "Lodgepole Pine",
    3: "Ponderosa Pine",
    4: "Cottonwood/Willow",
    5: "Aspen",
    6: "Douglas-fir",
    7: "Krummholz"
}

wilderness_area_names = {
    1: "Rawah Wilderness Area",
    2: "Neota Wilderness Area",
    3: "Comanche Peak Wilderness Area",
    4: "Cache la Poudre Wilderness Area",
}


soil_type_names = {
    1:  (2702, "Cathedral family — rock outcrop complex, extremely stony"),
    2:  (2703, "Vanet–Ratake families complex, very stony"),
    3:  (2704, "Haploborolis–rock outcrop complex, rubbly"),
    4:  (2705, "Ratake family–rock outcrop complex, rubbly"),
    5:  (2706, "Vanet family–rock outcrop complex, rubbly"),
    6:  (2717, "Vanet–Wetmore families–rock outcrop complex, stony"),
    7:  (3501, "Gothic family"),
    8:  (3502, "Supervisor–Limber families complex"),
    9:  (4201, "Troutville family, very stony"),
    10: (4703, "Bullwark–Catamount families–rock outcrop complex, rubbly"),
    11: (4704, "Bullwark–Catamount families–rock land complex, rubbly"),
    12: (4744, "Legault family–rock land complex, stony"),
    13: (4758, "Catamount family–rock land–Bullwark family complex, rubbly"),
    14: (5101, "Pachic Argiborolis–Aquolis complex"),
    15: (5151, "Unspecified soil mapping unit"),
    16: (6101, "Cryaquolis–Cryoborolis complex"),
    17: (6102, "Gateview family–Cryaquolis complex"),
    18: (6731, "Rogert family, very stony"),
    19: (7101, "Typic Cryaquolis–Borohemists complex"),
    20: (7102, "Typic Cryaquepts–Typic Cryaquolls complex"),
    21: (7103, "Typic Cryaquolls–Leighcan family complex"),
    22: (7201, "Leighcan family, till substratum, extremely bouldery"),
    23: (7202, "Leighcan family–Typic Cryaquolls complex"),
    24: (7700, "Leighcan family, extremely stony"),
    25: (7701, "Leighcan family, warm, extremely stony"),
    26: (7702, "Granile–Catamount families complex, very stony"),
    27: (7709, "Leighcan family, warm–rock outcrop complex"),
    28: (7710, "Leighcan family–rock outcrop complex"),
    29: (7745, "Como–Legault families complex, extremely stony"),
    30: (7746, "Como family–rock land–Legault family complex"),
    31: (7755, "Leighcan–Catamount families complex"),
    32: (7756, "Catamount–Leighcan families complex"),
    33: (7757, "Leighcan–Catamount families–rock outcrop complex"),
    34: (7790, "Cryorthents–rock land complex"),
    35: (8703, "Cryumbrepts–rock outcrop–Cryaquepts complex"),
    36: (8707, "Bross family–rock land–Cryumbrepts complex"),
    37: (8708, "Rock outcrop–Cryumbrepts–Cryorthents complex"),
    38: (8771, "Leighcan–Moran families–Cryaquolls complex"),
    39: (8772, "Moran family–Cryorthents–Leighcan family complex"),
    40: (8776, "Moran family–Cryorthents–rock land complex"),
}


def predict_samples(sample_id):
    if not sample_id:
        return None

    # Convert the user-facing sample number (starting at 1)
    # to the DataFrame position (starting at 0).
    position = int(sample_id) - 1
    new_data = X_demo.iloc[[position]]
        # Get the selected sample's landscape characteristics
    sample = new_data.iloc[0]

    elevation = int(sample["elevation"])

    wilderness_value = int(sample["wilderness_area"])
    wilderness_name = wilderness_area_names.get(
        wilderness_value,
        f"Wilderness Area {wilderness_value}"
    )

    soil_value = int(sample["soil_type"])

    soil_elu_code, soil_description = soil_type_names.get(
    soil_value,
    (None, f"Unknown soil category: {soil_value}")
)

    # Calculate probabilities for all seven forest types
    probabilities = pipeline.predict_proba(new_data)[0]
    classes = pipeline.classes_

    # Find the forest type with the highest probability
    highest_index = np.argmax(probabilities)
    predicted_class = int(classes[highest_index])
    predicted_forest = cover_type_names[predicted_class]
    highest_probability = probabilities[highest_index]

    # Prepare chart labels and bar colors
    forest_names = [
        cover_type_names[int(class_label)]
        for class_label in classes
    ]

    colors = [
        "green" if i == highest_index else "grey"
        for i in range(len(probabilities))
    ]

    # Create the vertical bar chart
    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.bar(
        forest_names,
        probabilities * 100,
        color=colors
    )

    # Add probability percentages above the bars
    ax.bar_label(
        bars,
        labels=[f"{p:.1%}" for p in probabilities],
        padding=3
    )

    ax.set_title(
        f"Sample {sample_id}: {predicted_forest} "
        f"({highest_probability:.1%} probability)"
    )
    ax.set_xlabel("Forest type")
    ax.set_ylabel("Predicted probability (%)")
    ax.set_ylim(0, max(probabilities * 100) * 1.2)
    ax.tick_params(axis="x", rotation=30)
    plt.setp(ax.get_xticklabels(), ha="right")

    fig.tight_layout()

    # Create a separate card with the sample characteristics
    details_html = f"""
    <div style="
        margin-top: 16px;
        padding: 22px;
        border: 1px solid #E5EAE7;
        border-radius: 18px;
        background: #FFFFFF;
        font-family: Inter, -apple-system, BlinkMacSystemFont,
                     'Segoe UI', sans-serif;
        color: #26352D;
    ">
        <div style="
            font-size: 17px;
            font-weight: 700;
            margin-bottom: 5px;
        ">
            Landscape characteristics
        </div>

        <div style="
            color: #7A877F;
            font-size: 13px;
            margin-bottom: 20px;
        ">
            Environmental features of sample {sample_id}
        </div>

        <div style="
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 12px;
        ">
            <div style="
                padding: 16px;
                background: #F5F8F6;
                border-radius: 13px;
            ">
                <div style="font-size: 12px; color: #78857D;">
                    ELEVATION
                </div>
                <div style="
                    font-size: 23px;
                    font-weight: 700;
                    margin-top: 7px;
                ">
                    {elevation:,} m
                </div>
            </div>

            <div style="
                padding: 16px;
                background: #F5F8F6;
                border-radius: 13px;
            ">
                <div style="font-size: 12px; color: #78857D;">
                    WILDERNESS AREA
                </div>
                <div style="
                    font-size: 16px;
                    font-weight: 650;
                    margin-top: 7px;
                ">
                    {html.escape(wilderness_name)}
                </div>
            </div>

            <div style="
                padding: 16px;
                background: #F5F8F6;
                border-radius: 13px;
            ">
                <div style="font-size: 12px; color: #78857D;">

            <div style="
                padding: 16px;
                background: #F5F8F6;
                border-radius: 13px;
            ">
                <div style="font-size: 12px; color: #78857D;">
                    SOIL TYPE
                </div>
                <div style="
                    font-size: 16px;
                    font-weight: 650;
                    margin-top: 7px;
                    line-height: 1.5;
                ">
                    {html.escape(soil_description)}
                </div>
                <div style="
                    font-size: 12px;
                    color: #78857D;
                    margin-top: 8px;
                ">
                    Category {soil_value}
                    {f" · ELU code {soil_elu_code}" if soil_elu_code is not None else ""}
                </div>
            </div>
    """

    return fig, details_html

with gr.Blocks(title="Forest Cover Type Predictor") as demo:
    gr.Markdown("# Forest Cover Type Predictor")
    gr.Markdown(
        "Select sample observations to see the predicted forest "
        "type and probabilities for all seven classes."
    )

    sample_selector = gr.Dropdown(
    choices=[str(i) for i in range(1, len(X_demo) + 1)],
    value="1",
    multiselect=False,
    label="Select one sample"
)

    predict_button = gr.Button("Predict forest cover type")
    output = gr.Plot(label="Prediction probabilities")
    details_output = gr.HTML(label="Sample characteristics")

    predict_button.click(
        fn=predict_samples,
        inputs=sample_selector,
        outputs=[output, details_output]
    )

demo.launch()