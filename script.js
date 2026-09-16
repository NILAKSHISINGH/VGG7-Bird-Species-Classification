const uploadInput = document.getElementById("imageUpload");
const preview = document.getElementById("preview");
const uploadArea = document.getElementById("uploadArea");
const predictButton = document.getElementById("predictButton");

const resultSection = document.getElementById("resultSection");
const speciesName = document.getElementById("speciesName");
const confidence = document.getElementById("confidence");
const resultBar = document.getElementById("resultBar");
const predictionsList = document.getElementById("predictionsList");
const loading = document.getElementById("loading");

let selectedFile = null;


// ============================================================
// IMAGE SELECTION
// ============================================================

uploadInput.addEventListener("change", function () {

    const file = this.files[0];

    if (!file) {
        return;
    }

    if (!file.type.startsWith("image/")) {
        alert("Please select an image file.");
        return;
    }

    selectedFile = file;

    const reader = new FileReader();

    reader.onload = function (event) {

        preview.src = event.target.result;
        preview.style.display = "block";

        uploadArea.classList.add("has-image");

        predictButton.disabled = false;
    };

    reader.readAsDataURL(file);
});


// ============================================================
// PREDICT BUTTON
// ============================================================

predictButton.addEventListener("click", async function () {

    if (!selectedFile) {
        alert("Please upload a bird image first.");
        return;
    }

    const formData = new FormData();

    formData.append("image", selectedFile);


    // Show loading
    loading.style.display = "block";
    resultSection.style.display = "none";

    predictButton.disabled = true;
    predictButton.textContent = "Analyzing...";


    try {

        const response = await fetch(
            "http://127.0.0.1:5000/predict",
            {
                method: "POST",
                body: formData
            }
        );


        const data = await response.json();


        if (!response.ok || !data.success) {

            throw new Error(
                data.error || "Prediction failed."
            );
        }


        // ====================================================
        // MAIN PREDICTION
        // ====================================================

        speciesName.textContent = formatSpecies(
            data.prediction
        );

        confidence.textContent =
            data.confidence.toFixed(2) + "% confidence";


        resultBar.style.width =
            data.confidence + "%";


        // ====================================================
        // TOP 3 PREDICTIONS
        // ====================================================

        predictionsList.innerHTML = "";


        data.top_predictions.forEach(
            function (prediction, index) {

                const row = document.createElement("div");

                row.className = "prediction-row";

                row.innerHTML = `
                    <div class="prediction-info">
                        <span>
                            ${index + 1}. 
                            ${formatSpecies(prediction.species)}
                        </span>

                        <span>
                            ${prediction.confidence.toFixed(2)}%
                        </span>
                    </div>

                    <div class="prediction-bar">
                        <div
                            class="prediction-fill"
                            style="width:${prediction.confidence}%"
                        ></div>
                    </div>
                `;

                predictionsList.appendChild(row);
            }
        );


        resultSection.style.display = "block";


    } catch (error) {

        console.error(error);

        alert(
            "Could not connect to the VGG7 backend.\n\n" +
            error.message
        );

    } finally {

        loading.style.display = "none";

        predictButton.disabled = false;

        predictButton.textContent = "Identify Bird";
    }
});


// ============================================================
// FORMAT SPECIES NAME
// ============================================================

function formatSpecies(name) {

    return name
        .toLowerCase()
        .replace(/\b\w/g, function (letter) {
            return letter.toUpperCase();
        });
}