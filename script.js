const uploadInput = document.getElementById("imageInput");
const imagePreviewContainer = document.getElementById("imagePreviewContainer");
const imagePreview = document.getElementById("imagePreview");
const emptyPreview = document.getElementById("emptyPreview");
const predictionBox = document.getElementById("predictionBox");
const predictionName = document.getElementById("predictionName");
const confidenceValue = document.getElementById("confidenceValue");
const confidenceText = document.getElementById("confidenceText");
const confidenceFill = document.getElementById("confidenceFill");
const fileName = document.getElementById("fileName");

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

        imagePreview.src = event.target.result;
        fileName.textContent = file.name;

        emptyPreview.classList.add("hidden");
        imagePreviewContainer.classList.remove("hidden");

    };

    reader.readAsDataURL(file);
});
