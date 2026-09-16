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

console.log("Script loaded");
console.log("uploadInput:", uploadInput);
console.log("imagePreview:", imagePreview);
console.log("emptyPreview:", emptyPreview);
console.log("imagePreviewContainer:", imagePreviewContainer);

let selectedFile = null;


// ============================================================
// IMAGE SELECTION
// ============================================================

if (uploadInput) {
    uploadInput.addEventListener("change", function () {
        console.log("File input change event fired");
        
        const file = this.files[0];
        console.log("Selected file:", file);

        if (!file) {
            return;
        }

        if (!file.type.startsWith("image/")) {
            alert("Please select an image file.");
            return;
        }

        selectedFile = file;
        console.log("Processing file:", file.name);

        const reader = new FileReader();

        reader.onload = function (event) {
            console.log("FileReader onload triggered");
            console.log("Image preview element:", imagePreview);
            
            imagePreview.src = event.target.result;
            fileName.textContent = file.name;

            console.log("Adding hidden to emptyPreview");
            emptyPreview.classList.add("hidden");
            
            console.log("Removing hidden from imagePreviewContainer");
            imagePreviewContainer.classList.remove("hidden");

            console.log("Current classes:", imagePreviewContainer.className);
        };

        reader.readAsDataURL(file);
    });
} else {
    console.error("uploadInput element not found!");
}
