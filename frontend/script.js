const imageInput = document.getElementById("imageInput");
const emptyPreview = document.getElementById("emptyPreview");
const imagePreviewContainer = document.getElementById("imagePreviewContainer");
const imagePreview = document.getElementById("imagePreview");
const fileName = document.getElementById("fileName");
const identifyButton = document.getElementById("identifyButton");
const staticNotice = document.getElementById("staticNotice");

imageInput.addEventListener("change", () => {
    const [file] = imageInput.files;

    if (!file) return;

    imagePreview.src = URL.createObjectURL(file);
    imagePreview.onload = () => URL.revokeObjectURL(imagePreview.src);
    imagePreview.alt = `Uploaded bird: ${file.name}`;
    fileName.textContent = file.name;
    emptyPreview.classList.add("hidden");
    imagePreviewContainer.classList.remove("hidden");
    identifyButton.disabled = false;
    staticNotice.classList.add("hidden");
});

identifyButton.addEventListener("click", () => {
    staticNotice.classList.remove("hidden");
});
