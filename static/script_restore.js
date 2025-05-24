const filtersByNoise = {
  salt_pepper: [
    { value: "median", text: "Median Filter" },
    { value: "adaptive_median", text: "Adaptive Median Filter" },
  ],
  gaussian: [
    { value: "gaussian", text: "Gaussian Filter" },
    { value: "wiener", text: "Wiener Filter" },
  ],
  periodic: [
    { value: "notch", text: "Notch Filter" },
    { value: "fourier", text: "Fourier Transform Filter" },
  ],
};

const noiseSelect = document.getElementById("noiseSelect");
const filterSelect = document.getElementById("filterSelect");
const imageInput = document.getElementById("imageInput");
const imgBefore = document.getElementById("imgBefore");
const imgAfter = document.getElementById("imgAfter");
const imageWidth = document.getElementById("imageWidth");
const imageHeight = document.getElementById("imageHeight");
const mseValue = document.getElementById("mseValue");
const downloadBtn = document.getElementById("downloadBtn");

// Update pilihan filter berdasarkan noise yang dipilih
function populateFilters() {
  const noise = noiseSelect.value;
  filterSelect.innerHTML = "";
  filtersByNoise[noise].forEach((f) => {
    const option = document.createElement("option");
    option.value = f.value;
    option.textContent = f.text;
    filterSelect.appendChild(option);
  });
}
noiseSelect.addEventListener("change", populateFilters);
window.onload = populateFilters;

// Preview gambar dan ukuran saat pilih file
imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (e) => {
    imgBefore.src = e.target.result;
  };
  reader.readAsDataURL(file);

  const img = new Image();
  img.onload = () => {
    imageWidth.value = img.width + " px";
    imageHeight.value = img.height + " px";
  };
  img.src = URL.createObjectURL(file);

  // Reset hasil restore, mse, tombol download saat ganti file
  imgAfter.src = "";
  mseValue.textContent = "";
  downloadBtn.style.display = "none";
});

// Fungsi upload dan request restorasi noise
function uploadImage() {
  if (!imageInput.files.length) {
    alert("Silakan pilih gambar terlebih dahulu.");
    return;
  }
  const noiseType = noiseSelect.value;
  const filterType = filterSelect.value;
  const file = imageInput.files[0];

  const formData = new FormData();
  formData.append("image", file);
  formData.append("noise_type", noiseType);
  formData.append("filter_type", filterType);

  fetch("/upload", {
    method: "POST",
    body: formData,
  })
    .then((res) => res.json())
    .then((data) => {
      if (data.error) {
        alert("Error: " + data.error);
        return;
      }
      imgAfter.src = data.filtered + "?t=" + Date.now();
      document.getElementById("mseValue").value = data.mse.toFixed(4);

      downloadBtn.style.display = "inline-block";
      downloadBtn.onclick = () => {
        window.location.href =
          "/download/restore/" + data.filtered.split("/").pop();
      };
    })
    .catch(() => alert("Terjadi kesalahan saat upload."));
}
