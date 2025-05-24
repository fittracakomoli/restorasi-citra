const noiseSelect = document.getElementById("noiseSelect");
const imageInput = document.getElementById("imageInput");
const imgOriginal = document.getElementById("imgOriginal");
const imageWidth = document.getElementById("imageWidth");
const imageHeight = document.getElementById("imageHeight");
const downloadBtn = document.getElementById("downloadBtn");
const imgNoisy = document.getElementById("imgNoisy");

// Preview gambar asli dan tampilkan ukuran saat pilih file
imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (e) => {
    imgOriginal.src = e.target.result;
  };
  reader.readAsDataURL(file);

  const img = new Image();
  img.onload = () => {
    imageWidth.value = img.width + " px";
    imageHeight.value = img.height + " px";
  };
  img.src = URL.createObjectURL(file);

  // Reset hasil noise dan tombol download saat ganti file
  imgNoisy.src = "";
  downloadBtn.style.display = "none";
});

// Fungsi upload dan request tambah noise
function uploadNoise() {
  if (!imageInput.files.length) {
    alert("Silakan pilih gambar terlebih dahulu.");
    return;
  }
  const noiseType = noiseSelect.value;
  const file = imageInput.files[0];

  const formData = new FormData();
  formData.append("image", file);
  formData.append("noise_type", noiseType);

  fetch("/add_noise", {
    method: "POST",
    body: formData,
  })
    .then((res) => res.json())
    .then((data) => {
      if (data.error) {
        alert("Error: " + data.error);
        return;
      }
      imgNoisy.src = data.noisy + "?t=" + Date.now();

      downloadBtn.style.display = "inline-block";
      downloadBtn.onclick = () => {
        window.location.href = "/download/noise/" + data.noisy.split("/").pop();
      };
    })
    .catch(() => alert("Terjadi kesalahan saat upload."));
}
