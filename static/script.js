// ===============================
// DOM READY
// ===============================
document.addEventListener("DOMContentLoaded", () => {

    // -------------------------------
    // DOM ELEMENTS
    // -------------------------------
    const modelSelect = document.getElementById("modelSelect");
    const layerSelect = document.getElementById("layerSelect");
    const featureGrid = document.getElementById("featureGrid");
    const runBtn = document.getElementById("runBtn");
    const metricSelect = document.getElementById("compareMetric");
    const featureCount = document.getElementById("featureCount");

    const accBox = document.getElementById("accBox");
    const precBox = document.getElementById("precBox");
    const recBox = document.getElementById("recBox");
    const f1Box = document.getElementById("f1Box");
    const cmImage = document.getElementById("cmImage");

    const zoomModal = document.getElementById("zoomModal");
    const zoomImg = document.getElementById("zoomImg");
    const zoomClose = document.querySelector(".zoom-close");

    const compareLayerSelect = document.getElementById("compareLayerSelect");
    const compareGrid = document.getElementById("compareGrid");
    const compareBtn = document.getElementById("compareBtn");

    const imageUpload = document.getElementById("imageUpload");
    const downloadBtn = document.getElementById("downloadBtn");

    let IMAGE_PATH = "renaissance.jpg";
    let compareChart = null;

    // ✅ CACHE TO PREVENT RELOADING LAYERS
    const layerCache = {};

    // -------------------------------
    // IMAGE UPLOAD
    // -------------------------------
    if (imageUpload) {
        imageUpload.addEventListener("change", e => {
            const file = e.target.files[0];
            if (!file) return;

            const formData = new FormData();
            formData.append("image", file);

            fetch("/upload_image", {
                method: "POST",
                body: formData
            })
            .then(res => res.json())
            .then(data => {
                IMAGE_PATH = data.path;
            });
        });
    }

    // -------------------------------
    // DOWNLOAD FEATURE MAPS
    // -------------------------------
    if (downloadBtn) {
        downloadBtn.onclick = () => {
            if (!window.LAST_RUN_ID) {
                alert("Run visualization first");
                return;
            }
            window.location.href = `/download_features?run_id=${window.LAST_RUN_ID}`;
        };
    }

    // ===============================
    // LOAD LAYERS (FAST + SAFE)
    // ===============================
    async function loadLayers() {
        const model = modelSelect.value;

        layerSelect.disabled = true;
        compareLayerSelect.disabled = true;

        layerSelect.innerHTML = "<option>Loading…</option>";
        compareLayerSelect.innerHTML = "<option>Loading…</option>";

        // Serve instantly from cache
        if (layerCache[model]) {
            populateLayers(layerCache[model]);
            return;
        }

        const res = await fetch(`/layers?model=${model}`);
        const layers = await res.json();

        layerCache[model] = layers;
        populateLayers(layers);
    }

    function populateLayers(layers) {
        layerSelect.innerHTML = "";
        compareLayerSelect.innerHTML = "";

        layers.forEach(layer => {
            const o1 = document.createElement("option");
            o1.value = layer;
            o1.textContent = layer;
            layerSelect.appendChild(o1);

            const o2 = document.createElement("option");
            o2.value = layer;
            o2.textContent = layer;
            compareLayerSelect.appendChild(o2);
        });

        layerSelect.disabled = false;
        compareLayerSelect.disabled = false;
    }

    // ===============================
    // ARCHITECTURE
    // ===============================
    async function loadArchitecture() {
        const res = await fetch(`/architecture?model=${modelSelect.value}`);
        const data = await res.json();
        document.getElementById("architectureBox").textContent = data.architecture;
    }

    // ===============================
    // METRICS
    // ===============================
    async function loadMetrics() {
        const res = await fetch("/metrics");
        const data = await res.json();

        accBox.textContent = `Accuracy: ${data.accuracy.toFixed(3)}`;
        precBox.textContent = `Precision: ${data.precision.toFixed(3)}`;
        recBox.textContent = `Recall: ${data.recall.toFixed(3)}`;
        f1Box.textContent = `F1: ${data.f1.toFixed(3)}`;

        cmImage.src = `/outputs/${data.confusion_matrix}?${Date.now()}`;
        cmImage.onclick = () => openZoom(cmImage.src);
    }

    // ===============================
    // HISTOGRAM (CRITICAL FIX)
    // ===============================
    async function loadComparison() {
        if (!metricSelect) return;

        const res = await fetch("/compare_metrics", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image_path: IMAGE_PATH })
        });

        const data = await res.json();
        const metric = metricSelect.value;

        const labels = Object.keys(data);
        const values = labels.map(m => data[m][metric]);

        const canvas = document.getElementById("comparisonChart");

        // 🔴 CRITICAL: lock canvas height from parent
        canvas.height = canvas.parentElement.clientHeight;

        const ctx = canvas.getContext("2d");

        if (compareChart) {
            compareChart.destroy();
            compareChart = null;
        }

        compareChart = new Chart(ctx, {
            type: "bar",
            data: {
                labels,
                datasets: [{
                    label: metric.toUpperCase(),
                    data: values,
                    backgroundColor: "#beeb1c"
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                scales: {
                    y: { min: 0, max: 1 }
                }
            }
        });
    }

    // ===============================
    // RENDER IMAGES
    // ===============================
    function renderImages(images, grid) {
        grid.innerHTML = "";

        images.forEach(file => {
            const img = document.createElement("img");
            img.src = `/outputs/${file}?${Date.now()}`;
            img.style.cursor = "zoom-in";

            img.addEventListener("click", () => {
                zoomImg.src = img.src;
                zoomModal.style.display = "flex";
            });

            grid.appendChild(img);
        });
    }



    // ===============================
    // RUN MAIN VIS
    // ===============================
    runBtn.onclick = async () => {
        const payload = {
            model: modelSelect.value,
            layer_name: layerSelect.value,
            num_features: featureCount.value,
            image_path: IMAGE_PATH
        };

        const res = await fetch("/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        const data = await res.json();
        renderImages(data.images, featureGrid);
        window.LAST_RUN_ID = data.run_id;

        loadMetrics();
        loadComparison();
        loadArchitecture();
    };

    // ===============================
    // COMPARE LAYERS
    // ===============================
    compareBtn.onclick = async () => {
        const payload = {
            model: modelSelect.value,
            layer_name: compareLayerSelect.value,
            num_features: featureCount.value,
            image_path: IMAGE_PATH
        };

        const res = await fetch("/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        const data = await res.json();
        renderImages(data.images, compareGrid);
    };

    // ===============================
    // ZOOM MODAL
    // ===============================
    function openZoom(src) {
        zoomImg.src = src;
        zoomModal.style.display = "flex";
    }

    zoomClose.onclick = () => zoomModal.style.display = "none";

    zoomModal.onclick = e => {
        if (e.target === zoomModal) zoomModal.style.display = "none";
    };

    // ===============================
    // EVENTS
    // ===============================
    metricSelect.addEventListener("change", loadComparison);

    modelSelect.addEventListener("change", () => {
        loadLayers();
        loadArchitecture();
    });

    document.addEventListener("fullscreenchange", () => {
        if (compareChart) {
            compareChart.resize();
        }
    });

    // ===============================
    // INIT
    // ===============================
    loadLayers();
    loadArchitecture();
    loadComparison();
});
