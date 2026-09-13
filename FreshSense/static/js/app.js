/* ================================================================
   FreshSense — Client Logic
   Handles drag-and-drop upload, AJAX prediction, animated probability bars,
   and dynamic history rendering.
   ================================================================ */

(function () {
    "use strict";

    // ---- DOM Elements ----
    const uploadZone    = document.getElementById("uploadZone");
    const uploadContent = document.getElementById("uploadContent");
    const uploadPreview = document.getElementById("uploadPreview");
    const previewImage  = document.getElementById("previewImage");
    const removeBtn     = document.getElementById("removeImage");
    const fileInput     = document.getElementById("fileInput");

    const resultEmpty   = document.getElementById("resultEmpty");
    const resultLoading = document.getElementById("resultLoading");
    const resultCard    = document.getElementById("resultCard");
    const resultBadge   = document.getElementById("resultBadge");
    const resultConf    = document.getElementById("resultConfidence");
    const resultDesc    = document.getElementById("resultDescription");
    const probChart     = document.getElementById("probChart");
    const analyzeBtn    = document.getElementById("analyzeAnother");

    const historyGrid   = document.getElementById("historyGrid");
    const historyEmpty  = document.getElementById("historyEmpty");
    const clearBtn      = document.getElementById("clearHistory");

    let selectedFile = null;

    // ---- Badge and Bar CSS Class Mapping ----
    const CLASS_KEYS = {
        "Fresh":         "fresh",
        "Slightly Aged": "slightly-aged",
        "Stale":         "stale",
        "Spoiled":       "spoiled",
        "Rotten":        "rotten"
    };

    function badgeClass(cls) {
        return "badge-" + (CLASS_KEYS[cls] || "fresh");
    }

    function barClass(cls) {
        return "bar-" + (CLASS_KEYS[cls] || "fresh");
    }

    // ---- Upload Zone Interactions ----
    uploadZone.addEventListener("click", function (e) {
        if (e.target === removeBtn || removeBtn.contains(e.target)) return;
        fileInput.click();
    });

    fileInput.addEventListener("change", function () {
        if (fileInput.files.length) {
            handleFile(fileInput.files[0]);
        }
    });

    // Drag & drop handlers
    ["dragenter", "dragover"].forEach(function (evt) {
        uploadZone.addEventListener(evt, function (e) {
            e.preventDefault();
            e.stopPropagation();
            uploadZone.classList.add("dragover");
        });
    });

    ["dragleave", "drop"].forEach(function (evt) {
        uploadZone.addEventListener(evt, function (e) {
            e.preventDefault();
            e.stopPropagation();
            uploadZone.classList.remove("dragover");
        });
    });

    uploadZone.addEventListener("drop", function (e) {
        e.preventDefault();
        e.stopPropagation();
        if (e.dataTransfer.files && e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    function handleFile(file) {
        const validTypes = ["image/jpeg", "image/png", "image/webp"];
        if (validTypes.indexOf(file.type) === -1 && !/\.(jpe?g|png|webp)$/i.test(file.name)) {
            alert("Please upload a valid JPG, PNG, or WebP image.");
            return;
        }
        selectedFile = file;
        showPreview(file);
        uploadImage(file);
    }

    function showPreview(file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            previewImage.src = e.target.result;
            uploadContent.style.display = "none";
            uploadPreview.style.display = "flex";
        };
        reader.readAsDataURL(file);
    }

    removeBtn.addEventListener("click", function (e) {
        e.stopPropagation();
        resetUpload();
    });

    function resetUpload() {
        selectedFile = null;
        fileInput.value = "";
        previewImage.src = "";
        uploadContent.style.display = "block";
        uploadPreview.style.display = "none";
        showResult("empty");
    }

    // ---- Predict Request ----
    function uploadImage(file) {
        showResult("loading");

        const formData = new FormData();
        formData.append("file", file);

        fetch("/predict", {
            method: "POST",
            body: formData
        })
        .then(function (res) {
            return res.json().then(function (data) {
                if (!res.ok) {
                    throw new Error(data.error || ("Server error (" + res.status + ")"));
                }
                return data;
            });
        })
        .then(function (data) {
            renderResult(data);
            showResult("card");
            loadHistory();
        })
        .catch(function (err) {
            alert("Error: " + err.message);
            showResult("empty");
        });
    }

    function showResult(state) {
        resultEmpty.style.display   = state === "empty"   ? "block" : "none";
        resultLoading.style.display = state === "loading" ? "block" : "none";
        resultCard.style.display    = state === "card"    ? "block" : "none";
    }

    function renderResult(data) {
        resultBadge.textContent = data.predicted_class;
        resultBadge.className = "result-tag " + badgeClass(data.predicted_class);
        resultConf.textContent = data.confidence + "% Confidence";
        resultDesc.textContent = data.description;

        // Populate Probability Bars
        probChart.innerHTML = "";
        const classOrder = ["Fresh", "Slightly Aged", "Stale", "Spoiled", "Rotten"];
        classOrder.forEach(function (cls) {
            const pct = data.probabilities[cls] || 0;
            const row = document.createElement("div");
            row.className = "prob-row";

            row.innerHTML =
                '<span class="prob-label">' + cls + '</span>' +
                '<div class="prob-bar-track">' +
                    '<div class="prob-bar-fill ' + barClass(cls) + '"></div>' +
                '</div>' +
                '<span class="prob-value">' + pct.toFixed(1) + '%</span>';

            probChart.appendChild(row);

            // Trigger animation frame for CSS transition
            requestAnimationFrame(function () {
                requestAnimationFrame(function () {
                    const fill = row.querySelector(".prob-bar-fill");
                    if (fill) {
                        fill.style.width = pct + "%";
                    }
                });
            });
        });
    }

    analyzeBtn.addEventListener("click", function () {
        resetUpload();
    });

    // ---- History Management ----
    function loadHistory() {
        fetch("/history")
            .then(function (res) { return res.json(); })
            .then(function (items) { renderHistory(items); })
            .catch(function () {});
    }

    function renderHistory(items) {
        if (!items || !items.length) {
            historyEmpty.style.display = "block";
            historyGrid.innerHTML = "";
            clearBtn.style.display = "none";
            return;
        }

        historyEmpty.style.display = "none";
        clearBtn.style.display = "inline-flex";

        historyGrid.innerHTML = items.map(function (item) {
            return (
                '<div class="history-card">' +
                    '<img class="history-img" src="' + item.image_url + '" alt="' + item.filename + '">' +
                    '<div class="history-body">' +
                        '<div class="history-top">' +
                            '<span class="history-badge ' + badgeClass(item.predicted_class) + '">' +
                                item.predicted_class +
                            '</span>' +
                            '<span class="history-conf">' + item.confidence + '%</span>' +
                        '</div>' +
                        '<div class="history-filename" title="' + item.filename + '">' +
                            item.filename +
                        '</div>' +
                        '<div class="history-time">' + item.timestamp + '</div>' +
                    '</div>' +
                '</div>'
            );
        }).join("");
    }

    clearBtn.addEventListener("click", function () {
        if (!confirm("Are you sure you want to clear all scan records?")) return;
        fetch("/history", { method: "DELETE" })
            .then(function () { loadHistory(); })
            .catch(function (err) {
                alert("Failed to clear history: " + err.message);
            });
    });

    // Initialize history on load
    loadHistory();
})();
