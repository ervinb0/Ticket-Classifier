// script.js
document.addEventListener("DOMContentLoaded", function () {
    const predictBtn = document.getElementById("predict-btn");
    const confirmBtn = document.getElementById("confirm-btn");
    const titleInput = document.getElementById("issue-title");
    const bodyInput = document.getElementById("issue-body");
    const sidebar = document.getElementById("sidebar");
    let predictedLabel = "";
    let selectedLabel = "";
    let rawTitle = "";
    let rawBody = "";

    predictBtn.addEventListener("click", function () {
        const title = titleInput.value.trim();
        const body = bodyInput.value.trim();

        if (!title && !body) {
            alert("Please enter a title or body.");
            return;
        }

        fetch("http://127.0.0.1:5000/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ title, body })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert(data.error);
                return;
            }

            predictedLabel = data.prediction;
            selectedLabel = predictedLabel;
            rawTitle = data.title;
            rawBody = data.body;

            updateSidebar(data.probabilities);
        })
        .catch(error => console.error("Error:", error));
    });

    function updateSidebar(probabilities) {
        sidebar.innerHTML = "";
        sidebar.style.display = "block";
        const labels = ["bug", "enhancement", "question"];

        labels.forEach(label => {
            const button = document.createElement("button");
            button.textContent = `${label} (${(probabilities[label] * 100).toFixed(1)}%)`;
            button.className = `label-btn ${label === predictedLabel ? "selected" : ""}`;
            button.style.backgroundColor = getColor(probabilities[label]);
            button.addEventListener("click", () => {
                document.querySelectorAll(".label-btn").forEach(btn => btn.classList.remove("selected"));
                button.classList.add("selected");
                selectedLabel = label;
            });
            sidebar.appendChild(button);
        });

        confirmBtn.style.display = "block";
    }

    confirmBtn.addEventListener("click", function () {
        fetch("http://127.0.0.1:5000/api/correct", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                title: rawTitle,
                body: rawBody,
                predicted_label: predictedLabel,
                corrected_label: selectedLabel
            })
        })
        .then(response => response.json())
        .then(data => {
            alert("Label confirmed/corrected successfully.");
            sidebar.style.display = "none";
            titleInput.value = "";
            bodyInput.value = "";
        })
        .catch(error => console.error("Error:", error));
    });

    function getColor(probability) {
        if (probability > 0.7) return "green";
        if (probability > 0.4) return "yellow";
        return "red";
    }
});
