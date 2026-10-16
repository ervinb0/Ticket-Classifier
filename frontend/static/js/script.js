// script.js
document.addEventListener("DOMContentLoaded", function () {
    const predictBtn = document.getElementById("predict-btn");
    const confirmBtn = document.getElementById("confirm-btn");
    const titleInput = document.getElementById("issue-title");
    const bodyInput = document.getElementById("issue-body");
    const result = document.getElementById("result");
    const labelsContainer = document.getElementById("labels");
    const message = document.getElementById("message");
    const LABELS = ["bug", "enhancement", "question"];
    let predictionId = null;
    let selectedLabel = "";

    function showMessage(text, isError) {
        message.textContent = text;
        message.className = isError ? "error" : "success";
    }

    function hideMessage() {
        message.className = "hidden";
    }

    async function postJson(url, payload) {
        const response = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || `Request failed (${response.status})`);
        }
        return data;
    }

    predictBtn.addEventListener("click", async function () {
        const title = titleInput.value.trim();
        const body = bodyInput.value.trim();
        hideMessage();

        if (!title && !body) {
            showMessage("Please enter a title or body.", true);
            return;
        }

        predictBtn.disabled = true;
        try {
            const data = await postJson("api/predict", { title, body });
            predictionId = data.id;
            selectedLabel = data.predicted_label;
            renderLabels(data.probabilities);
        } catch (error) {
            result.className = "hidden";
            showMessage(error.message, true);
        } finally {
            predictBtn.disabled = false;
        }
    });

    function renderLabels(probabilities) {
        labelsContainer.innerHTML = "";

        LABELS.forEach(label => {
            const box = document.createElement("button");
            box.type = "button";
            box.textContent = `${label} (${(probabilities[label] * 100).toFixed(1)}%)`;
            box.className = `label-box ${label === selectedLabel ? "selected" : ""}`;
            box.addEventListener("click", () => {
                labelsContainer.querySelectorAll(".label-box").forEach(b => b.classList.remove("selected"));
                box.classList.add("selected");
                selectedLabel = label;
            });
            labelsContainer.appendChild(box);
        });

        result.className = "";
    }

    confirmBtn.addEventListener("click", async function () {
        confirmBtn.disabled = true;
        try {
            await postJson("api/correct", { id: predictionId, corrected_label: selectedLabel });
            showMessage(`Saved label "${selectedLabel}". Thanks!`, false);
            result.className = "hidden";
            titleInput.value = "";
            bodyInput.value = "";
        } catch (error) {
            showMessage(error.message, true);
        } finally {
            confirmBtn.disabled = false;
        }
    });
});
