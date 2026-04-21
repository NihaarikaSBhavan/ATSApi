const analyzeButton = document.getElementById("analyzeButton");
const resumeText = document.getElementById("resumeText");
const jobText = document.getElementById("jobText");
const resultBox = document.getElementById("resultBox");

analyzeButton.addEventListener("click", async () => {
  resultBox.textContent = "Analyzing...";

  try {
    const response = await fetch("http://127.0.0.1:8000/api/v1/ats/evaluate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        resume_text: resumeText.value,
        job_text: jobText.value,
      }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || `Request failed with ${response.status}`);
    }

    resultBox.textContent = JSON.stringify(data, null, 2);
  } catch (error) {
    resultBox.textContent = `Request failed: ${error.message}`;
  }
});
