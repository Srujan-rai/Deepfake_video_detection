document.getElementById("upload-form").addEventListener("submit", async (e) => {
    e.preventDefault();

    const realImageInput = document.getElementById("real-image");
    const videoInput = document.getElementById("video-file");

    if (!realImageInput.files[0] || !videoInput.files[0]) {
        alert("Please upload both a real image and a video.");
        return;
    }

    const formData = new FormData();
    formData.append("real_image", realImageInput.files[0]);
    formData.append("video_file", videoInput.files[0]);

    document.getElementById("loading").style.display = "block";
    document.getElementById("output").style.display = "none";

    try {
        const response = await fetch("/upload", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            throw new Error("Failed to process the video and image.");
        }

        const data = await response.json();
        renderChart(data.similarities_to_real, data.similarities_to_prev);
        displayProcessedVideo(data.processed_video_url);
    } catch (error) {
        alert(error.message);
    } finally {
        document.getElementById("loading").style.display = "none";
    }
});

function renderChart(similaritiesToReal, similaritiesToPrev) {
    const ctx = document.getElementById("result-chart").getContext("2d");
    document.getElementById("output").style.display = "block";

    new Chart(ctx, {
        type: "line",
        data: {
            labels: Array.from({ length: similaritiesToReal.length }, (_, i) => i + 1),
            datasets: [
                {
                    label: "Similarity to Real Image",
                    data: similaritiesToReal,
                    borderColor: "blue",
                    fill: false,
                },
                {
                    label: "Similarity to Previous Frame",
                    data: similaritiesToPrev,
                    borderColor: "red",
                    fill: false,
                },
            ],
        },
        options: {
            responsive: true,
            scales: {
                x: { title: { display: true, text: "Frame Index" } },
                y: { title: { display: true, text: "Similarity Score" } },
            },
        },
    });
}

function displayProcessedVideo(videoUrl) {
    const videoElement = document.getElementById("processed-video");
    videoElement.src = videoUrl;
}
