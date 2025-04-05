let mediaStream = null;
let capturedImages = {
    happy: null,
    serious: null,
    body: null
};

function showCountdown(duration, callback) {
    const countdown = document.getElementById("countdown");
    let current = duration;

    countdown.innerText = current;
    countdown.style.display = "block";

    const interval = setInterval(() => {
        current--;
        if (current <= 0) {
            clearInterval(interval);
            countdown.style.display = "none";
            callback(); // run capture logic
        } else {
            countdown.innerText = current;
        }
    }, 1000);
}

async function captureImage(type) {
    const video = document.getElementById(`${type}Preview`);
    const canvas = document.getElementById(`${type}Canvas`);
    const countdown = document.getElementById("countdown");

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });

        // Confirm camera is working
        video.srcObject = stream;
        video.play();

        // Wait a tiny bit for camera to warm up
        await new Promise(resolve => setTimeout(resolve, 100));

        // ✅ Begin countdown
        let count = 1;
        countdown.innerText = count;
        countdown.style.display = "block";

        const countdownInterval = setInterval(() => {
            count--;
            if (count <= 0) {
                clearInterval(countdownInterval);
                countdown.style.display = "none";

                // 🎯 Capture the frame
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                const ctx = canvas.getContext("2d");
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                capturedImages[type] = canvas.toDataURL("image/png");

                // 🛑 Stop webcam
                stream.getTracks().forEach(track => track.stop());
                video.srcObject = null;

                alert(`${type} image captured ✅`);
            } else {
                countdown.innerText = count;
            }
        }, 1000);

    } catch (err) {
        console.error("Webcam error: ", err);
        alert("Webcam access denied or unavailable. Make sure your browser has permission.");
    }
}


function skipImage(type) {
    capturedImages[type] = null;
    document.getElementById(`${type}Preview`).srcObject = null;
    alert(`Skipped ${type} image`);
}

function analyzeImages() {
    const gender = document.querySelector('input[name="gender"]:checked').value;
    const resultDiv = document.getElementById("results");

    const hasAny = capturedImages.happy || capturedImages.serious || capturedImages.body;

    if (!hasAny) {
        resultDiv.innerHTML = "Please provide at least one image to analyze!";
        return;
    }

    // First clear previous results
    resultDiv.innerHTML = '<p>Analyzing your images... 🤔</p>';

    // Prepare the data to send
    const imageData = {
        happy: capturedImages.happy,
        serious: capturedImages.serious,
        body: capturedImages.body,
        gender: gender
    };

    // Send images to backend for analysis
    fetch('http://localhost:7000/analyze_images', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(imageData)
    })
    .then(response => response.json())
    .then(data => {
        resultDiv.innerHTML = `
            <div class="results-container">
                <p>Gender: <strong>${gender}</strong></p>
                ${data.emotion_score ? `<p>😊 Emotion Score: <strong>${data.emotion_score}%</strong></p>` : ''}
                ${data.symmetry_score ? `<p>✨ Symmetry Score: <strong>${data.symmetry_score}%</strong></p>` : ''}
                ${data.body_score ? `<p>💪 Body Score: <strong>${data.body_score}%</strong></p>` : ''}
                ${data.error_msg ? `<p class="error">${data.error_msg}</p>` : ''}
            </div>
        `;
    })
    .catch(error => {
        console.error('Error analyzing images:', error);
        resultDiv.innerHTML = '<p class="error">Failed to analyze images. Please try again.</p>';
    });
}
