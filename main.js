const labels = [...'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'SPACE'];
let model = null;
let running = false;
let showMesh = true;

const video = document.getElementById("input_video");
const canvas = document.getElementById("output_canvas");
const ctx = canvas.getContext("2d");
const status = document.getElementById("status");
const char = document.querySelector('.character');
const outputText = document.getElementById("outputText");
const startBtn = document.getElementById("startBtn");
const speechToTextBtn = document.getElementById("speechToTextBtn");
const textToSpeechBtn = document.getElementById("textToSpeechBtn");
const clearBtn = document.getElementById("clearBtn");
const copyBtn = document.getElementById("copyBtn");
const toggleMeshBtn = document.getElementById("toggleMeshBtn");
const imageBtn = document.getElementById("imageBtn");

const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [5, 9], [9, 10], [10, 11], [11, 12],
    [9, 13], [13, 14], [14, 15], [15, 16],
    [13, 17], [17, 18], [18, 19], [19, 20],
    [0, 17]
];

let lastLetter = '';
let startTime = null;
const HOLD_TIME = 1500;

const hands = new Hands({
    locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
});

hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.7,
    minTrackingConfidence: 0.7,
});

const camera = new Camera(video, {
    onFrame: async () => running && await hands.send({ image: video }),
    width: 640,
    height: 480,
});

function flatten(landmarks) {
    return landmarks.flatMap(pt => [pt.x, pt.y, pt.z]);
}

function allFingersOpen(landmarks) {
    const thumbOpen = landmarks[4].x > landmarks[3].x;
    const indexOpen = landmarks[8].y < landmarks[6].y;
    const middleOpen = landmarks[12].y < landmarks[10].y;
    const ringOpen = landmarks[16].y < landmarks[14].y;
    const pinkyOpen = landmarks[20].y < landmarks[18].y;

    return thumbOpen && indexOpen && middleOpen && ringOpen && pinkyOpen;
}

function isThumbsUp(landmarks) {
    return (
        landmarks[4].y  < landmarks[3].y && // Thumb up
        landmarks[8].y  > landmarks[6].y && // Index down
        landmarks[8].x  < landmarks[6].x && // Index down
        landmarks[12].y > landmarks[10].y && // Middle down
        landmarks[12].x < landmarks[10].x && // Middle down
        landmarks[16].y > landmarks[14].y && // Ring down
        landmarks[16].x < landmarks[14].x && // Ring down
        landmarks[20].y > landmarks[18].y &&// Pinky down
        landmarks[20].x < landmarks[18].x // Pinky down
    );
}

async function onResults(results) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    if (results.multiHandLandmarks.length > 0) {
        const landmarks = results.multiHandLandmarks[0];
        
        if (showMesh) {
            ctx.strokeStyle = "#2ecc71";
            ctx.lineWidth = 3;
            for (const [i, j] of HAND_CONNECTIONS) {
                const start = landmarks[i];
                const end = landmarks[j];
                ctx.beginPath();
                ctx.moveTo(start.x * canvas.width, start.y * canvas.height);
                ctx.lineTo(end.x * canvas.width, end.y * canvas.height);
                ctx.stroke();
            }
            ctx.fillStyle = "#4a90e2";
            for (const point of landmarks) {
                ctx.beginPath();
                ctx.arc(point.x * canvas.width, point.y * canvas.height, 5, 0, Math.PI * 2);
                ctx.fill();
            }
        }

        // SPACE - all fingers open
        if (allFingersOpen(landmarks)) {
            status.textContent = `Detected: SPACE`;
            char.textContent = ' - ';
            if (!startTime) {
                startTime = Date.now();
            } else if (Date.now() - startTime > HOLD_TIME) {
                outputText.value += ' , ';
                startTime = null;
                lastLetter = '';
            }
            return;
        }
        
        // THUMBS UP - speak the text
        if (isThumbsUp(landmarks)) {
            status.textContent = `Detected: Thumbs Up`;
            char.textContent = '👍';
            if (!startTime) {
                startTime = Date.now();
            } else if (Date.now() - startTime > HOLD_TIME) {
                const text = outputText.value.trim();
                if (text) {
                    const utterance = new SpeechSynthesisUtterance(text);
                    utterance.lang = "en-US";
                    utterance.rate = 0.8; // slow speech
                    // window.speechSynthesis.cancel();
                    window.speechSynthesis.speak(utterance);
                }
                startTime = null;
                lastLetter = '';
            }
            return;
        }

        // Prediction using model
        if (model && running) {
            const input = tf.tensor2d([flatten(landmarks)]);
            const prediction = model.predict(input);
            const data = await prediction.data();
            const index = data.indexOf(Math.max(...data));
            const confidence = data[index];
            const currentLetter = labels[index];
            
            status.textContent = `Detected: ${currentLetter} (${(confidence * 100).toFixed(1)}%)`;
            char.textContent = `${currentLetter}`;
            
            if (currentLetter === lastLetter) {
                if (!startTime) {
                    startTime = Date.now();
                } else if (Date.now() - startTime > HOLD_TIME) {
                    outputText.value += (currentLetter === 'SPACE') ? ' - ' : currentLetter.toLowerCase();
                    startTime = null;
                    lastLetter = '';
                }
            } else {
                lastLetter = currentLetter;
                startTime = null;
            }
        }
    }
} 

async function loadModel() {
    try {
        model = await tf.loadLayersModel("model/asl_model.json");
        status.textContent = "Model loaded! Click Start to begin.";
        startBtn.disabled = false;
    } catch (error) {
        status.textContent = "Error loading model. Please refresh.";
        console.error("Model loading error:", error);
    }
}

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const recognition = new SpeechRecognition();
recognition.continuous = true;
recognition.interimResults = true;

recognition.onresult = (event) => {
    const transcript = Array.from(event.results)
    .map(result => result[0].transcript)
    .join('');
    outputText.value = transcript;
};

hands.onResults(onResults);

startBtn.addEventListener("click", () => {
    if (!model) {
        status.textContent = "Loading model...";
        loadModel().then(() => {
            camera.start();
            running = true;
            startBtn.textContent = "Stop Recognition";
            startBtn.classList.add("success");
        });
    } else {
        if (running) {
            running = false;
            camera.stop();
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            startBtn.textContent = "Start Recognition";
            startBtn.classList.remove("success");
        } else {
            running = true;
            camera.start();
            startBtn.textContent = "Stop Recognition";
            startBtn.classList.add("success");
        }
    }
});

speechToTextBtn.addEventListener("click", () => {
    if (speechToTextBtn.classList.contains("success")) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        recognition.stop();
        speechToTextBtn.classList.remove("success");
        speechToTextBtn.innerHTML = `<i class="fa-solid fa-microphone-lines"></i>`
    } else {
        recognition.start();
        speechToTextBtn.classList.add("success");
        speechToTextBtn.innerHTML = `<i class="fa-solid fa-microphone-lines-slash"></i>`
    }
});

imageBtn.addEventListener('mouseenter',()=>{
    imageBtn.classList.add('success')
})

imageBtn.addEventListener('mouseleave',()=>{
    imageBtn.classList.remove('success')
})


textToSpeechBtn.addEventListener("click", async () => {
    const text = outputText.value.trim();
    if (text) {
        textToSpeechBtn.classList.add('success')
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = "en-US";
        utterance.rate = 0.8; // Slow speech
        // window.speechSynthesis.cancel();
        window.speechSynthesis.speak(utterance);
        utterance.onend = () => {
            textToSpeechBtn.classList.remove('success')
        };
    }
});

clearBtn.addEventListener("click", () => {
    outputText.value = "";
});

clearBtn.addEventListener('mouseenter',()=>{
    clearBtn.classList.add('del')
})

clearBtn.addEventListener('mouseleave',()=>{
    clearBtn.classList.remove('del')
})

const toggleTheme = document.getElementById("toggleTheme");

toggleTheme.addEventListener("click", () => {
    document.body.classList.toggle("light-theme");
    let dark = document.body.classList.contains('light-theme')
    toggleTheme.innerHTML = dark? '<i class="fa-solid fa-moon"></i>' : `<i class="fa-solid fa-sun"></i>`
});

toggleTheme.addEventListener('mouseenter',()=>{
    toggleTheme.classList.add('success')
})

toggleTheme.addEventListener('mouseleave',()=>{
    toggleTheme.classList.remove('success')
})

copyBtn.addEventListener("click", () => {
    outputText.select();
    document.execCommand("copy");
    const originalText = copyBtn.textContent;
    copyBtn.innerHTML = "Copied!";
    setTimeout(() => {
        copyBtn.innerHTML = '<i class="fa-solid fa-copy"></i>';
    }, 2000);
});

copyBtn.addEventListener('mouseenter',()=>{
    copyBtn.classList.add('success')
})

copyBtn.addEventListener('mouseleave',()=>{
    copyBtn.classList.remove('success')
})

toggleMeshBtn.addEventListener("click", () => {
    toggleMeshBtn.innerHTML = !showMesh ? `<i class="fa-solid fa-eye-slash"></i>` : `<i class="fa-solid fa-eye"></i>`;
    showMesh = !showMesh;
    toggleMeshBtn.classList.toggle("success", showMesh);
});

document.addEventListener("keydown", (e) => {
    if (e.code === "Space" && lastLetter) {
        outputText.value += lastLetter;
        startTime = null;
        lastLetter = '';
        e.preventDefault();
    }
});


loadModel();
