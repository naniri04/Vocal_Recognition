// content.js
function getCurrentTime() {
    const video = document.querySelector('video');
    if (video) {
      const minutes = Math.floor(video.currentTime / 60);
      const seconds = Math.floor(video.currentTime % 60);
      const millisec = Math.floor(video.currentTime * 10 % 100);
      return `${minutes}m ${seconds}.${millisec}s`;
    }
    return "No video playing";
}

function rewindVideo(seconds) {
    const video = document.querySelector('video');
    if (video) {
        video.currentTime = Math.max(0, video.currentTime - seconds);
    }
}

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "getTime") {
        const currentTime = getCurrentTime();
        sendResponse({time: currentTime});
    } else if (request.action === "rewind") {
        rewindVideo(request.seconds);
        sendResponse({status: "rewinded"});
    }
});