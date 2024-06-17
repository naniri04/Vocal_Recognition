// popup.js
document.addEventListener('DOMContentLoaded', () => {
  function updateTime() {
    chrome.tabs.query({ active: true, lastFocusedWindow: true }, (tabs) => {
      // Ensure the tab is a YouTube tab
      // if (tabs.length != 0) {console.log(tabs[0].url)}
      if (tabs.length != 0 && tabs[0].url.includes("youtube.com/watch")) {
        chrome.tabs.sendMessage(tabs[0].id, { action: "getTime" }, (response) => {
          const timeDiv = document.getElementById('time');
          if (response && response.time) {
            timeDiv.textContent = response.time;
          } else {
            timeDiv.textContent = "Press f5 to refresh page";
          }
        });
      } else {
        document.getElementById('time').textContent = "Not a YouTube video tab";
      }
    });
  }

  // Initial call to set the time
  updateTime();

  // Set interval to update the time every second
  setInterval(updateTime, 100);

  // Listen for keyboard event to rewind video
  document.addEventListener('keydown', (event) => {
    const key = event.key
    if (key === ',' || key === '.') {
      chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        if (tabs[0].url.includes("youtube.com/watch")) {
          chrome.tabs.sendMessage(tabs[0].id, { action: "rewind", seconds: (key === ',' ? 5 : -5) }, (response) => {
            console.log('Video rewinded by 5 seconds');
          });
        }
      });
    }
  });
});
