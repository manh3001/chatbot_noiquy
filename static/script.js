const chatBox = document.getElementById("chat-box");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");

function appendMessage(sender, text) {
  const msg = document.createElement("div");
  msg.classList.add("message", sender);
  msg.textContent = text;
  chatBox.appendChild(msg);
  chatBox.scrollTop = chatBox.scrollHeight;
}

sendBtn.addEventListener("click", async () => {
  const question = userInput.value.trim();
  if (!question) return;
  appendMessage("user", question);
  userInput.value = "";

  appendMessage("bot", "Đang trả lời...");

  try {
    const response = await fetch("http://127.0.0.1:8000/ask", {
      method: "POST",
      mode: "cors",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    const data = await response.json();
    const botMsg = document.querySelector(".bot:last-child");
    botMsg.textContent = data.answer || "Xin lỗi, tôi chưa có câu trả lời phù hợp.";
  } catch (err) {
    console.error(err);
    appendMessage("bot", "Lỗi kết nối đến server ");
  }
});
