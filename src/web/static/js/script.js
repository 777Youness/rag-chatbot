document.addEventListener('DOMContentLoaded', () => {
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-btn');
    const statusMessage = document.getElementById('status-message');

    // Function to add a message to the chat
    function addMessage(message, isUser = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';

        // Convert markdown code blocks to HTML
        let formattedMessage = message.replace(/```([\s\S]*?)```/g, (match, code) => {
            return `<pre><code>${code.trim()}</code></pre>`;
        });

        // Convert inline code to HTML
        formattedMessage = formattedMessage.replace(/`([^`]+)`/g, '<code>$1</code>');

        // Add paragraphs
        const paragraphs = formattedMessage.split('\n\n');
        paragraphs.forEach(paragraph => {
            if (paragraph.trim()) {
                const p = document.createElement('p');
                p.innerHTML = paragraph.replace(/\n/g, '<br>');
                contentDiv.appendChild(p);
            }
        });

        messageDiv.appendChild(contentDiv);
        chatMessages.appendChild(messageDiv);

        // Scroll to the bottom of the chat
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Function to show loading indicator
    function showLoading() {
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'message bot';
        loadingDiv.id = 'loading-message';

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';

        const loading = document.createElement('div');
        loading.className = 'loading';
        loading.innerHTML = '<div></div><div></div><div></div>';

        contentDiv.appendChild(loading);
        loadingDiv.appendChild(contentDiv);
        chatMessages.appendChild(loadingDiv);

        // Scroll to the bottom of the chat
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Function to hide loading indicator
    function hideLoading() {
        const loadingMessage = document.getElementById('loading-message');
        if (loadingMessage) {
            loadingMessage.remove();
        }
    }

    // Function to handle sending a message
    async function sendMessage() {
        const message = userInput.value.trim();

        if (!message) return;

        // Add user message to chat
        addMessage(message, true);

        // Clear input field
        userInput.value = '';

        // Show loading indicator
        showLoading();

        try {
            // Send request to server
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: message }),
            });

            // Hide loading indicator
            hideLoading();

            if (response.ok) {
                const data = await response.json();
                addMessage(data.response);
                statusMessage.textContent = 'Message sent successfully';
            } else {
                const errorData = await response.json();
                addMessage(`Error: ${errorData.error || 'Unknown error occurred'}`);
                statusMessage.textContent = 'Error sending message';
            }
        } catch (error) {
            // Hide loading indicator
            hideLoading();

            console.error('Error:', error);
            addMessage(`Error: Could not connect to the server`);
            statusMessage.textContent = 'Connection error';
        }
    }

    // Event listeners
    sendButton.addEventListener('click', sendMessage);

    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    // Focus input on page load
    userInput.focus();
});