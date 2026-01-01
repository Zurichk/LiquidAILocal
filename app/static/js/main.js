/**
 * AEP LiquidAI Chat - Main JavaScript
 * Handles chat functionality, API interactions, and UI state management.
 */

// =============================================================================
// Constants & Configuration
// =============================================================================

const AEP_API_BASE_URL = '/api/v1';
const AEP_DEFAULT_SETTINGS = {
    systemPrompt: 'You are a helpful assistant trained by Liquid AI.',
    maxTokens: 512,
    temperature: 0.3,
    streaming: true,
    apiKey: ''
};

// =============================================================================
// State Management
// =============================================================================

const state = {
    messages: [],
    isModelLoaded: false,
    isGenerating: false,
    settings: { ...AEP_DEFAULT_SETTINGS },
    chatHistory: []
};

// =============================================================================
// DOM Elements
// =============================================================================

const elements = {
    chatMessages: document.getElementById('chatMessages'),
    chatInput: document.getElementById('chatInput'),
    chatForm: document.getElementById('chatForm'),
    sendBtn: document.getElementById('sendBtn'),
    newChatBtn: document.getElementById('newChatBtn'),
    loadModelBtn: document.getElementById('loadModelBtn'),
    modelStatus: document.getElementById('modelStatus'),
    settingsBtn: document.getElementById('settingsBtn'),
    settingsModal: document.getElementById('settingsModal'),
    closeSettings: document.getElementById('closeSettings'),
    saveSettings: document.getElementById('saveSettings'),
    resetSettings: document.getElementById('resetSettings'),
    loadingOverlay: document.getElementById('loadingOverlay'),
    loadingText: document.getElementById('loadingText'),
    menuToggle: document.getElementById('menuToggle'),
    sidebar: document.querySelector('.sidebar'),
    // Settings inputs
    systemPrompt: document.getElementById('systemPrompt'),
    maxTokens: document.getElementById('maxTokens'),
    temperature: document.getElementById('temperature'),
    tempValue: document.getElementById('tempValue'),
    streamingToggle: document.getElementById('streamingToggle'),
    apiKey: document.getElementById('apiKey')
};

// =============================================================================
// API Functions
// =============================================================================

/**
 * Makes an API request with optional authentication.
 * @param {string} endpoint - API endpoint.
 * @param {Object} options - Fetch options.
 * @returns {Promise<Response>} - Fetch response.
 */
async function apiRequest(endpoint, options = {}) {
    const headers = {
        'Content-Type': 'application/json',
        ...options.headers
    };

    if (state.settings.apiKey) {
        headers['X-API-Key'] = state.settings.apiKey;
    }

    return fetch(`${AEP_API_BASE_URL}${endpoint}`, {
        ...options,
        headers
    });
}

/**
 * Checks if the model is loaded.
 * @returns {Promise<boolean>} - Model loaded status.
 */
async function checkModelStatus() {
    try {
        const response = await apiRequest('/health');
        const data = await response.json();
        return data.model_loaded;
    } catch (error) {
        console.error('Error checking model status:', error);
        return false;
    }
}

/**
 * Loads the model.
 * @returns {Promise<boolean>} - Success status.
 */
async function loadModel() {
    showLoading('Cargando modelo... Esto puede tardar varios minutos.');
    
    try {
        const response = await apiRequest('/model/load', { method: 'POST' });
        const data = await response.json();
        
        if (data.success) {
            updateModelStatus(true);
            hideLoading();
            showNotification('Modelo cargado exitosamente', 'success');
            return true;
        } else {
            throw new Error(data.error || 'Error al cargar el modelo');
        }
    } catch (error) {
        hideLoading();
        showNotification(`Error: ${error.message}`, 'error');
        return false;
    }
}

/**
 * Sends a chat message and gets a response.
 * @param {string} message - User message.
 */
async function sendMessage(message) {
    if (!message.trim() || state.isGenerating) return;

    // Add user message to state and UI
    const userMessage = { role: 'user', content: message };
    state.messages.push(userMessage);
    appendMessage(userMessage);

    // Clear input and disable while generating
    elements.chatInput.value = '';
    autoResizeTextarea(elements.chatInput);
    setGeneratingState(true);

    // Create assistant message placeholder
    const assistantDiv = createMessageElement({ role: 'assistant', content: '' });
    elements.chatMessages.appendChild(assistantDiv);
    scrollToBottom();

    const contentDiv = assistantDiv.querySelector('.message-content');
    
    // Show typing indicator
    contentDiv.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';

    try {
        const requestBody = {
            messages: state.messages,
            max_tokens: state.settings.maxTokens,
            temperature: state.settings.temperature,
            system: state.settings.systemPrompt,
            stream: state.settings.streaming
        };

        if (state.settings.streaming) {
            await handleStreamingResponse(requestBody, contentDiv);
        } else {
            await handleNonStreamingResponse(requestBody, contentDiv);
        }
    } catch (error) {
        contentDiv.textContent = `Error: ${error.message}`;
        console.error('Error sending message:', error);
    }

    setGeneratingState(false);
    scrollToBottom();
}

/**
 * Handles streaming response from the API.
 * @param {Object} requestBody - Request body.
 * @param {HTMLElement} contentDiv - Content element to update.
 */
async function handleStreamingResponse(requestBody, contentDiv) {
    const response = await apiRequest('/chat/completions', {
        method: 'POST',
        body: JSON.stringify(requestBody)
    });

    if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Error en la API');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let assistantContent = '';

    contentDiv.innerHTML = '';

    while (true) {
        const { done, value } = await reader.read();
        
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const data = line.slice(6);
                
                if (data === '[DONE]') continue;

                try {
                    const parsed = JSON.parse(data);
                    const content = parsed.choices?.[0]?.delta?.content;
                    
                    if (content) {
                        assistantContent += content;
                        contentDiv.innerHTML = formatMessage(assistantContent);
                        scrollToBottom();
                    }
                } catch (e) {
                    // Ignore parsing errors for incomplete chunks
                }
            }
        }
    }

    // Save assistant message to state
    state.messages.push({ role: 'assistant', content: assistantContent });
}

/**
 * Handles non-streaming response from the API.
 * @param {Object} requestBody - Request body.
 * @param {HTMLElement} contentDiv - Content element to update.
 */
async function handleNonStreamingResponse(requestBody, contentDiv) {
    requestBody.stream = false;
    
    const response = await apiRequest('/chat/completions', {
        method: 'POST',
        body: JSON.stringify(requestBody)
    });

    const data = await response.json();

    if (!response.ok) {
        throw new Error(data.message || 'Error en la API');
    }

    const assistantContent = data.choices[0].message.content;
    contentDiv.innerHTML = formatMessage(assistantContent);
    
    // Save assistant message to state
    state.messages.push({ role: 'assistant', content: assistantContent });
}

// =============================================================================
// UI Functions
// =============================================================================

/**
 * Creates a message element.
 * @param {Object} message - Message object with role and content.
 * @returns {HTMLElement} - Message element.
 */
function createMessageElement(message) {
    const div = document.createElement('div');
    div.className = `message ${message.role}`;

    const avatarSvg = message.role === 'user' 
        ? '<svg viewBox="0 0 24 24"><path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z" fill="currentColor"/></svg>'
        : '<svg viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z" fill="currentColor"/></svg>';

    div.innerHTML = `
        <div class="message-avatar">${avatarSvg}</div>
        <div class="message-content">${formatMessage(message.content)}</div>
    `;

    return div;
}

/**
 * Appends a message to the chat.
 * @param {Object} message - Message object.
 */
function appendMessage(message) {
    // Remove welcome message if present
    const welcomeMessage = elements.chatMessages.querySelector('.welcome-message');
    if (welcomeMessage) {
        welcomeMessage.remove();
    }

    const messageElement = createMessageElement(message);
    elements.chatMessages.appendChild(messageElement);
    scrollToBottom();
}

/**
 * Formats message content with basic markdown.
 * @param {string} content - Raw content.
 * @returns {string} - Formatted HTML.
 */
function formatMessage(content) {
    if (!content) return '';

    // Escape HTML
    let formatted = content
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');

    // Code blocks
    formatted = formatted.replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
        return `<pre><code class="language-${lang || ''}">${code.trim()}</code></pre>`;
    });

    // Inline code
    formatted = formatted.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Bold
    formatted = formatted.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

    // Italic
    formatted = formatted.replace(/\*([^*]+)\*/g, '<em>$1</em>');

    // Line breaks
    formatted = formatted.replace(/\n/g, '<br>');

    return formatted;
}

/**
 * Scrolls chat to bottom.
 */
function scrollToBottom() {
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
}

/**
 * Updates model status indicator.
 * @param {boolean} isLoaded - Model loaded status.
 */
function updateModelStatus(isLoaded) {
    state.isModelLoaded = isLoaded;
    
    const indicator = elements.modelStatus.querySelector('.status-indicator');
    const text = elements.modelStatus.querySelector('.status-text');

    if (isLoaded) {
        indicator.className = 'status-indicator online';
        text.textContent = 'Modelo cargado';
        elements.loadModelBtn.textContent = 'Modelo listo';
        elements.loadModelBtn.disabled = true;
        elements.chatInput.disabled = false;
        elements.sendBtn.disabled = false;
        elements.chatInput.placeholder = 'Escribe tu mensaje aquí...';
    } else {
        indicator.className = 'status-indicator offline';
        text.textContent = 'Modelo no cargado';
        elements.loadModelBtn.textContent = 'Cargar modelo';
        elements.loadModelBtn.disabled = false;
        elements.chatInput.disabled = true;
        elements.sendBtn.disabled = true;
        elements.chatInput.placeholder = 'Carga el modelo para comenzar...';
    }
}

/**
 * Sets generating state.
 * @param {boolean} isGenerating - Is generating.
 */
function setGeneratingState(isGenerating) {
    state.isGenerating = isGenerating;
    elements.sendBtn.disabled = isGenerating || !state.isModelLoaded;
    elements.chatInput.disabled = isGenerating;
}

/**
 * Shows loading overlay.
 * @param {string} text - Loading text.
 */
function showLoading(text) {
    elements.loadingText.textContent = text;
    elements.loadingOverlay.classList.add('active');
}

/**
 * Hides loading overlay.
 */
function hideLoading() {
    elements.loadingOverlay.classList.remove('active');
}

/**
 * Shows a notification.
 * @param {string} message - Notification message.
 * @param {string} type - Notification type (success, error, info).
 */
function showNotification(message, type = 'info') {
    // Simple alert for now, can be replaced with a toast notification
    console.log(`[${type.toUpperCase()}] ${message}`);
}

/**
 * Auto-resizes textarea based on content.
 * @param {HTMLTextAreaElement} textarea - Textarea element.
 */
function autoResizeTextarea(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 150) + 'px';
}

/**
 * Starts a new chat.
 */
function newChat() {
    state.messages = [];
    elements.chatMessages.innerHTML = `
        <div class="welcome-message">
            <div class="welcome-icon">
                <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z" fill="currentColor"/>
                </svg>
            </div>
            <h2>¡Nueva conversación!</h2>
            <p>Escribe tu mensaje para comenzar a chatear con LiquidAI LFM2-2.6B.</p>
        </div>
    `;
}

// =============================================================================
// Settings Functions
// =============================================================================

/**
 * Loads settings from localStorage.
 */
function loadSettings() {
    const saved = localStorage.getItem('aep_chat_settings');
    if (saved) {
        try {
            state.settings = { ...AEP_DEFAULT_SETTINGS, ...JSON.parse(saved) };
        } catch (e) {
            console.error('Error loading settings:', e);
        }
    }
    updateSettingsUI();
}

/**
 * Saves settings to localStorage.
 */
function saveSettings() {
    state.settings.systemPrompt = elements.systemPrompt.value || AEP_DEFAULT_SETTINGS.systemPrompt;
    state.settings.maxTokens = parseInt(elements.maxTokens.value) || AEP_DEFAULT_SETTINGS.maxTokens;
    state.settings.temperature = parseFloat(elements.temperature.value) || AEP_DEFAULT_SETTINGS.temperature;
    state.settings.streaming = elements.streamingToggle.checked;
    state.settings.apiKey = elements.apiKey.value;

    localStorage.setItem('aep_chat_settings', JSON.stringify(state.settings));
    closeSettingsModal();
    showNotification('Configuración guardada', 'success');
}

/**
 * Resets settings to defaults.
 */
function resetSettings() {
    state.settings = { ...AEP_DEFAULT_SETTINGS };
    updateSettingsUI();
    localStorage.removeItem('aep_chat_settings');
    showNotification('Configuración restaurada', 'info');
}

/**
 * Updates settings UI with current values.
 */
function updateSettingsUI() {
    elements.systemPrompt.value = state.settings.systemPrompt;
    elements.maxTokens.value = state.settings.maxTokens;
    elements.temperature.value = state.settings.temperature;
    elements.tempValue.textContent = state.settings.temperature;
    elements.streamingToggle.checked = state.settings.streaming;
    elements.apiKey.value = state.settings.apiKey;
}

/**
 * Opens settings modal.
 */
function openSettingsModal() {
    updateSettingsUI();
    elements.settingsModal.classList.add('active');
}

/**
 * Closes settings modal.
 */
function closeSettingsModal() {
    elements.settingsModal.classList.remove('active');
}

// =============================================================================
// Event Listeners
// =============================================================================

function initEventListeners() {
    // Chat form submission
    elements.chatForm.addEventListener('submit', (e) => {
        e.preventDefault();
        sendMessage(elements.chatInput.value);
    });

    // Input auto-resize and keyboard shortcuts
    elements.chatInput.addEventListener('input', () => {
        autoResizeTextarea(elements.chatInput);
    });

    elements.chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            elements.chatForm.dispatchEvent(new Event('submit'));
        }
    });

    // New chat button
    elements.newChatBtn.addEventListener('click', newChat);

    // Load model button
    elements.loadModelBtn.addEventListener('click', loadModel);

    // Settings modal
    elements.settingsBtn.addEventListener('click', openSettingsModal);
    elements.closeSettings.addEventListener('click', closeSettingsModal);
    elements.saveSettings.addEventListener('click', saveSettings);
    elements.resetSettings.addEventListener('click', resetSettings);

    // Close modal on overlay click
    elements.settingsModal.querySelector('.modal-overlay').addEventListener('click', closeSettingsModal);

    // Temperature slider
    elements.temperature.addEventListener('input', (e) => {
        elements.tempValue.textContent = e.target.value;
    });

    // Mobile menu toggle
    elements.menuToggle.addEventListener('click', () => {
        elements.sidebar.classList.toggle('open');
    });

    // Close sidebar when clicking outside on mobile
    document.addEventListener('click', (e) => {
        if (window.innerWidth <= 768 && 
            !elements.sidebar.contains(e.target) && 
            !elements.menuToggle.contains(e.target)) {
            elements.sidebar.classList.remove('open');
        }
    });

    // Escape key to close modal
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            closeSettingsModal();
        }
    });
}

// =============================================================================
// Initialization
// =============================================================================

async function init() {
    loadSettings();
    initEventListeners();
    
    // Check initial model status
    const isLoaded = await checkModelStatus();
    updateModelStatus(isLoaded);

    // Periodic status check (every 30 seconds)
    setInterval(async () => {
        if (!state.isGenerating) {
            const isLoaded = await checkModelStatus();
            updateModelStatus(isLoaded);
        }
    }, 30000);
}

// Start the application
document.addEventListener('DOMContentLoaded', init);
