// CardioPredict Platform - Client-side JavaScript
class CardioPredict {
  constructor() {
    this.baseUrl = window.location.origin;
    this.authToken = localStorage.getItem("cardiopredict_token");
    this.isAuthenticated = false;
    this.init();
  }

  init() {
    console.log("🚀 CardioPredict Platform initialized");
    this.checkAuthentication();
    this.setupEventListeners();
    this.loadEndpoints();
  }

  setupEventListeners() {
    // Health check button
    const healthBtn = document.getElementById("health-check-btn");
    if (healthBtn) {
      healthBtn.addEventListener("click", () => this.checkHealth());
    }

    // Prediction form
    const predictionForm = document.getElementById("prediction-form");
    if (predictionForm) {
      predictionForm.addEventListener("submit", (e) => {
        e.preventDefault();
        this.makePrediction();
      });
    }

    // Authentication forms
    const loginForm = document.getElementById("login-form");
    if (loginForm) {
      loginForm.addEventListener("submit", (e) => {
        e.preventDefault();
        this.login();
      });
    }

    const registerForm = document.getElementById("register-form");
    if (registerForm) {
      registerForm.addEventListener("submit", (e) => {
        e.preventDefault();
        this.register();
      });
    }
  }

  async checkHealth() {
    this.showLoading("health-result");
    try {
      const response = await fetch(`${this.baseUrl}/health`);
      const data = await response.json();
      this.showResult("health-result", data, response.ok);
    } catch (error) {
      this.showResult("health-result", { error: error.message }, false);
    }
  }

  async makePrediction() {
    const formData = new FormData(document.getElementById("prediction-form"));
    const data = Object.fromEntries(formData.entries());

    this.showLoading("prediction-result");
    try {
      const response = await fetch(`${this.baseUrl}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(this.authToken && { Authorization: `Bearer ${this.authToken}` }),
        },
        body: JSON.stringify(data),
      });
      const result = await response.json();
      this.showResult("prediction-result", result, response.ok);
    } catch (error) {
      this.showResult("prediction-result", { error: error.message }, false);
    }
  }

  async login() {
    const formData = new FormData(document.getElementById("login-form"));
    const data = Object.fromEntries(formData.entries());

    this.showLoading("login-result");
    try {
      const response = await fetch(`${this.baseUrl}/api/v3/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });
      const result = await response.json();

      if (response.ok && result.access_token) {
        this.authToken = result.access_token;
        localStorage.setItem("cardiopredict_token", this.authToken);
        this.isAuthenticated = true;
        this.showResult("login-result", { message: "Login successful!" }, true);
        this.updateAuthUI();
      } else {
        this.showResult("login-result", result, false);
      }
    } catch (error) {
      this.showResult("login-result", { error: error.message }, false);
    }
  }

  async register() {
    const formData = new FormData(document.getElementById("register-form"));
    const data = Object.fromEntries(formData.entries());

    this.showLoading("register-result");
    try {
      const response = await fetch(`${this.baseUrl}/api/v3/auth/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });
      const result = await response.json();
      this.showResult("register-result", result, response.ok);
    } catch (error) {
      this.showResult("register-result", { error: error.message }, false);
    }
  }

  async checkAuthentication() {
    if (!this.authToken) return;

    try {
      const response = await fetch(`${this.baseUrl}/api/v3/auth/profile`, {
        headers: { Authorization: `Bearer ${this.authToken}` },
      });

      if (response.ok) {
        this.isAuthenticated = true;
        this.updateAuthUI();
      } else {
        this.logout();
      }
    } catch (error) {
      console.warn("Auth check failed:", error);
      this.logout();
    }
  }

  logout() {
    this.authToken = null;
    this.isAuthenticated = false;
    localStorage.removeItem("cardiopredict_token");
    this.updateAuthUI();
  }

  updateAuthUI() {
    const authSection = document.getElementById("auth-section");
    const protectedSection = document.getElementById("protected-section");

    if (this.isAuthenticated) {
      if (authSection) authSection.style.display = "none";
      if (protectedSection) protectedSection.style.display = "block";
    } else {
      if (authSection) authSection.style.display = "block";
      if (protectedSection) protectedSection.style.display = "none";
    }
  }

  async loadEndpoints() {
    try {
      const response = await fetch(`${this.baseUrl}/api/endpoints`);
      const data = await response.json();
      this.displayEndpoints(data.endpoints || []);
    } catch (error) {
      console.warn("Could not load endpoints:", error);
    }
  }

  displayEndpoints(endpoints) {
    const container = document.getElementById("endpoints-list");
    if (!container) return;

    container.innerHTML = endpoints
      .map(
        (endpoint) => `
            <div class="endpoint-card">
                <h4>
                    <span class="method ${endpoint.methods[0]}">${
          endpoint.methods[0]
        }</span>
                    ${endpoint.rule}
                </h4>
                <p>${endpoint.endpoint}</p>
                ${
                  endpoint.phase
                    ? `<span class="phase-indicator">Phase ${endpoint.phase}</span>`
                    : ""
                }
            </div>
        `
      )
      .join("");
  }

  showLoading(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
      element.innerHTML = '<div class="loading">⏳ Loading...</div>';
    }
  }

  showResult(elementId, data, isSuccess) {
    const element = document.getElementById(elementId);
    if (element) {
      const className = isSuccess ? "result success" : "result error";
      element.innerHTML = `
                <div class="${className}">
                    <pre>${JSON.stringify(data, null, 2)}</pre>
                </div>
            `;
    }
  }

  // Utility methods
  async testEndpoint(url, method = "GET", data = null) {
    const options = {
      method,
      headers: {
        "Content-Type": "application/json",
        ...(this.authToken && { Authorization: `Bearer ${this.authToken}` }),
      },
    };

    if (data && method !== "GET") {
      options.body = JSON.stringify(data);
    }

    try {
      const response = await fetch(`${this.baseUrl}${url}`, options);
      return {
        ok: response.ok,
        status: response.status,
        data: await response.json(),
      };
    } catch (error) {
      return {
        ok: false,
        status: 0,
        data: { error: error.message },
      };
    }
  }
}

// Initialize the app when DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
  window.cardioPredict = new CardioPredict();
});

// WebSocket connection for real-time features
function initWebSocket() {
  if (typeof io !== "undefined") {
    const socket = io();

    socket.on("connect", () => {
      console.log("🔌 WebSocket connected");
      updateConnectionStatus(true);
    });

    socket.on("disconnect", () => {
      console.log("🔌 WebSocket disconnected");
      updateConnectionStatus(false);
    });

    socket.on("prediction_update", (data) => {
      console.log("📊 Prediction update:", data);
      displayRealtimeUpdate(data);
    });

    window.socket = socket;
  }
}

function updateConnectionStatus(isConnected) {
  const indicator = document.getElementById("connection-status");
  if (indicator) {
    indicator.className = `status-indicator ${
      isConnected ? "online" : "offline"
    }`;
    indicator.title = isConnected ? "Connected" : "Disconnected";
  }
}

function displayRealtimeUpdate(data) {
  const container = document.getElementById("realtime-updates");
  if (container) {
    const update = document.createElement("div");
    update.className = "result";
    update.innerHTML = `
            <strong>${new Date().toLocaleTimeString()}</strong>: 
            ${JSON.stringify(data, null, 2)}
        `;
    container.prepend(update);

    // Keep only last 10 updates
    while (container.children.length > 10) {
      container.removeChild(container.lastChild);
    }
  }
}

// Initialize WebSocket when the page loads
setTimeout(initWebSocket, 1000);
