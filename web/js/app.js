import { config } from "../config.js";
import { checkHealth } from "./api.js";
import { SignCaptureController } from "./camera.js";
import { TextToSignController } from "./hand3d.js";

/**
 * Root application coordinator:
 * - Tab switching & ARIA state synchronization
 * - Startup GET /health check with graceful fallback banner
 * - Central toast notification & global loading spinner helpers
 * - Theme persistence (light / dark), quick phrase presets, and keyboard shortcuts
 */
class ISLStudioApp {
  constructor() {
    this.toastContainer = document.getElementById("toast-container");
    this.spinnerPill = document.getElementById("global-spinner");
    this.spinnerText = document.getElementById("spinner-text");

    this.backendStatusEl = document.getElementById("backend-status");
    this.backendStatusTextEl = document.getElementById("backend-status-text");
    this.systemBanner = document.getElementById("system-banner");
    this.systemBannerMsg = document.getElementById("system-banner-msg");

    this.#initTheme();
    this.#initTabs();

    const uiHooks = {
      onToast: (msg, type = "info") => this.showToast(msg, type),
      onLoading: (busy, label = "Processing…") => this.setLoading(busy, label)
    };

    this.signToText = new SignCaptureController(uiHooks);
    this.textToSign = new TextToSignController(uiHooks);

    this.#initEnhancements();

    document.getElementById("btn-retry-health").addEventListener("click", () => {
      this.verifyBackendHealth();
    });

    this.verifyBackendHealth();
  }

  #initTheme() {
    const saved = localStorage.getItem("isl-studio-theme");
    if (saved === "light" || saved === "dark") {
      document.documentElement.setAttribute("data-theme", saved);
    }

    document.getElementById("theme-toggle").addEventListener("click", () => {
      const current = document.documentElement.getAttribute("data-theme");
      const next = current === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      localStorage.setItem("isl-studio-theme", next);
    });
  }

  #initTabs() {
    const tabButtons = document.querySelectorAll(".tab-btn");
    const panels = {
      "sign-to-text": document.getElementById("panel-sign-to-text"),
      "text-to-sign": document.getElementById("panel-text-to-sign")
    };

    tabButtons.forEach((btn) => {
      btn.addEventListener("click", () => {
        const target = btn.dataset.tab;

        tabButtons.forEach((b) => {
          const active = b.dataset.tab === target;
          b.classList.toggle("is-active", active);
          b.setAttribute("aria-selected", String(active));
        });

        Object.entries(panels).forEach(([key, panel]) => {
          const active = key === target;
          panel.classList.toggle("is-active", active);
          panel.hidden = !active;
        });

        if (target !== "sign-to-text" && this.signToText.isRunning) {
          this.signToText.stop();
        }
        if (target === "text-to-sign") {
          this.textToSign.handleResize();
        }
      });
    });
  }

  #initEnhancements() {
    // Undo last word in sentence box
    const btnUndo = document.getElementById("btn-undo-word");
    const sentenceBox = document.getElementById("sentence-output");
    if (btnUndo && sentenceBox) {
      btnUndo.addEventListener("click", () => {
        const words = sentenceBox.value.trim().split(/\s+/).filter(Boolean);
        words.pop();
        sentenceBox.value = words.join(" ");
      });
    }

    // One-click sample phrase buttons in Tab 2
    const textInput = document.getElementById("text-input");
    const synthForm = document.getElementById("synthesize-form");
    document.querySelectorAll(".chip-preset").forEach((chip) => {
      chip.addEventListener("click", () => {
        if (!textInput || !synthForm) return;
        textInput.value = chip.dataset.phrase || "";
        synthForm.requestSubmit();
      });
    });

    // Spacebar shortcut for Capture Sign when not typing in an input/textarea
    window.addEventListener("keydown", (e) => {
      const tag = document.activeElement?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") return;
      if (e.code === "Space") {
        const captureBtn = document.getElementById("btn-capture-sign");
        if (captureBtn && !captureBtn.disabled) {
          e.preventDefault();
          captureBtn.click();
        }
      }
    });
  }

  async verifyBackendHealth() {
    if (config.USE_MOCK) {
      this.backendStatusEl.className = "status-pill is-mock";
      this.backendStatusTextEl.textContent = "Mock Mode";
      this.systemBanner.classList.remove("hidden");
      this.systemBannerMsg.textContent =
        "Running in offline Mock Mode (config.USE_MOCK = true). Synthetic poses and predictions are active.";
      return;
    }

    this.backendStatusEl.className = "status-pill";
    this.backendStatusTextEl.textContent = "Checking API…";

    try {
      await checkHealth();
      this.backendStatusEl.className = "status-pill is-online";
      this.backendStatusTextEl.textContent = "API Connected";
      this.systemBanner.classList.add("hidden");
    } catch (err) {
      this.backendStatusEl.className = "status-pill is-offline";
      this.backendStatusTextEl.textContent = "API Offline";
      this.systemBanner.classList.remove("hidden");
      this.systemBannerMsg.textContent = `Backend unreachable at ${config.API_BASE} (${err.message}). Start FastAPI or enable USE_MOCK in config.js.`;
    }
  }

  setLoading(busy, label = "Processing…") {
    this.spinnerText.textContent = label;
    this.spinnerPill.classList.toggle("hidden", !busy);
  }

  showToast(message, type = "info") {
    const toast = document.createElement("div");
    toast.className = `toast ${type === "error" ? "toast-error" : ""}`;
    toast.textContent = message;
    this.toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.remove();
    }, 4500);
  }
}

window.addEventListener("DOMContentLoaded", () => {
  new ISLStudioApp();
});