"use strict";

const form = document.querySelector("#login-form");
const tokenInput = document.querySelector("[data-testid=login-token]");
const statusText = document.querySelector("[data-testid=session-status]");
const logoutButton = document.querySelector("[data-testid=logout-button]");
const prepareLink = document.querySelector("[data-testid=prepare-link]");
let csrfToken = null;

function showSession(signedIn) {
  statusText.textContent = signedIn ? "Signed in." : "Not signed in.";
  form.hidden = signedIn;
  logoutButton.hidden = !signedIn;
  prepareLink.hidden = !signedIn;
  if (!signedIn) csrfToken = null;
}

async function readSession() {
  const response = await fetch("/session", {cache: "no-store"});
  if (!response.ok) {
    showSession(false);
    return;
  }
  const session = await response.json();
  csrfToken = session.csrfToken;
  showSession(true);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const token = tokenInput.value;
  tokenInput.value = "";
  try {
    const response = await fetch("/session", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({token}),
    });
    if (!response.ok) {
      statusText.textContent = "Sign-in failed.";
      return;
    }
    await readSession();
  } catch {
    statusText.textContent = "Sign-in failed.";
  }
});

logoutButton.addEventListener("click", async () => {
  try {
    const response = await fetch("/session", {
      method: "DELETE",
      headers: {"X-CSRF-Token": csrfToken},
    });
    if (response.ok) showSession(false);
    else statusText.textContent = "Sign-out failed.";
  } catch {
    statusText.textContent = "Sign-out failed.";
  }
});

readSession().catch(() => showSession(false));
