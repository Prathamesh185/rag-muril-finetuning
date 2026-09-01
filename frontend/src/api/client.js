export async function sendChat(question, modelChoice) {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      question,
      model_choice: modelChoice,
    }),
  });

  if (!response.ok) {
    let message = `Request failed with status ${response.status}`;

    try {
      const error = await response.json();

      if (error?.detail) {
        message =
          typeof error.detail === "string"
            ? error.detail
            : JSON.stringify(error.detail);
      }
    } catch {
      // Keep the HTTP status message.
    }

    throw new Error(message);
  }

  return response.json();
}


export async function retrievePassages(question, topK = 5) {
  const response = await fetch("/api/retrieve", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      question,
      top_k: topK,
    }),
  });

  if (!response.ok) {
    let message = `Request failed with status ${response.status}`;

    try {
      const error = await response.json();

      if (error?.detail) {
        message =
          typeof error.detail === "string"
            ? error.detail
            : JSON.stringify(error.detail);
      }
    } catch {
      // Keep the HTTP status message.
    }

    throw new Error(message);
  }

  return response.json();
}

export async function compareModels(
  question,
  topK = 5
) {
  const response = await fetch("/api/compare", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      question,
      top_k: topK,
    }),
  });

  if (!response.ok) {
    let message =
      `Request failed with status ${response.status}`;

    try {
      const error = await response.json();

      if (error?.detail) {
        message =
          typeof error.detail === "string"
            ? error.detail
            : JSON.stringify(error.detail);
      }
    } catch {
      // Keep HTTP status message.
    }

    throw new Error(message);
  }

  return response.json();
}