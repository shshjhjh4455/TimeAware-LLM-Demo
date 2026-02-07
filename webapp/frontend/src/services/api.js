const API_BASE = '/api';

export async function fetchHealth() {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.json();
}

export async function fetchCharacters() {
  const res = await fetch(`${API_BASE}/characters`);
  if (!res.ok) throw new Error(`Failed to fetch characters: ${res.status}`);
  return res.json();
}

export async function sendChat({ character, character_period, question, include_debug = true }) {
  const res = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ character, character_period, question, include_debug }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
    throw new Error(err.error || `Chat failed: ${res.status}`);
  }
  return res.json();
}

/**
 * Streaming chat endpoint using Server-Sent Events.
 * @param {Object} params - character, character_period, question
 * @param {Function} onDebug - called with debug info when pipeline prep is done
 * @param {Function} onToken - called with each response token chunk
 * @param {Function} onDone - called with final { answer, debug } when complete
 * @param {Function} onError - called on error
 */
export async function sendChatStream({ character, character_period, question }, { onDebug, onToken, onDone, onError }) {
  try {
    const res = await fetch(`${API_BASE}/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ character, character_period, question }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
      throw new Error(err.error || `Stream failed: ${res.status}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        try {
          const event = JSON.parse(line.slice(6));
          if (event.type === 'debug' && onDebug) onDebug(event);
          else if (event.type === 'token' && onToken) onToken(event.content);
          else if (event.type === 'done' && onDone) onDone(event);
          else if (event.type === 'error' && onError) onError(new Error(event.error));
        } catch {
          // skip malformed JSON
        }
      }
    }
  } catch (err) {
    if (onError) onError(err);
  }
}
