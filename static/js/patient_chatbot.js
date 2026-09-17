(function () {
  const widget = document.getElementById('patientChatbotWidget');
  if (!widget) return;

  const toggle = document.getElementById('patientChatbotToggle');
  const panel = document.getElementById('patientChatbotPanel');
  const closeBtn = document.getElementById('patientChatbotClose');
  const newChatBtn = document.getElementById('patientChatbotNewChat');
  const messages = document.getElementById('patientChatbotMessages');
  const form = document.getElementById('patientChatbotForm');
  const input = document.getElementById('patientChatbotInput');
  const sendBtn = document.getElementById('patientChatbotSend');

  // Patient pages share this widget, so it also supplies a consistent
  // keyboard shortcut past repeated navigation on every public portal page.
  const mainContent = document.querySelector('main');
  if (mainContent) {
    if (!mainContent.id) mainContent.id = 'main-content';
    if (!document.querySelector('.skip-link')) {
      const skipLink = document.createElement('a');
      skipLink.className = 'skip-link';
      skipLink.href = '#main-content';
      skipLink.textContent = 'Skip to main content';
      document.body.insertBefore(skipLink, document.body.firstChild);
    }
  }

  const STORAGE_KEY = 'patientChatbotConversationId';
  let returnFocusElement = null;
  let conversationId = null;
  try {
    conversationId = sessionStorage.getItem(STORAGE_KEY);
  } catch (err) {
    conversationId = null;
  }

  function setConversationId(value) {
    conversationId = value;
    try {
      sessionStorage.setItem(STORAGE_KEY, value || '');
    } catch (err) {
      // sessionStorage unavailable (private mode, etc.) - conversation just won't
      // survive a full page reload, which is an acceptable degradation.
    }
  }

  function newClientId() {
    if (typeof crypto !== 'undefined' && crypto.randomUUID) return crypto.randomUUID();
    return 'c' + Date.now().toString(36) + Math.random().toString(36).slice(2);
  }

  function openPanel() {
    returnFocusElement = document.activeElement;
    panel.classList.add('is-open');
    panel.setAttribute('aria-hidden', 'false');
    toggle.setAttribute('aria-expanded', 'true');
    input.focus();
  }

  function closePanel() {
    panel.classList.remove('is-open');
    panel.setAttribute('aria-hidden', 'true');
    toggle.setAttribute('aria-expanded', 'false');
    if (returnFocusElement && document.contains(returnFocusElement)) {
      returnFocusElement.focus();
    }
    returnFocusElement = null;
  }

  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  function formatInline(str) {
    return str
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/(^|[^*])\*(?!\*)(.+?)\*(?!\*)/g, '$1<em>$2</em>');
  }

  function isTableRow(line) {
    return line.indexOf('|') !== -1;
  }

  function isTableSeparatorRow(line) {
    return /^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?$/.test(line);
  }

  function splitTableRow(line) {
    let trimmed = line.trim();
    if (trimmed.startsWith('|')) trimmed = trimmed.slice(1);
    if (trimmed.endsWith('|')) trimmed = trimmed.slice(0, -1);
    return trimmed.split('|').map((cell) => cell.trim());
  }

  // Ava's answers come back in Markdown (bold, numbered/bulleted lists, and
  // occasionally a pipe table). There's no Markdown library here, so this
  // renders just the handful of patterns Gemini actually uses into real HTML
  // elements. Escaping runs BEFORE any tags are added, so model/user text
  // can never inject markup.
  function renderAssistantText(text) {
    const lines = escapeHtml(text).split('\n');
    const htmlParts = [];
    let listItems = [];
    let listType = null;

    function flushList() {
      if (listItems.length) {
        htmlParts.push('<' + listType + '>' + listItems.join('') + '</' + listType + '>');
        listItems = [];
        listType = null;
      }
    }

    let i = 0;
    while (i < lines.length) {
      const trimmed = lines[i].trim();

      if (isTableRow(trimmed) && i + 1 < lines.length && isTableSeparatorRow(lines[i + 1].trim())) {
        flushList();
        const headerCells = splitTableRow(trimmed);
        const rows = [];
        i += 2;
        while (i < lines.length && isTableRow(lines[i].trim())) {
          rows.push(splitTableRow(lines[i].trim()));
          i += 1;
        }
        const theadHtml = '<thead><tr>' + headerCells.map((cell) => '<th>' + formatInline(cell) + '</th>').join('') + '</tr></thead>';
        const tbodyHtml = '<tbody>' + rows.map((row) =>
          '<tr>' + row.map((cell) => '<td>' + formatInline(cell) + '</td>').join('') + '</tr>'
        ).join('') + '</tbody>';
        htmlParts.push('<div class="chatbot-table-scroll"><table class="chatbot-table">' + theadHtml + tbodyHtml + '</table></div>');
        continue;
      }

      const orderedMatch = trimmed.match(/^\d+\.\s+(.*)$/);
      const bulletMatch = trimmed.match(/^[-*]\s+(.*)$/);
      if (orderedMatch) {
        if (listType !== 'ol') { flushList(); listType = 'ol'; }
        listItems.push('<li>' + formatInline(orderedMatch[1]) + '</li>');
      } else if (bulletMatch) {
        if (listType !== 'ul') { flushList(); listType = 'ul'; }
        listItems.push('<li>' + formatInline(bulletMatch[1]) + '</li>');
      } else {
        flushList();
        if (trimmed) {
          htmlParts.push('<p>' + formatInline(trimmed) + '</p>');
        }
      }
      i += 1;
    }
    flushList();
    return htmlParts.join('') || '<p></p>';
  }

  function addMessage(role, text) {
    const bubble = document.createElement('div');
    bubble.className = 'chatbot-message chatbot-message-' + role;
    if (role === 'assistant') {
      bubble.innerHTML = renderAssistantText(text);
    } else {
      bubble.textContent = text;
    }
    messages.appendChild(bubble);
    messages.scrollTop = messages.scrollHeight;
    return bubble;
  }

  const WORST_CASE_LABEL = 'may take up to a minute';

  function addTypingIndicator() {
    const bubble = document.createElement('div');
    bubble.className = 'chatbot-message chatbot-message-assistant chatbot-typing';
    bubble.innerHTML =
      '<div class="chatbot-typing-row">' +
        '<span class="chatbot-typing-label">Ava is thinking</span>' +
        '<span class="chatbot-typing-dots"><span></span><span></span><span></span></span>' +
      '</div>' +
      '<div class="chatbot-typing-timer"><span class="chatbot-typing-elapsed">0s</span> elapsed &middot; ' + WORST_CASE_LABEL + '</div>';
    messages.appendChild(bubble);
    messages.scrollTop = messages.scrollHeight;

    const elapsedEl = bubble.querySelector('.chatbot-typing-elapsed');
    const startedAt = Date.now();
    const intervalId = setInterval(() => {
      elapsedEl.textContent = Math.round((Date.now() - startedAt) / 1000) + 's';
    }, 1000);
    bubble._chatbotTimerId = intervalId;

    return bubble;
  }

  function resolveTypingIndicator(bubble, text, isError) {
    if (bubble._chatbotTimerId) clearInterval(bubble._chatbotTimerId);
    if (bubble._chatbotProgressId) clearInterval(bubble._chatbotProgressId);
    bubble.classList.remove('chatbot-typing');
    if (isError) {
      bubble.textContent = text;
      bubble.classList.add('chatbot-message-error');
    } else {
      bubble.innerHTML = renderAssistantText(text);
    }
  }

  function startProgressPolling(bubble, pollConversationId) {
    if (!pollConversationId) return;
    const labelEl = bubble.querySelector('.chatbot-typing-label');
    if (!labelEl) return;
    const poll = async () => {
      try {
        const response = await fetch('/api/patient-chatbot/progress?conversation_id=' + encodeURIComponent(pollConversationId));
        if (!response.ok) return;
        const data = await response.json();
        if (data && data.status) labelEl.textContent = data.status;
      } catch (err) {
        // Transient poll failure -- leave the label as-is, next tick will retry.
      }
    };
    poll();
    bubble._chatbotProgressId = setInterval(poll, 1500);
  }

  function addWarning(text) {
    const bubble = document.createElement('div');
    bubble.className = 'chatbot-warning';
    bubble.textContent = text;
    messages.appendChild(bubble);
    messages.scrollTop = messages.scrollHeight;
  }

  const SEND_LABEL = sendBtn.textContent;

  function setBusy(isBusy) {
    input.disabled = isBusy;
    sendBtn.disabled = isBusy;
    sendBtn.textContent = isBusy ? 'Sending...' : SEND_LABEL;
  }

  async function askChatbot(question) {
    const response = await fetch('/api/patient-chatbot/ask', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': typeof CSRF_TOKEN !== 'undefined' ? CSRF_TOKEN : '',
      },
      body: JSON.stringify({ question: question, conversation_id: conversationId }),
    });
    let data = null;
    try {
      data = await response.json();
    } catch (err) {
      data = null;
    }
    if (!response.ok) {
      const message = (data && data.error) || 'Something went wrong. Please try again.';
      throw new Error(message);
    }
    return data;
  }

  toggle.addEventListener('click', () => {
    if (panel.classList.contains('is-open')) {
      closePanel();
    } else {
      openPanel();
    }
  });

  closeBtn.addEventListener('click', closePanel);

  document.addEventListener('keydown', (event) => {
    if (!panel.classList.contains('is-open')) return;
    if (event.key === 'Escape') {
      event.preventDefault();
      closePanel();
      return;
    }
    if (event.key !== 'Tab') return;

    const focusable = Array.from(panel.querySelectorAll(
      'button:not([disabled]), input:not([disabled]), [href], [tabindex]:not([tabindex="-1"])'
    )).filter((element) => !element.hidden);
    if (!focusable.length) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  });

  newChatBtn.addEventListener('click', () => {
    setConversationId(null);
    messages.innerHTML = '';
    addMessage('assistant', "Started a new chat. What would you like to know?");
    input.focus();
  });

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const question = input.value.trim();
    if (!question) return;

    addMessage('user', question);
    input.value = '';
    setBusy(true);
    if (!conversationId) {
      setConversationId(newClientId());
    }
    const pending = addTypingIndicator();
    startProgressPolling(pending, conversationId);

    try {
      const data = await askChatbot(question);
      resolveTypingIndicator(pending, data.answer || 'No answer returned.', false);
      if (data.conversation_id) {
        setConversationId(data.conversation_id);
      }
      if (Array.isArray(data.warnings)) {
        data.warnings.forEach(addWarning);
      }
    } catch (err) {
      resolveTypingIndicator(pending, err.message || 'Something went wrong. Please try again.', true);
    } finally {
      setBusy(false);
      input.focus();
    }
  });
})();
