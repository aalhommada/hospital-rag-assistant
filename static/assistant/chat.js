/*
 * Two small jobs, no framework.
 *
 * 1. Consume the Server-Sent Events stream for an answer and paint tokens as
 *    they arrive.
 * 2. Fill the composer when a suggestion chip is clicked.
 *
 * HTMX does everything else: it posts the form and swaps Django's HTML into
 * the thread. This file only handles the part a form post genuinely cannot do.
 */
(function () {
  "use strict";

  function scrollToBottom() {
    window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
  }

  function startStream(article) {
    var url = article.dataset.streamUrl;
    if (!url || article.dataset.streamStarted) return;
    article.dataset.streamStarted = "1";

    var statusEl = article.querySelector("[data-status]");
    var answerEl = article.querySelector("[data-answer]");
    var source = new EventSource(url);

    source.onmessage = function (event) {
      var payload = JSON.parse(event.data);

      switch (payload.type) {
        case "status":
          if (statusEl) statusEl.textContent = payload.text;
          break;

        case "token":
          if (statusEl) statusEl.remove(), (statusEl = null);
          answerEl.textContent += payload.text;
          scrollToBottom();
          break;

        case "replace":
          // The finished answer, rendered by Django with its citations and
          // sources. Swapping the whole article keeps one source of truth for
          // markup instead of rebuilding it in JavaScript.
          article.outerHTML = payload.html;
          scrollToBottom();
          break;

        case "error":
          if (statusEl) statusEl.remove(), (statusEl = null);
          article.classList.remove("is-streaming");
          answerEl.innerHTML = '<p class="stream-error"></p>';
          answerEl.querySelector(".stream-error").textContent = payload.text;
          break;

        case "end":
          source.close();
          break;
      }
    };

    source.onerror = function () {
      source.close();
      if (statusEl) {
        statusEl.classList.add("stream-error");
        statusEl.textContent = "The connection dropped. Please ask again.";
      }
    };
  }

  function scanForStreams(root) {
    (root || document).querySelectorAll(".message-assistant[data-stream-url]").forEach(startStream);
  }

  document.addEventListener("DOMContentLoaded", function () {
    scanForStreams();

    document.addEventListener("click", function (event) {
      var chip = event.target.closest(".suggestion");
      if (!chip) return;
      var input = document.getElementById("question");
      input.value = chip.textContent.trim();
      input.focus();
    });
  });

  // HTMX has just inserted the new exchange; look for an answer bubble to fill.
  document.body.addEventListener("htmx:afterSwap", function (event) {
    scanForStreams(event.target);
    scrollToBottom();
  });
})();
