import { useEffect, useRef, useState } from "react";
import { sendChat } from "./api/client";
import { AnalysisPage, ComparisonPage } from "./App";
import "./research.css";

const NAV = [
  ["assistant", "Ask AgriSahayak", "01"],
  ["analysis", "Retrieval explorer", "02"],
  ["compare", "Model comparison", "03"],
  ["architecture", "How it works", "04"],
];

const EXAMPLES = {
  hi: [
    ["Crop cultivation", "धान की खेती कैसे करें?"],
    ["Soil & nutrition", "जीवामृत कैसे तैयार करें?"],
    ["Agricultural practices", "प्राकृतिक खेती के क्या लाभ हैं?"],
  ],
};

function Leaf() {
  return (
    <svg viewBox="0 0 32 32" fill="none" aria-hidden="true">
      <path
        d="M25 6C12 5 5 10 7 19c2 8 17 10 18-13Z"
        stroke="currentColor"
        strokeWidth="1.7"
      />
      <path
        d="m7 27 13-15M13 21l-1-7m5 3 6 1"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinecap="round"
      />
    </svg>
  );
}

function safeUrl(value) {
  try {
    const u = new URL(value);

    return ["https:", "http:"].includes(u.protocol) ? u.href : null;
  } catch {
    return null;
  }
}

function Answer({ text }) {
  return text
    .split(/(\*\*[^*]+\*\*)/g)
    .map((part, i) =>
      part.startsWith("**") ? (
        <strong key={i}>{part.slice(2, -2)}</strong>
      ) : (
        <span key={i}>{part}</span>
      ),
    );
}

function Evidence({ items, done }) {
  return (
    <aside className="evidence-panel" aria-label="Retrieved evidence">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">THE KNOWLEDGE BEHIND THE ANSWER</span>
          <h2>Retrieved evidence</h2>
        </div>

        <span className="count">{items.length}</span>
      </div>

      {!items.length ? (
        <div className="evidence-empty">
          <div className="paper-stack">
            <span />
            <span />
            <span />
          </div>

          <h3>
            {done
              ? "No passages returned"
              : "Evidence belongs beside the answer."}
          </h3>

          <p>
            {done
              ? "The backend returned no supporting passages. Treat any generated answer as unverified."
              : "Ask a question to inspect the agriculture passages returned by MuRIL and FAISS."}
          </p>

          <div className="source-label">
            Knowledge source <strong>Vikaspedia</strong>
          </div>
        </div>
      ) : (
        <>
          <p className="evidence-note">
            Retrieved context, not verified sentence-level citations.
            Similarity is not answer confidence.
          </p>

          <p className="retrieval-meta">
            {items.length} {items.length === 1 ? "passage" : "passages"}{" "}
            retrieved · Top-K 5 · FAISS cosine similarity
          </p>

          {items.map((item, index) => {
            const url = safeUrl(item.url);

            const score =
              typeof item.score === "number" && Number.isFinite(item.score)
                ? item.score
                : null;

            return (
              <details
                className="passage"
                key={`${item.chunk_id || "passage"}-${index}`}
                open={index === 0}
              >
                <summary>
                  <span className="rank">{item.rank ?? index + 1}</span>

                  <span>
                    {item.title || "Agriculture passage"}
                  </span>
                </summary>

                <div className="passage-body">
                  <p dir="auto">
                    {item.text || "Passage text was not returned."}
                  </p>

                  <div className="passage-meta">
                    <span>{item.source || "Source unspecified"}</span>

                    {score !== null && (
                      <span
                        title="Backend retrieval score; not an answer-confidence percentage"
                      >
                        Score {score.toFixed(3)}
                      </span>
                    )}
                  </div>

                  {url ? (
                    <a
                      href={url}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      Open original source ↗
                    </a>
                  ) : (
                    <span className="missing-source">
                      Source link unavailable
                    </span>
                  )}
                </div>
              </details>
            );
          })}
        </>
      )}
    </aside>
  );
}

function Assistant() {
  const [language, setLanguage] = useState("hi");
  const [modelChoice, setModelChoice] = useState("Gemini API");
  const [query, setQuery] = useState("");
  const [submitted, setSubmitted] = useState("");
  const [lastAttempt, setLastAttempt] = useState("");
  const [answer, setAnswer] = useState("");
  const [items, setItems] = useState([]);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");

  const request = useRef(0);
  const busy = useRef(false);
  const input = useRef(null);

  useEffect(() => {
    return () => {
      request.current += 1;
    };
  }, []);

  function clear() {
    if (busy.current) return;

    request.current += 1;

    setQuery("");
    setSubmitted("");
    setLastAttempt("");
    setAnswer("");
    setItems([]);
    setError("");
    setStatus("idle");

    input.current?.focus();
  }

  async function ask(event, overrideQuery) {
    event?.preventDefault();

    const text = (overrideQuery ?? query).trim();

    if (!text || busy.current) return;

    const id = ++request.current;

    busy.current = true;

    setLastAttempt(text);
    setSubmitted(text);
    setAnswer("");
    setItems([]);
    setError("");
    setStatus("loading");

    try {
      const result = await sendChat(text, modelChoice);

      if (id !== request.current) return;

      if (
        typeof result?.answer !== "string" ||
        !result.answer.trim()
      ) {
        throw new Error(
          "The server returned an empty answer. Please try again.",
        );
      }

      setAnswer(result.answer);

      setItems(
        Array.isArray(result.retrieved)
          ? result.retrieved.filter(
              (x) => x && typeof x === "object",
            )
          : [],
      );

      setStatus("success");
    } catch (e) {
      if (id === request.current) {
        setError(
          e?.message || "Could not reach the server.",
        );

        setStatus("error");
      }
    } finally {
      if (id === request.current) {
        busy.current = false;
      }
    }
  }

  return (
    <>
      <section className="intro">
        <div>
          <span className="eyebrow">
            AGRICULTURE KNOWLEDGE, IN YOUR LANGUAGE
          </span>

          <h1>
            A clearer answer.
            <br />
            <em>With the evidence behind it.</em>
          </h1>

          <p>
            Hindi agriculture questions, answered using a MuRIL encoder
            fine-tuned specifically for this domain — not a generic
            multilingual model. Kannada support is planned next.
          </p>
        </div>

        <div className="intro-stamp">
          <Leaf />

          <span>
            Fine-tuned for
            <br />
            <strong>agricultural meaning.</strong>
          </span>
        </div>
      </section>

      <div className="assistant-grid">
        <section
          className="question-panel"
          aria-label="Agriculture assistant"
        >
          <div className="panel-heading">
            <h2>Ask AgriSahayak</h2>

            <button
              className="text-button"
              disabled={status === "loading"}
              onClick={clear}
              type="button"
            >
              New question ↗
            </button>
          </div>

          <div className="language-bar">
            <span>Example language</span>

            <div
              className="segmented"
              aria-label="Example question language"
            >
              <button
                type="button"
                lang="hi"
                aria-pressed={language === "hi"}
                onClick={() => setLanguage("hi")}
              >
                हिन्दी
              </button>

              <button
                type="button"
                className="coming-soon"
                lang="kn"
                disabled
                aria-label="Kannada support coming soon"
              >
                ಕನ್ನಡ <span>Coming soon</span>
              </button>
            </div>
          </div>

          <div className="model-bar">
            <span>Generate answer with</span>

            <select
              value={modelChoice}
              onChange={(e) => setModelChoice(e.target.value)}
              aria-label="Generation model"
              disabled={status === "loading"}
            >
              <option value="Gemini API">Gemini API</option>
              <option value="Local Qwen">Local Qwen</option>
            </select>
          </div>

          {!submitted && (
            <div className="starter">
              <div className="assistant-mark">
                <Leaf />
              </div>

              <h3>What would you like to know?</h3>

              <p>
                Start with a crop, a cultivation practice or a soil
                question.
              </p>

              <div className="examples">
                {EXAMPLES[language].map(([label, text]) => (
                  <button
                    type="button"
                    key={text}
                    onClick={() => {
                      setQuery(text);
                      input.current?.focus();
                    }}
                  >
                    <span>{label}</span>
                    <strong lang={language}>{text}</strong>
                    <i aria-hidden="true">↗</i>
                  </button>
                ))}
              </div>
            </div>
          )}

          {submitted && (
            <div className="response" aria-live="polite">
              <div className="user-question">
                <span className="eyebrow">YOUR QUESTION</span>

                <p dir="auto">{submitted}</p>
              </div>

              {status === "loading" && (
                <div className="loading" role="status">
                  <span className="spinner" />
                  Preparing your answer and retrieved evidence…
                </div>
              )}

              {status === "error" && (
                <div className="error" role="alert">
                  <strong>
                    We couldn’t prepare an answer.
                  </strong>

                  <p>{error}</p>

                  <button
                    type="button"
                    onClick={() => ask(null, lastAttempt)}
                  >
                    Retry question
                  </button>
                </div>
              )}

              {status === "success" && (
                <>
                  <span className="answer-label">
                    AGRISAHAYAK · {modelChoice}
                  </span>

                  <div className="answer" dir="auto">
                    <Answer text={answer} />
                  </div>

                  {!items.length && (
                    <p className="warning">
                      No supporting passages were returned for this
                      answer.
                    </p>
                  )}
                </>
              )}
            </div>
          )}

          <form className="composer" onSubmit={ask}>
            <label
              htmlFor="agri-question"
              className="sr-only"
            >
              Your agriculture question
            </label>

            <textarea
              id="agri-question"
              ref={input}
              value={query}
              disabled={status === "loading"}
              onChange={(e) => setQuery(e.target.value)}
              dir="auto"
              rows="3"
              maxLength={400}
              placeholder={
                language === "hi"
                  ? "अपना कृषि प्रश्न यहाँ लिखें…"
                  : "ನಿಮ್ಮ ಕೃಷಿ ಪ್ರಶ್ನೆಯನ್ನು ಇಲ್ಲಿ ಬರೆಯಿರಿ…"
              }
            />

            <div className="composer-bottom">
              <span>
                Hindi · Text questions · Kannada coming soon
              </span>

              <span className="char-count">
                {query.length}/400
              </span>

              <button
                type="submit"
                className="primary"
                disabled={
                  !query.trim() ||
                  status === "loading"
                }
              >
                {status === "loading"
                  ? "Preparing…"
                  : "Ask question ↗"}
              </button>
            </div>
          </form>

          <p className="small-note">
            Responses are AI-generated. Review the retrieved evidence
            before applying advice.
          </p>
        </section>

        <Evidence
          items={items}
          done={status === "success"}
        />
      </div>

      <section className="contribution">
        <span className="eyebrow">
          THE RESEARCH CONTRIBUTION
        </span>

        <h2>
          Better representations start with domain knowledge.
        </h2>

        <p>
          The project fine-tunes MuRIL for agriculture-domain semantic
          retrieval. This assistant demonstrates the encoder inside a
          retrieval-augmented generation pipeline.
        </p>

        <p className="contribution-stat">
          On the held-out retrieval benchmark, Accuracy@1 increased
          from 21.46% with base MuRIL to 74.90% with MuRIL V3.
        </p>

        <div className="tech-row">
          <span>Fine-Tuned MuRIL</span>
          <span>768-D embeddings</span>
          <span>FAISS retrieval</span>
          <span>Gemini API</span>
        </div>
      </section>
    </>
  );
}

function Flow({
  title,
  description,
  steps,
  variant,
}) {
  return (
    <section className={`flow-card ${variant || ""}`}>
      <span className="eyebrow">{title}</span>

      <p>{description}</p>

      <ol>
        {steps.map(([name, note]) => (
          <li key={name}>
            <strong>{name}</strong>
            <span>{note}</span>
          </li>
        ))}
      </ol>
    </section>
  );
}

function WhyRAG() {
  return (
    <section className="why-rag">
      <span className="eyebrow">
        WHY RETRIEVAL, NOT JUST AN LLM
      </span>

      <h2>
        An LLM alone can't know what it was never shown.
      </h2>

      <p className="why-rag-intro">
        A farmer asks:{" "}
        <strong lang="hi" dir="auto">
          धान में ब्लास्ट रोग कैसे रोकें?
        </strong>{" "}
        <span className="translit">
          (How do I control blast disease in rice?)
        </span>
      </p>

      <div className="why-rag-grid">
        <div className="why-rag-card bad">
          <span className="why-rag-label">
            Without RAG — ungrounded
          </span>

          <p lang="hi" dir="auto">
            “यह एक सामान्य कृषि प्रश्न है। कृपया स्थानीय विशेषज्ञ से
            सलाह लें।”
          </p>

          <p className="why-rag-note">
            A generic response with no retrieved agricultural evidence
            behind it.
          </p>
        </div>

        <div className="why-rag-card good">
          <span className="why-rag-label">
            With RAG — grounded
          </span>

          <p lang="hi" dir="auto">
            “प्राप्त कृषि दस्तावेज़ के अनुसार, धान में ब्लास्ट रोग के
            नियंत्रण के लिए अनुशंसित उपाय अपनाएँ।”
          </p>

          <p className="why-rag-note">
            The answer is generated from the retrieved passage shown
            in the evidence panel.
          </p>
        </div>
      </div>

      <div className="open-book">
        <span className="why-rag-label">
          OPEN-BOOK ANALOGY
        </span>

        <p>
          Without RAG, the model answers from learned knowledge. With
          RAG, it receives relevant pages first and uses them as
          context for the answer.
        </p>
      </div>
    </section>
  );
}

const TOOLS = [
  [
    "AI / ML Core",
    "Python · PyTorch · Hugging Face Transformers · MuRIL",
  ],
  ["Retrieval & RAG", "FAISS · Gemini API"],
  ["Data Processing", "Pandas · NumPy · BeautifulSoup"],
  ["Development", "VS Code · Kaggle · Git + GitHub"],
];

function TechStack() {
  return (
    <section className="tech-stack">
      <span className="eyebrow">BUILT WITH</span>

      <div className="tech-stack-grid">
        {TOOLS.map(([category, items]) => (
          <div key={category} className="tech-stack-row">
            <span className="tech-stack-category">
              {category}
            </span>

            <span className="tech-stack-items">
              {items}
            </span>
          </div>
        ))}
      </div>
    </section>
  );
}

function RagInThreeSteps() {
  const steps = [
    [
      "1. Retrieval",
      "Find the relevant passages for the question.",
    ],
    [
      "2. Augmented",
      "Add those passages into the model prompt as context.",
    ],
    [
      "3. Generation",
      "The model writes the answer using that context.",
    ],
  ];

  return (
    <section
      className="rag-three-steps-section"
      aria-label="RAG in three steps"
    >
      <div className="section-kicker">THE RAG LOOP</div>

      <div className="rag-three-steps">
        {steps.map(([title, desc]) => (
          <div key={title} className="rag-step">
            <strong>{title}</strong>
            <span>{desc}</span>
          </div>
        ))}
      </div>
    </section>
  );
}

function Architecture() {
  return (
    <>
      <section className="how-it-works-hero">
        <span className="eyebrow">HOW IT WORKS</span>

        <h1>
          Retrieve first.
          <br />
          <em>Then generate.</em>
        </h1>

        <p>
          AgriSahayak retrieves relevant agriculture evidence first,
          then uses that context to generate a grounded response.
        </p>

        <div
          className="hero-flow"
          aria-label="RAG pipeline"
        >
          <span>Question</span>
          <i aria-hidden="true">→</i>
          <span>Retrieve</span>
          <i aria-hidden="true">→</i>
          <span>Augment</span>
          <i aria-hidden="true">→</i>
          <span>Generate</span>
        </div>
      </section>

      <WhyRAG />

      <RagInThreeSteps />

      <section className="page-title architecture-title">
        <span className="eyebrow">
          SYSTEM ARCHITECTURE
        </span>

        <h1>How the pipeline works.</h1>

        <p>
          The system separates knowledge-base preparation from the
          live retrieval-and-generation pipeline.
        </p>
      </section>

      <div className="architecture-grid">
        <Flow
          variant="offline"
          title="01 / OFFLINE — BUILD"
          description="Prepare the agriculture knowledge base once, before live questions arrive."
          steps={[
            [
              "Vikaspedia",
              "Collect agriculture content",
            ],
            [
              "Cleaning",
              "Remove noise and normalize the source text",
            ],
            [
              "Semantic chunking",
              "Split content into meaningful agriculture passages",
            ],
            [
              "Q–P pairs + hard negatives",
              "Build domain-specific retrieval training examples",
            ],
            [
              "Fine-Tuned MuRIL",
              "Train the retrieval encoder on agriculture examples",
            ],
            [
              "Passage embeddings",
              "Encode passages into 768-dimensional vectors",
            ],
            [
              "FAISS index",
              "Store vectors and map them back to source passages",
            ],
          ]}
        />

        <Flow
          variant="online"
          title="02 / ONLINE — LIVE QUERY"
          description="For every question, retrieve evidence first and then generate the response."
          steps={[
            [
              "Hindi question",
              "The user asks an agriculture question",
            ],
            [
              "Fine-Tuned MuRIL",
              "Create a 768-dimensional query embedding",
            ],
            [
              "FAISS similarity search",
              "Retrieve the Top-K relevant passages",
            ],
            [
              "Context assembly",
              "Combine the question with the retrieved evidence",
            ],
            [
              "Gemini API",
              "Generate the answer using the supplied context",
            ],
            [
              "Answer + sources",
              "Return the Hindi response with retrieved source passages",
            ],
          ]}
        />
      </div>

      <section className="contribution">
        <span className="eyebrow">
          WHAT IS FINE-TUNED?
        </span>

        <h2>The retrieval encoder: MuRIL.</h2>

        <p>
          The research contribution is the agriculture-aware
          fine-tuning of MuRIL for semantic retrieval. The encoder
          converts questions and passages into representations that
          can be compared for relevance.
        </p>

        <div
          className="fine-tune-flow"
          aria-label="MuRIL fine-tuning flow"
        >
          <div>
            <strong>Base MuRIL</strong>
            <span>Multilingual encoder</span>
          </div>

          <i aria-hidden="true">→</i>

          <div>
            <strong>
              Q–P pairs + hard negatives
            </strong>
            <span>
              Agriculture retrieval examples
            </span>
          </div>

          <i aria-hidden="true">→</i>

          <div>
            <strong>Fine-tuning</strong>
            <span>Domain-specific training</span>
          </div>

          <i aria-hidden="true">→</i>

          <div>
            <strong>Domain-aware MuRIL</strong>
            <span>
              Retrieval encoder for agriculture
            </span>
          </div>
        </div>

        <div className="role-strip">
          <div>
            <strong>MuRIL</strong>
            <span>retrieval encoder</span>
          </div>

          <div>
            <strong>FAISS</strong>
            <span>similarity search</span>
          </div>

          <div>
            <strong>Gemini</strong>
            <span>answer generation</span>
          </div>
        </div>

        <p className="small-note">
          This section describes the system design, not a live
          execution trace or a measured result.
        </p>
      </section>

      <div
        className="outcomes"
        aria-label="Project outcomes"
      >
        <div>
          <span className="eyebrow">OUTCOME 01</span>

          <strong>Working system</strong>

          <span>
            An end-to-end Hindi agriculture RAG assistant.
          </span>
        </div>

        <div>
          <span className="eyebrow">OUTCOME 02</span>

          <strong>Research contribution</strong>

          <span>
            A domain-aware fine-tuned MuRIL sentence encoder.
          </span>
        </div>
      </div>

      <TechStack />
    </>
  );
}

/*
 * Main component
 *
 * IMPORTANT:
 * The hooks and JSX that were previously outside a component
 * are now correctly contained inside ResearchUI.
 */
export default function ResearchUI() {
  const [tab, setTab] = useState("assistant");

  useEffect(() => {
    const onHashChange = () => {
      const hash = window.location.hash.slice(1);

      if (NAV.some(([id]) => id === hash)) {
        setTab(hash);
      } else {
        setTab("assistant");
      }
    };

    onHashChange();

    window.addEventListener(
      "hashchange",
      onHashChange,
    );

    return () => {
      window.removeEventListener(
        "hashchange",
        onHashChange,
      );
    };
  }, []);

  function navigate(id) {
    if (!NAV.some(([navId]) => navId === id)) {
      return;
    }

    setTab(id);

    if (window.location.hash !== `#${id}`) {
      window.history.pushState(
        null,
        "",
        `#${id}`,
      );
    }

    window.scrollTo({
      top: 0,
      behavior: "smooth",
    });
  }

  return (
    <div className="research-app">
      <a
        className="skip-link"
        href="#workspace"
      >
        Skip to content
      </a>

      <header className="site-header">
        <a
          href="#assistant"
          className="brand"
          onClick={(e) => {
            e.preventDefault();
            navigate("assistant");
          }}
        >
          <span className="brand-icon">
            <Leaf />
          </span>

          <span>
            AgriSahayak <b>AI</b>

            <small>
              AGRICULTURE · LANGUAGE · RETRIEVAL
            </small>
          </span>
        </a>

        <span className="project-tag">
          Final-year research project
        </span>
      </header>

      <div className="workspace-layout">
        <aside className="sidebar">
          <span className="eyebrow">WORKSPACE</span>

          <nav aria-label="Main navigation">
            {NAV.map(([id, title, n]) => (
              <button
                key={id}
                type="button"
                aria-current={
                  tab === id ? "page" : undefined
                }
                onClick={() => navigate(id)}
              >
                <span>{n}</span>

                {title}

                <i aria-hidden="true">↗</i>
              </button>
            ))}
          </nav>

          <div className="sidebar-note">
            <span className="eyebrow">
              OUR FOCUS
            </span>

            <p>
              Domain-aware
              <br />
              semantic retrieval.
            </p>

            <span>
              Fine-tuned MuRIL for Hindi agriculture
              retrieval,
              <br />
              built on a multilingual foundation model.
            </span>
          </div>

          <div className="sidebar-bottom">
            KNOWLEDGE SOURCE
            <br />

            <strong>
              Vikaspedia Agriculture
            </strong>

            <span>
              Hindi · Kannada coming soon
            </span>
          </div>
        </aside>

        <main
          id="workspace"
          className="workspace"
          tabIndex="-1"
        >
          {tab === "assistant" && <Assistant />}

          {tab === "architecture" && (
            <Architecture />
          )}

          {["analysis", "compare"].includes(tab) && (
            <>
              <section className="research-notice">
                <strong>
                  Research workspace
                </strong>

                <p>
                  Live searches use your existing backend.
                  Benchmark figures below are reported from
                  the held-out retrieval evaluation used in
                  this project. Similarity scores are not
                  answer confidence.
                </p>
              </section>

              <div className="legacy-research">
                {tab === "analysis" ? (
                  <AnalysisPage />
                ) : (
                  <ComparisonPage />
                )}
              </div>
            </>
          )}

          <footer>
            <div>
              AgriSahayak AI{" "}
              <span>
                Hindi-first. Domain-aware. Evidence-led.
              </span>

              <div className="footer-credit">
                Built by Pramod Pujar, Pratham Muragude,
                Prathamesh Patil &amp; Shrishail Patil ·
                Guided by Prof. Sunita N Karabasannavar
              </div>
            </div>
          </footer>
        </main>
      </div>
    </div>
  );
}
