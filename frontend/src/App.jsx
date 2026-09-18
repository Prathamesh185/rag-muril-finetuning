import { useState } from "react";

import {
  sendChat,
  retrievePassages,
  compareModels,
  getLabeledAnswer,
} from "./api/client";

/* ---------------------------------------------------------
AgriSahayak AI — Domain-Aware Sentence Encoder for
Agricultural RAG. Research demo frontend.

Palette
--forest #234D33 primary text / heading ink
--green #2F6D4F primary accent (fine-tuned model, actions)
--green-lt #E8F4EB selected state / evidence bg
--paper #FFFFFF base background
--mist #F6F8F5 section background
--line #E4E8E2 hairline borders
--ink #1C231E body text
--mute #667369 secondary text
--neutral #F1F1EF base MuRIL panel bg
--neutral-ink #57605A base MuRIL text
--amber #B8722B weaker-rank / caution accent
--------------------------------------------------------- */

const FONT_IMPORT = `@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');`;

const TOKENS = {
  forest: "#234D33",
  green: "#2F6D4F",
  greenLt: "#E8F4EB",
  greenLine: "#BFDFC9",
  paper: "#FFFFFF",
  mist: "#F6F8F5",
  line: "#E4E8E2",
  ink: "#1C231E",
  mute: "#667369",
  neutral: "#F1F1EF",
  neutralInk: "#57605A",
  neutralLine: "#DCDCD8",
  amber: "#B8722B",
  amberLt: "#FBF0E4",
};

const EXAMPLES = [
  "हल्दी की फसल में थ्रिप्स कीट से बचाव के लिए किस दवा का छिड़काव करें?",
  "धान में कीट नियंत्रण कैसे करें?",
  "जीवामृत कैसे तैयार करें?",
  "प्राकृतिक खेती के क्या लाभ हैं?",
];

const PIPELINE_STEPS = [
  "User Query",
  "Fine-Tuned MuRIL V3",
  "FAISS",
  "LLM",
  "Grounded Answer",
];

const RETRIEVAL_STEPS = [
  "User Query",
  "Fine-Tuned MuRIL V3",
  "Query Embedding",
  "FAISS",
  "Top-K Passages",
];

const ACCURACY_AT_1 = [
  { label: "Base MuRIL", value: 21.46 },
  { label: "MuRIL V2", value: 70.10 },
  { label: "BGE-M3", value: 69.75 },
  { label: "E5-base", value: 73.28 },
  { label: "MuRIL V3 (ours)", value: 74.90, highlight: true },
];

function AccuracyChart() {
  const max = 80;
  return (
    <div className="accuracy-chart" aria-label="Accuracy at 1 comparison">
      {ACCURACY_AT_1.map((row) => (
        <div key={row.label} className="accuracy-row">
          <span className="accuracy-label">{row.label}</span>
          <div className="accuracy-track">
            <div
              className={`accuracy-fill${row.highlight ? " highlight" : ""}`}
              style={{ width: `${(row.value / max) * 100}%` }}
            />
          </div>
          <span className="accuracy-value">{row.value.toFixed(2)}%</span>
        </div>
      ))}
    </div>
  );
}

const METRICS = [
  {
    metric: "Accuracy@1",
    base: "21.46%",
    v2: "70.10%",
    v3: "74.90%",
    e5: "73.28%",
    bge: "69.75%",
  },
  {
    metric: "Recall@5",
    base: "39.39%",
    v2: "93.18%",
    v3: "94.34%",
    e5: "93.03%",
    bge: "90.86%",
  },
  {
    metric: "Recall@10",
    base: "48.84%",
    v2: "96.62%",
    v3: "97.32%",
    e5: "96.11%",
    bge: "94.90%",
  },
  {
    metric: "MRR@10",
    base: "0.2919",
    v2: "0.7999",
    v3: "0.8331",
    e5: "0.8196",
    bge: "0.7889",
  },
  {
    metric: "NDCG@10",
    base: "0.3383",
    v2: "0.8410",
    v3: "0.8677",
    e5: "0.8546",
    bge: "0.8281",
  },
];

const DEMO_QUERIES = [
  {
    id: "soil-ph",
    query:
      "गेहूँ की अच्छी उपज के लिए प्रति हेक्टेयर कितनी नत्रजन की आवश्यकता होती है?",
  },
  {
    id: "sunflower-pest",
    query:
      "सूरजमुखी की फसल में कीट नियंत्रण के लिए किस रसायन का छिड़काव करना चाहिए?",
  },
  {
    id: "garlic-oil",
    query:
      "लहसुन का तेल किस विधि से प्राप्त किया जाता है?",
  }
];

/* ---------------- shared UI atoms ---------------- */

function Badge({ children }) {
  return (
    <span
      className="inline-flex items-center rounded-full px-3 py-1 text-xs font-medium"
      style={{
        background: TOKENS.greenLt,
        color: TOKENS.forest,
        border: `1px solid ${TOKENS.greenLine}`,
      }}
    >
      {children}
    </span>
  );
}

function SimBadge({ value }) {
  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-xs font-medium"
      style={{
        background: TOKENS.mist,
        color: TOKENS.ink,
        border: `1px solid ${TOKENS.line}`,
        fontFamily: "'JetBrains Mono', monospace",
      }}
    >
      cos sim {value.toFixed(2)}
    </span>
  );
}

function formatAnswer(text) {
  if (!text) return text;

  const parts = text.split(/\*\*(.*?)\*\*/g);

  return parts.map((part, i) =>
    i % 2 === 1 ? (
      <mark
        key={i}
        style={{
          background: TOKENS.greenLt,
          color: TOKENS.forest,
          padding: "1px 4px",
          borderRadius: "4px",
          fontWeight: 600,
        }}
      >
        {part}
      </mark>
    ) : (
      part
    )
  );
}

function extractHighlightedPhrases(text) {
  if (!text) return [];

  const matches = [...text.matchAll(/\*\*(.*?)\*\*/g)];

  return [
    ...new Set(
      matches
        .map((match) => match[1].trim())
        .filter((phrase) => phrase.length >= 3)
    ),
  ];
}

function escapeRegExp(text) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function getEvidenceCandidates(phrase) {
  const candidates = [];

  const clean = phrase
    .replace(/\s+/g, " ")
    .trim();

  if (!clean) return candidates;

  // 1. Try complete phrase first
  candidates.push(clean);

  // 2. Extract useful number/range expressions
  // Examples:
  // "6.5 से 7.5"
  // "20-25 दिन"
  // "18-20 दिनों"
  const numericMatches = clean.match(
    /\d+(?:\.\d+)?(?:\s*[-–]\s*\d+(?:\.\d+)?)?(?:\s+(?:से|तक)\s+\d+(?:\.\d+)?)?(?:\s+(?:दिन|दिनों|प्रतिशत|किलो|किग्रा|लीटर|हेक्टेयर))?/g
  );

  if (numericMatches) {
    numericMatches.forEach((match) => {
      if (match.trim().length >= 3) {
        candidates.push(match.trim());
      }
    });
  }

  // 3. Try meaningful word groups if full phrase failed
  const words = clean.split(/\s+/);

  // Prefer longer chunks first
  for (let size = Math.min(4, words.length); size >= 2; size--) {
    for (let i = 0; i <= words.length - size; i++) {
      const chunk = words.slice(i, i + size).join(" ");

      if (chunk.length >= 5) {
        candidates.push(chunk);
      }
    }
  }

  return [...new Set(candidates)];
}


function highlightEvidenceText(text, phrases) {
  if (
    !text ||
    !Array.isArray(phrases) ||
    phrases.length === 0
  ) {
    return text;
  }

  const matches = [];

  phrases.forEach((phrase) => {
    const candidates = getEvidenceCandidates(phrase);

    // Pick only the first / strongest candidate
    // that actually exists in this passage
    const found = candidates.find((candidate) =>
      text.toLowerCase().includes(candidate.toLowerCase())
    );

    if (found) {
      matches.push(found);
    }
  });

  const uniqueMatches = [
    ...new Set(matches)
  ].sort((a, b) => b.length - a.length);

  if (uniqueMatches.length === 0) {
    return text;
  }

  const regex = new RegExp(
    `(${uniqueMatches
      .map(escapeRegExp)
      .join("|")})`,
    "gi"
  );

  const matchSet = new Set(
    uniqueMatches.map((item) =>
      item.toLowerCase()
    )
  );

  return text.split(regex).map((part, index) => {
    if (matchSet.has(part.toLowerCase())) {
      return (
        <mark
          key={`${index}-${part}`}
          style={{
            background: "#BFE8C9",
            color: TOKENS.forest,
            padding: "1px 3px",
            borderRadius: 3,
            fontWeight: 600,
          }}
        >
          {part}
        </mark>
      );
    }

    return part;
  });
}

function EvidenceCard({ item, accent }) {
  const isGreen = accent === "green";

  const highlightIndex = item.passage.indexOf(item.highlight);

  const before =
    highlightIndex >= 0
      ? item.passage.slice(0, highlightIndex)
      : item.passage;

  const after =
    highlightIndex >= 0
      ? item.passage.slice(
        highlightIndex + item.highlight.length
      )
      : "";

  return (
    <div
      className="rounded-xl p-4 sm:p-5"
      style={{
        background: item.correct
          ? TOKENS.greenLt
          : TOKENS.paper,
        border: `1px solid ${item.correct
          ? TOKENS.greenLine
          : TOKENS.line
          }`,
      }}
    >
      <div className="flex items-center justify-between mb-2.5 flex-wrap gap-2">
        <div className="flex items-center gap-2">
          <span
            className="flex items-center justify-center w-6 h-6 rounded-full text-xs font-semibold"
            style={{
              background: isGreen
                ? TOKENS.green
                : TOKENS.neutralInk,
              color: "#fff",
            }}
          >
            {item.rank}
          </span>

          <span
            className="text-sm font-medium"
            style={{ color: TOKENS.ink }}
          >
            {item.title}
          </span>

          {item.correct && (
            <span
              className="text-xs font-semibold"
              style={{ color: TOKENS.green }}
            >
              ✓ ground truth
            </span>
          )}
        </div>

        <SimBadge value={item.similarity} />
      </div>

      <p
        dir="auto"
        className="text-[15px] leading-relaxed"
        style={{
          color: TOKENS.mute,
          fontFamily:
            "'Noto Sans Devanagari', 'Inter', sans-serif",
        }}
      >
        {highlightIndex >= 0 ? (
          <>
            {before}

            <mark
              style={{
                background: item.correct
                  ? "#CFEAD8"
                  : TOKENS.greenLt,
                color: TOKENS.forest,
                padding: "0 2px",
                borderRadius: 3,
              }}
            >
              {item.highlight}
            </mark>

            {after}
          </>
        ) : (
          item.passage
        )}
      </p>

      <div className="mt-3 flex items-center justify-between">
        <span
          className="text-xs"
          style={{ color: TOKENS.mute }}
        >
          Source: Vikaspedia
        </span>

        <a
          href="https://vikaspedia.in"
          target="_blank"
          rel="noreferrer"
          className="text-xs font-medium hover:underline cursor-pointer"
          style={{ color: TOKENS.green }}
        >
          View source →
        </a>
      </div>
    </div>
  );
}

const SOURCE_LABELS = {
  vikaspedia_hindi: "Vikaspedia (Hindi)",
};

function LiveEvidenceCard({
  item,
  groundTruthChunkId = null,
  supportingText = "",
  highlightPhrases = [],
}) {
  const isGroundTruth =
    Boolean(groundTruthChunkId) &&
    item.chunk_id === groundTruthChunkId;

  const phrasesToHighlight = [
    ...(Array.isArray(highlightPhrases)
      ? highlightPhrases
      : []),

    ...(isGroundTruth && supportingText
      ? [supportingText]
      : []),
  ];

  return (
    <div
      className="rounded-xl p-4 sm:p-5"
      style={{
        background: isGroundTruth
          ? TOKENS.greenLt
          : TOKENS.paper,
        border: `1px solid ${isGroundTruth
          ? TOKENS.greenLine
          : TOKENS.line
          }`,
      }}
    >
      <div className="flex items-center justify-between gap-3 mb-3 flex-wrap">
        <div className="flex items-center gap-2">
          <span
            className="flex items-center justify-center w-6 h-6 rounded-full text-xs font-semibold"
            style={{
              background: TOKENS.green,
              color: "#fff",
            }}
          >
            {item.rank}
          </span>

          <span
            className="text-sm font-medium"
            style={{ color: TOKENS.ink }}
          >
            {item.title || "Retrieved passage"}
          </span>

          {isGroundTruth && (
            <span
              className="text-xs font-semibold"
              style={{ color: TOKENS.green }}
            >
              ✓ Labeled ground truth
            </span>
          )}
        </div>

        <span
          className="inline-flex items-center rounded-md px-2 py-1 text-xs font-medium"
          title="Model-specific cosine similarity. Compare ranks, not scores across models."
          style={{
            background: TOKENS.mist,
            color: TOKENS.mute,
            border: `1px solid ${TOKENS.line}`,
            fontFamily: "'JetBrains Mono', monospace",
          }}
        >
          cosine&nbsp;{Number(item.score).toFixed(3)}
        </span>
      </div>

      <p
        dir="auto"
        className="text-[15px] leading-relaxed"
        style={{
          color: TOKENS.mute,
          fontFamily:
            "'Noto Sans Devanagari','Inter',sans-serif",
        }}
      >
        {highlightEvidenceText(
          item.text,
          phrasesToHighlight
        )}
      </p>

      <div className="mt-3 flex items-center justify-between gap-3">
        <span
          className="text-xs"
          style={{ color: TOKENS.mute }}
        >
          Source: {SOURCE_LABELS[item.source] || item.source || "Unknown"}
        </span>

        {item.url && (
          <a
            href={item.url}
            target="_blank"
            rel="noreferrer"
            className="text-xs font-medium hover:underline cursor-pointer"
            style={{ color: TOKENS.green }}
          >
            View source →
          </a>
        )}
      </div>
    </div>
  );
}

function PipelineDiagram({
  compact,
  wrap = false,
  noScroll = false,
  steps = PIPELINE_STEPS,
}) {
  return (
    <div
      className={
        wrap
          ? "flex flex-wrap items-center justify-center gap-1.5"
          : noScroll
            ? "flex items-center justify-center gap-1.5"
            : "flex items-center overflow-x-auto gap-1.5 pb-1"
      }
    >
      {steps.map((step, i) => {
        const isHero = step.startsWith("Fine-Tuned MuRIL");

        return (
          <div
            key={step}
            className="flex items-center flex-shrink-0"
          >
            <div
              className={`rounded-lg text-center whitespace-nowrap ${compact
                ? "px-3 py-2 text-xs"
                : "px-4 py-3 text-sm"
                }`}
              style={{
                background: isHero
                  ? TOKENS.green
                  : TOKENS.mist,
                color: isHero
                  ? "#fff"
                  : TOKENS.ink,
                border: `1px solid ${isHero
                  ? TOKENS.green
                  : TOKENS.line
                  }`,
                fontWeight: isHero
                  ? 600
                  : 500,
              }}
            >
              {step}
            </div>

            {i < steps.length - 1 && (
              <span
                className="mx-1.5 flex-shrink-0"
                style={{
                  color: TOKENS.mute,
                }}
              >
                →
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
}

function RankBar({
  label,
  rank,
  maxRank,
  isWinner,
  accent,
}) {
  const pct = Math.max(
    6,
    100 - ((rank - 1) / maxRank) * 100
  );

  const rankLabel =
    rank > maxRank
      ? `Not in Top-${maxRank}`
      : `#${rank}`;

  return (
    <div>
      <div className="flex items-baseline justify-between mb-1.5">
        <span
          className="text-sm font-medium"
          style={{ color: TOKENS.ink }}
        >
          {label}
        </span>

        <span
          className="text-sm font-semibold"
          style={{
            color: isWinner
              ? TOKENS.green
              : TOKENS.amber,
            fontFamily:
              "'JetBrains Mono', monospace",
          }}
        >
          {rankLabel} {isWinner && "✓"}
        </span>
      </div>

      <div
        className="h-2.5 rounded-full overflow-hidden"
        style={{
          background: TOKENS.mist,
        }}
      >
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{
            width: `${pct}%`,
            background: isWinner
              ? TOKENS.green
              : "#D7BB98",
          }}
        />
      </div>
    </div>
  );
}

function ResearchProofStrip() {
  const stats = [
    {
      label: "Accuracy@1",
      value: "74.90%",
      note: "Final MuRIL V3",
    },
    {
      label: "MRR@10",
      value: "0.8331",
      note: "Held-out test set",
    },
    {
      label: "OUTPERFORMS E5-BASE",
      value: "+1.62 pp",
      note: "Accuracy@1",
    },
  ];

  return (
    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
      {stats.map((stat) => (
        <div
          key={stat.label}
          className="rounded-xl border px-4 py-3 text-center"
          style={{
            borderColor: TOKENS.line,
            background: TOKENS.paper,
          }}
        >
          <p
            className="text-[11px] uppercase tracking-wide font-medium"
            style={{ color: TOKENS.mute }}
          >
            {stat.label}
          </p>

          <p
            className="text-xl font-semibold mt-1"
            style={{
              color: TOKENS.green,
              fontFamily: "'JetBrains Mono', monospace",
            }}
          >
            {stat.value}
          </p>

          <p
            className="text-[11px] mt-1"
            style={{ color: TOKENS.mute }}
          >
            {stat.note}
          </p>
        </div>
      ))}
    </div>
  );
}

/* ---------------- top nav ---------------- */

function NavBar({ tab, setTab }) {
  const tabs = [
    {
      id: "assistant",
      label: "AI Assistant",
    },
    {
      id: "analysis",
      label: "Retrieval Analysis",
    },
    {
      id: "compare",
      label: "Model Comparison",
    },
  ];

  return (
    <header
      className="sticky top-0 z-20 backdrop-blur"
      style={{
        background:
          "rgba(255,255,255,0.92)",
        borderBottom:
          `1px solid ${TOKENS.line}`,
      }}
    >
      <div className="max-w-6xl mx-auto px-5 sm:px-8 h-16 flex items-center justify-between">
        <div className="flex items-baseline gap-2.5">
          <span
            className="text-[19px] font-semibold tracking-tight"
            style={{
              color: TOKENS.forest,
              fontFamily:
                "'Newsreader', serif",
            }}
          >
            AgriSahayak AI
          </span>

          <span
            className="hidden md:inline text-xs"
            style={{ color: TOKENS.mute }}
          >
            Agriculture-Aware RAG using Fine-Tuned MuRIL V3
          </span>
        </div>

        <nav
          className="flex items-center gap-1 rounded-full p-1"
          style={{
            background: TOKENS.mist,
          }}
        >
          {tabs.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className="px-3.5 py-1.5 rounded-full text-sm font-medium transition-colors cursor-pointer"
              style={{
                background:
                  tab === t.id
                    ? TOKENS.paper
                    : "transparent",
                color:
                  tab === t.id
                    ? TOKENS.forest
                    : TOKENS.mute,
                boxShadow:
                  tab === t.id
                    ? "0 1px 2px rgba(35,77,51,0.12)"
                    : "none",
              }}
            >
              {t.label}
            </button>
          ))}
        </nav>
      </div>
    </header>
  );
}

/* ---------------- Page 1: AI Assistant ---------------- */

function AssistantPage({ setTab }) {
  const [query, setQuery] = useState("");
  const [submitted, setSubmitted] =
    useState(null);

  const [modelChoice, setModelChoice] =
    useState("Gemini API");

  const [answer, setAnswer] = useState("");
  const [retrieved, setRetrieved] =
    useState([]);

  const [loading, setLoading] =
    useState(false);

  const [error, setError] = useState("");

  const [evOpen, setEvOpen] =
    useState(false);

  const ask = async (q) => {
    const text = q ?? query;

    if (!text.trim() || loading) {
      return;
    }

    setSubmitted(text);
    setQuery(text);

    setAnswer("");
    setRetrieved([]);
    setError("");
    setEvOpen(false);
    setLoading(true);

    try {
      const result = await sendChat(
        text,
        modelChoice
      );

      setAnswer(result.answer);

      setRetrieved(
        Array.isArray(result.retrieved)
          ? result.retrieved
          : []
      );
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to get an answer from the backend."
      );
    } finally {
      setLoading(false);
    }
  };

  const resetQuestion = () => {
    setSubmitted(null);
    setQuery("");
    setAnswer("");
    setRetrieved([]);
    setError("");
    setEvOpen(false);
    setLoading(false);
  };

  const uniqueSources = Array.from(
    new Map(
      retrieved
        .filter(
          (item) =>
            typeof item.url ===
            "string" &&
            item.url.trim()
        )
        .map((item) => [
          item.url,
          {
            url: item.url,
            title:
              item.title ||
              item.source ||
              "Source",
            source: item.source,
          },
        ])
    ).values()
  );

  const answerHighlightPhrases =
    extractHighlightedPhrases(answer);

  if (!submitted) {
    return (
      <div className="relative">
        <div
          className="absolute inset-0 -z-10"
          style={{
            backgroundImage: `radial-gradient(${TOKENS.greenLine} 1px, transparent 1px)`,
            backgroundSize:
              "22px 22px",
            maskImage:
              "radial-gradient(ellipse 60% 50% at 50% 30%, black 40%, transparent 100%)",
            opacity: 0.5,
          }}
        />

        <div className="max-w-2xl mx-auto px-5 pt-16 pb-10 text-center">
          <h1
            className="text-[34px] sm:text-[42px] leading-tight font-medium mb-3"
            style={{
              color: TOKENS.forest,
              fontFamily:
                "'Newsreader', serif",
            }}
          >
            How can I help with agriculture?
          </h1>

          <p
            className="text-[15px] mb-6"
            style={{
              color: TOKENS.mute,
            }}
          >
            Ask agriculture questions and get
            answers grounded in retrieved
            agricultural knowledge.
          </p>

          <p
            className="text-xs font-medium mb-8 inline-flex items-center gap-1.5"
            style={{ color: TOKENS.green }}
          >
            Retrieval powered by MuRIL V3 — fine-tuned on Hindi agriculture data
            and outperforming E5-base and BGE-M3 on the held-out test set
          </p>

          <div className="flex flex-wrap justify-center gap-2 mb-8">
            {EXAMPLES.map((ex) => (
              <button
                key={ex}
                dir="auto"
                onClick={() => ask(ex)}
                className="px-3.5 py-2 rounded-full text-sm transition-colors cursor-pointer"
                style={{
                  background:
                    TOKENS.paper,
                  border:
                    `1px solid ${TOKENS.line}`,
                  color: TOKENS.ink,
                  fontFamily:
                    "'Noto Sans Devanagari','Inter',sans-serif",
                }}
              >
                {ex}
              </button>
            ))}
          </div>
          <div className="mb-3 flex items-center justify-between gap-3">
            <span
              className="text-xs font-medium"
              style={{
                color: TOKENS.mute,
              }}
            >
              Generate answer with
            </span>

            <select
              value={modelChoice}
              onChange={(e) =>
                setModelChoice(
                  e.target.value
                )
              }
              className="text-sm rounded-lg px-3 py-2 outline-none cursor-pointer"
              style={{
                background:
                  TOKENS.paper,
                color: TOKENS.ink,
                border:
                  `1px solid ${TOKENS.line}`,
              }}
            >
              <option value="Gemini API">
                Gemini API
              </option>

              <option value="Local Qwen">
                Local Qwen
              </option>
            </select>
          </div>

          <form
            onSubmit={(e) => {
              e.preventDefault();
              ask();
            }}
            className="flex items-center gap-2 rounded-2xl p-2 shadow-sm"
            style={{
              background:
                TOKENS.paper,
              border:
                `1px solid ${TOKENS.line}`,
            }}
          >
            <input
              dir="auto"
              value={query}
              onChange={(e) =>
                setQuery(e.target.value)
              }
              placeholder="Ask an agriculture question…"
              className="flex-1 bg-transparent outline-none px-3 py-2.5 text-[15px]"
              style={{
                color: TOKENS.ink,
                fontFamily:
                  "'Noto Sans Devanagari','Inter',sans-serif",
              }}
            />

            <button
              type="submit"
              disabled={loading}
              className="px-4 py-2.5 rounded-xl text-sm font-medium flex-shrink-0 cursor-pointer disabled:cursor-not-allowed"
              style={{
                background:
                  TOKENS.green,
                color: "#fff",
                opacity: loading
                  ? 0.7
                  : 1,
              }}
            >
              Ask
            </button>
          </form>

          <div className="mt-8 mb-6">
            <ResearchProofStrip />
          </div>

          <div
            className="rounded-2xl p-4 sm:p-5 mb-4 text-left"
            style={{
              background: TOKENS.mist,
              border: `1px solid ${TOKENS.line}`,
            }}
          >
            <div
              className="text-xs font-medium mb-3"
              style={{ color: TOKENS.mute }}
            >
              How AgriSahayak works
            </div>
            <PipelineDiagram compact noScroll steps={PIPELINE_STEPS} />
          </div>

          <p
            className="text-xs text-center"
            style={{ color: TOKENS.mute }}
          >
            See Base MuRIL vs Fine-Tuned MuRIL V3 side-by-side in{" "}
            <button
              type="button"
              onClick={() => setTab("compare")}
              className="font-semibold hover:underline cursor-pointer"
              style={{ color: TOKENS.green }}
            >
              Model Comparison →
            </button>
          </p>

          <p
            className="text-[11px] mt-4"
            style={{ color: TOKENS.mute }}
          >
            Built and evaluated on 20,141 Hindi agriculture question–passage pairs.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-2xl mx-auto px-5 py-10">
      <button
        onClick={resetQuestion}
        className="text-xs font-medium mb-6 cursor-pointer"
        style={{
          color: TOKENS.mute,
        }}
      >
        ← New question
      </button>

      <div className="mb-6">
        <div
          className="text-xs font-medium mb-1.5"
          style={{
            color: TOKENS.mute,
          }}
        >
          Your question
        </div>

        <p
          dir="auto"
          className="text-lg font-medium"
          style={{
            color: TOKENS.forest,
            fontFamily:
              "'Noto Sans Devanagari','Inter',sans-serif",
          }}
        >
          {submitted}
        </p>

        <div
          className="text-xs mt-2"
          style={{
            color: TOKENS.mute,
          }}
        >
          Generated with: {modelChoice}
        </div>
      </div>

      {loading && (
        <div
          className="rounded-xl p-6 flex items-center gap-3"
          style={{
            background:
              TOKENS.mist,
            border:
              `1px solid ${TOKENS.line}`,
          }}
        >
          <span
            className="w-4 h-4 rounded-full animate-spin flex-shrink-0"
            style={{
              border:
                `2px solid ${TOKENS.greenLine}`,
              borderTopColor:
                TOKENS.green,
            }}
          />

          <span
            className="text-sm"
            style={{
              color: TOKENS.mute,
            }}
          >
            Retrieving agricultural knowledge
            and generating answer…
          </span>
        </div>
      )}

      {!loading && error && (
        <div
          className="rounded-xl p-5"
          style={{
            background:
              TOKENS.amberLt,
            border:
              `1px solid ${TOKENS.amber}`,
            color: TOKENS.ink,
          }}
        >
          <div className="text-sm font-medium mb-1">
            Unable to get an answer
          </div>

          <div className="text-sm">
            {error}
          </div>
        </div>
      )}

      {!loading &&
        !error &&
        answer && (
          <>
            <div
              className="rounded-2xl p-5 sm:p-6 mb-4"
              style={{
                background:
                  TOKENS.paper,
                border:
                  `1px solid ${TOKENS.line}`,
                boxShadow:
                  "0 1px 3px rgba(28,35,30,0.05)",
              }}
            >
              <div className="mb-3">
                <Badge>
                  Generated from retrieved
                  agricultural evidence
                </Badge>
              </div>

              <p
                dir="auto"
                className="text-[15px] leading-relaxed mb-4"
                style={{
                  color: TOKENS.ink,
                  fontFamily:
                    "'Noto Sans Devanagari','Inter',sans-serif",
                  whiteSpace: "pre-line",
                }}
              >
                {formatAnswer(answer)}
              </p>

              {uniqueSources.length >
                0 && (
                  <div
                    className="pt-4 flex flex-wrap items-center gap-3"
                    style={{
                      borderTop:
                        `1px solid ${TOKENS.line}`,
                    }}
                  >
                    <span
                      className="text-xs font-medium"
                      style={{
                        color:
                          TOKENS.mute,
                      }}
                    >
                      Sources
                    </span>

                    {uniqueSources.map(
                      (
                        source,
                        index
                      ) => (
                        <a
                          key={
                            source.url
                          }
                          href={
                            source.url
                          }
                          target="_blank"
                          rel="noreferrer"
                          dir="auto"
                          className="text-xs font-medium px-2.5 py-1 rounded-full hover:underline cursor-pointer"
                          style={{
                            background:
                              TOKENS.greenLt,
                            color:
                              TOKENS.forest,
                          }}
                        >
                          [{index + 1}]{" "}
                          {
                            source.title
                          }
                        </a>
                      )
                    )}
                  </div>
                )}
            </div>

            {retrieved.length >
              0 && (
                <>
                  <button
                    onClick={() =>
                      setEvOpen(
                        (v) => !v
                      )
                    }
                    className="w-full flex items-center justify-between rounded-xl px-5 py-3.5 text-sm font-medium mb-3 cursor-pointer"
                    style={{
                      background:
                        TOKENS.mist,
                      color:
                        TOKENS.forest,
                      border:
                        `1px solid ${TOKENS.line}`,
                    }}
                  >
                    <span>
                      {evOpen
                        ? "▾"
                        : "▸"}{" "}
                      View retrieved
                      evidence (
                      {
                        retrieved.length
                      }
                      )
                    </span>

                    <span
                      className="text-xs font-normal"
                      style={{
                        color:
                          TOKENS.mute,
                      }}
                    >
                      via Fine-Tuned MuRIL V3 + FAISS
                    </span>
                  </button>

                  {evOpen && (
                    <div className="space-y-3">
                      {retrieved.map(
                        (item) => (
                          <LiveEvidenceCard
                            key={
                              item.chunk_id ??
                              `${item.rank}-${item.title}`
                            }
                            item={item}
                            highlightPhrases={
                              answerHighlightPhrases
                            }
                          />
                        )
                      )}
                    </div>
                  )}

                </>
              )}
          </>
        )}
    </div>
  );
}

/* ---------------- Page 2: Retrieval Analysis ---------------- */

export function AnalysisPage() {
  const [query, setQuery] =
    useState(DEMO_QUERIES[0].query);

  const [activeQuery, setActiveQuery] =
    useState("");

  const [retrieved, setRetrieved] =
    useState([]);

  const [loading, setLoading] =
    useState(false);

  const [error, setError] =
    useState("");

  const runRetrieval = async (
    question
  ) => {
    const text = question ?? query;

    if (!text.trim() || loading) {
      return;
    }

    setQuery(text);
    setActiveQuery(text);
    setRetrieved([]);
    setError("");
    setLoading(true);

    try {
      const result =
        await retrievePassages(
          text,
          5
        );

      setRetrieved(
        Array.isArray(
          result.retrieved
        )
          ? result.retrieved
          : []
      );
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to retrieve passages."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto px-5 py-10">
      <div className="mb-8">
        <h2
          className="text-2xl font-medium mb-1.5"
          style={{
            color: TOKENS.forest,
            fontFamily:
              "'Newsreader', serif",
          }}
        >
          Retrieval Analysis
        </h2>

        <p
          className="text-sm"
          style={{
            color: TOKENS.mute,
          }}
        >
          Inspect the real passages retrieved by
          Fine-Tuned MuRIL and FAISS.
        </p>
      </div>

      <div
        className="rounded-2xl p-5 mb-5"
        style={{
          background:
            TOKENS.paper,
          border:
            `1px solid ${TOKENS.line}`,
        }}
      >
        <form
          onSubmit={(e) => {
            e.preventDefault();
            runRetrieval();
          }}
          className="flex flex-col sm:flex-row gap-2"
        >
          <input
            dir="auto"
            value={query}
            onChange={(e) =>
              setQuery(
                e.target.value
              )
            }
            placeholder="Enter an agriculture question"
            className="flex-1 rounded-xl px-4 py-2.5 text-[15px] outline-none"
            style={{
              border:
                `1px solid ${TOKENS.line}`,
              color: TOKENS.ink,
              fontFamily:
                "'Noto Sans Devanagari','Inter',sans-serif",
            }}
          />

          <button
            type="submit"
            disabled={loading}
            className="px-5 py-2.5 rounded-xl text-sm font-medium cursor-pointer disabled:cursor-not-allowed"
            style={{
              background:
                TOKENS.green,
              color: "#fff",
              opacity: loading
                ? 0.7
                : 1,
            }}
          >
            Retrieve
          </button>
        </form>

        <div className="flex flex-wrap items-center gap-2 mt-3">
          <span
            className="text-xs"
            style={{
              color: TOKENS.mute,
            }}
          >
            Try an example:
          </span>

          {DEMO_QUERIES.map(
            (d) => (
              <button
                key={d.id}
                dir="auto"
                onClick={() =>
                  runRetrieval(
                    d.query
                  )
                }
                className="text-xs px-2.5 py-1 rounded-full cursor-pointer"
                style={{
                  background:
                    TOKENS.mist,
                  color:
                    TOKENS.forest,
                  border:
                    `1px solid ${TOKENS.line}`,
                  fontFamily:
                    "'Noto Sans Devanagari','Inter',sans-serif",
                }}
              >
                {d.query}
              </button>
            )
          )}
        </div>
      </div>

      <div
        className="rounded-2xl p-5 sm:p-6 mb-8"
        style={{
          background:
            TOKENS.mist,
          border:
            `1px solid ${TOKENS.line}`,
        }}
      >
        <div
          className="text-xs font-medium mb-4"
          style={{
            color: TOKENS.mute,
          }}
        >
          Query → retrieval pipeline
        </div>

        <PipelineDiagram
          steps={RETRIEVAL_STEPS}
        />

        <p
          className="text-xs mt-4"
          style={{
            color: TOKENS.mute,
          }}
        >
          <span
            style={{
              color: TOKENS.green,
              fontWeight: 600,
            }}
          >
            Fine-Tuned MuRIL
          </span>{" "}
          is this project's core research
          contribution — it produces the
          domain-aware query embedding used
          for FAISS retrieval.
        </p>
      </div>

      {loading && (
        <div
          className="rounded-xl p-6 flex items-center gap-3"
          style={{
            background:
              TOKENS.mist,
            border:
              `1px solid ${TOKENS.line}`,
          }}
        >
          <span
            className="w-4 h-4 rounded-full animate-spin flex-shrink-0"
            style={{
              border:
                `2px solid ${TOKENS.greenLine}`,
              borderTopColor:
                TOKENS.green,
            }}
          />

          <span
            className="text-sm"
            style={{
              color: TOKENS.mute,
            }}
          >
            Searching the Fine-Tuned MuRIL
            FAISS index…
          </span>
        </div>
      )}

      {!loading && error && (
        <div
          className="rounded-xl p-5"
          style={{
            background:
              TOKENS.amberLt,
            border:
              `1px solid ${TOKENS.amber}`,
          }}
        >
          <div
            className="text-sm font-medium"
            style={{
              color: TOKENS.ink,
            }}
          >
            Retrieval failed
          </div>

          <div
            className="text-sm mt-1"
            style={{
              color: TOKENS.mute,
            }}
          >
            {error}
          </div>
        </div>
      )}

      {!loading &&
        !error &&
        retrieved.length > 0 && (
          <>
            <div className="mb-4">
              <h3
                className="text-sm font-semibold mb-1"
                style={{
                  color: TOKENS.ink,
                }}
              >
                Top retrieved passages
              </h3>

              <p
                dir="auto"
                className="text-xs"
                style={{
                  color: TOKENS.mute,
                  fontFamily:
                    "'Noto Sans Devanagari','Inter',sans-serif",
                }}
              >
                Query: {activeQuery}
              </p>
            </div>

            <div className="space-y-3">
              {retrieved.map(
                (item) => (
                  <LiveEvidenceCard
                    key={
                      item.chunk_id ??
                      `${item.rank}-${item.title}`
                    }
                    item={item}
                  />
                )
              )}
            </div>
          </>
        )}

      {!loading &&
        !error &&
        retrieved.length === 0 &&
        !activeQuery && (
          <div
            className="rounded-2xl p-10 text-center"
            style={{
              background:
                TOKENS.mist,
              border:
                `1px dashed ${TOKENS.line}`,
            }}
          >
            <p
              className="text-sm"
              style={{
                color: TOKENS.mute,
              }}
            >
              Enter a question to inspect the
              real FAISS retrieval results.
            </p>
          </div>
        )}
    </div>
  );
}

/* ---------------- Page 3: Model Comparison ---------------- */

export function ComparisonPage() {
  const [customQuery, setCustomQuery] =
    useState("");

  const [active, setActive] =
    useState(null);

  const [loading, setLoading] =
    useState(false);

  const [error, setError] =
    useState("");

  const [labeledAnswer, setLabeledAnswer] =
    useState("");

  const [
    labeledAnswerLoading,
    setLabeledAnswerLoading,
  ] = useState(false);

  const [
    labeledSupportingText,
    setLabeledSupportingText,
  ] = useState("");

  const run = async (question) => {
    const text = question?.trim();

    if (!text || loading) {
      return;
    }

    setCustomQuery(text);
    setActive(null);
    setError("");
    setLabeledAnswer("");
    setLabeledAnswerLoading(false);
    setLabeledSupportingText("");
    setLoading(true);


    try {
      const result =
        await compareModels(
          text,
          5
        );

      setActive({
        query: text,

        base: Array.isArray(
          result.base
        )
          ? result.base
          : [],

        finetuned: Array.isArray(
          result.finetuned
        )
          ? result.finetuned
          : [],

        groundTruthAvailable:
          result.ground_truth_available,

        groundTruthInLiveCorpus:
          result.ground_truth_in_live_corpus,

        groundTruthChunkId:
          result.ground_truth_chunk_id,

        baseRank:
          result.base_rank,

        finetunedRank:
          result.finetuned_rank,

        rankCutoff:
          result.rank_cutoff,
      });

      // Retrieval comparison is finished.
      // Gemini answer generation is separate.
      if (result.ground_truth_available) {
        setLabeledAnswerLoading(true);

        getLabeledAnswer(text)
          .then((answerResult) => {
            if (
              answerResult.available &&
              answerResult.answer
            ) {
              setLabeledAnswer(
                answerResult.answer
              );

              setLabeledSupportingText(
                answerResult.supporting_text || ""
              );
            }
          })
          .catch((answerError) => {
            console.error(
              "Labeled answer generation failed:",
              answerError
            );
          })
          .finally(() => {
            setLabeledAnswerLoading(false);
          });
      }
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to compare the models."
      );
    } finally {
      setLoading(false);
    }
  };

  const runCustom = () => {
    run(customQuery);
  };

  const rankCutoff = active?.rankCutoff ?? 100;
  const baseRankForBar =
    active?.baseRank ?? rankCutoff + 1;
  const finetunedRankForBar =
    active?.finetunedRank ?? rankCutoff + 1;

  const baseWins =
    baseRankForBar < finetunedRankForBar;
  const v3Wins =
    finetunedRankForBar < baseRankForBar;

  return (
    <div className="max-w-5xl mx-auto px-5 py-10">
      <div className="mb-8">
        <h2
          className="text-2xl font-medium mb-1.5"
          style={{
            color: TOKENS.forest,
            fontFamily:
              "'Newsreader', serif",
          }}
        >
          Base MuRIL vs Fine-Tuned MuRIL V3
        </h2>

        <p
          className="text-sm"
          style={{
            color: TOKENS.mute,
          }}
        >
          Compare live retrieval results for
          the same agriculture query.
        </p>
      </div>

      <div
        className="rounded-2xl p-5 mb-4"
        style={{
          background:
            TOKENS.paper,
          border:
            `1px solid ${TOKENS.line}`,
        }}
      >
        <form
          onSubmit={(e) => {
            e.preventDefault();
            runCustom();
          }}
          className="flex flex-col sm:flex-row gap-2"
        >
          <input
            dir="auto"
            value={customQuery}
            onChange={(e) =>
              setCustomQuery(
                e.target.value
              )
            }
            placeholder="हल्दी की फसल में थ्रिप्स कीट से बचाव के लिए किस दवा का छिड़काव करें?"
            className="flex-1 rounded-xl px-4 py-2.5 text-[15px] outline-none"
            style={{
              border:
                `1px solid ${TOKENS.line}`,
              color: TOKENS.ink,
              fontFamily:
                "'Noto Sans Devanagari','Inter',sans-serif",
            }}
          />

          <button
            type="submit"
            disabled={
              loading ||
              !customQuery.trim()
            }
            className="px-5 py-2.5 rounded-xl text-sm font-medium cursor-pointer disabled:opacity-50"
            style={{
              background:
                TOKENS.green,
              color: "#fff",
            }}
          >
            {loading
              ? "Comparing..."
              : "Compare models"}
          </button>
        </form>

        <div className="flex flex-wrap items-center gap-2 mt-3">
          <span
            className="text-xs"
            style={{
              color: TOKENS.mute,
            }}
          >
            Try an example:
          </span>

          {DEMO_QUERIES.map(
            (d) => (
              <button
                key={d.id}
                type="button"
                dir="auto"
                disabled={loading}
                onClick={() =>
                  run(d.query)
                }
                className="text-xs px-2.5 py-1 rounded-full cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
                style={{
                  background:
                    TOKENS.mist,
                  color:
                    TOKENS.forest,
                  border:
                    `1px solid ${TOKENS.line}`,
                  fontFamily:
                    "'Noto Sans Devanagari','Inter',sans-serif",
                }}
              >
                {d.query}
              </button>
            )
          )}
        </div>
      </div>

      {loading && (
        <div
          className="rounded-xl p-6 flex items-center gap-3 mb-6"
          style={{
            background:
              TOKENS.mist,
            border:
              `1px solid ${TOKENS.line}`,
          }}
        >
          <span
            className="w-4 h-4 rounded-full animate-spin flex-shrink-0"
            style={{
              border:
                `2px solid ${TOKENS.greenLine}`,
              borderTopColor:
                TOKENS.green,
            }}
          />

          <span
            className="text-sm"
            style={{
              color: TOKENS.mute,
            }}
          >
            Running the same query through
            Base MuRIL and Fine-Tuned MuRIL…
          </span>
        </div>
      )}

      {error && !loading && (
        <div
          className="rounded-xl p-4 mb-6 text-sm"
          style={{
            background: "#FFF7ED",
            border:
              "1px solid #FED7AA",
            color: "#9A3412",
          }}
        >
          {error}
        </div>
      )}

      {active && !loading && (
        <>
          <div
            className="rounded-2xl overflow-hidden mb-5"
            style={{
              background: TOKENS.paper,
              border: `1px solid ${TOKENS.line}`,
            }}
          >
            {/* Query */}
            <div className="p-5 sm:p-6">
              <div
                className="text-xs font-medium mb-2"
                style={{
                  color: TOKENS.mute,
                }}
              >
                Query
              </div>

              <div
                dir="auto"
                className="text-[17px] sm:text-[18px] font-medium leading-relaxed"
                style={{
                  color: TOKENS.ink,
                  fontFamily:
                    "'Noto Sans Devanagari','Inter',sans-serif",
                }}
              >
                {active.query}
              </div>
            </div>

            {active.groundTruthAvailable && (
              <>
                <div
                  style={{
                    borderTop:
                      `1px solid ${TOKENS.line}`,
                  }}
                />

                {/* Generated answer */}
                <div
                  className="p-5 sm:p-6"
                  style={{
                    background: TOKENS.greenLt,
                  }}
                >
                  <div
                    className="text-xs font-semibold mb-2"
                    style={{
                      color: TOKENS.green,
                    }}
                  >
                    Generated answer from labeled passage
                  </div>

                  {labeledAnswerLoading ? (
                    <div className="flex items-center gap-2">
                      <span
                        className="w-3.5 h-3.5 rounded-full animate-spin flex-shrink-0"
                        style={{
                          border:
                            `2px solid ${TOKENS.greenLine}`,
                          borderTopColor:
                            TOKENS.green,
                        }}
                      />

                      <span
                        className="text-sm"
                        style={{
                          color: TOKENS.mute,
                        }}
                      >
                        Generating a short answer from the
                        labeled dataset passage...
                      </span>
                    </div>
                  ) : labeledAnswer ? (
                    <p
                      dir="auto"
                      className="text-[16px] leading-relaxed font-medium"
                      style={{
                        color: TOKENS.ink,
                        fontFamily:
                          "'Noto Sans Devanagari','Inter',sans-serif",
                      }}
                    >
                      {formatAnswer(labeledAnswer)}
                    </p>
                  ) : (
                    <p
                      className="text-sm"
                      style={{
                        color: TOKENS.mute,
                      }}
                    >
                      A generated answer is not available
                      for this labeled passage.
                    </p>
                  )}

                  <div
                    className="text-xs mt-3"
                    style={{
                      color: TOKENS.mute,
                    }}
                  >
                    Gemini-generated using only the labeled
                    positive passage for this dataset question.
                  </div>
                </div>
              </>
            )}
          </div>
          {/* Known labeled passage — live full-corpus rank */}
          {active.groundTruthAvailable && (
            <div
              className="rounded-2xl p-5 mb-5"
              style={{
                background:
                  TOKENS.paper,
                border:
                  `1px solid ${TOKENS.line}`,
              }}
            >
              <div
                className="text-xs font-medium mb-2"
                style={{
                  color: TOKENS.mute,
                }}
              >
                Known labeled passage — live
                full-corpus rank
              </div>

              <p
                className="text-xs mb-4"
                style={{
                  color: TOKENS.mute,
                }}
              >
                Live ranks are computed against
                the current 17,391-passage FAISS
                corpus and may differ from the
                held-out evaluation ranks shown
                below.
              </p>

              {!active.groundTruthInLiveCorpus ? (
                <div
                  className="text-sm"
                  style={{
                    color: TOKENS.amber,
                  }}
                >
                  The labeled ground-truth passage
                  is not present in the current live
                  corpus, so a live rank cannot be
                  computed for this question.
                </div>
              ) : (
                <div className="grid sm:grid-cols-2 gap-5">
                  <RankBar
                    label="Base MuRIL"
                    rank={baseRankForBar}
                    maxRank={rankCutoff}
                    isWinner={baseWins}
                  />

                  <RankBar
                    label="Fine-Tuned MuRIL V3"
                    rank={finetunedRankForBar}
                    maxRank={rankCutoff}
                    isWinner={v3Wins}
                  />
                </div>
              )}
            </div>
          )}

          <p
            className="text-xs mb-6"
            style={{
              color: TOKENS.mute,
            }}
          >
            Similarity scores belong to each
            model's own embedding space. Compare
            passage relevance and ranking rather
            than comparing Base and Fine-Tuned
            scores directly.
          </p>

          <div className="grid md:grid-cols-2 gap-5">
            {/* Base MuRIL */}
            <div>
              <div className="flex items-center gap-2 mb-3">
                <span
                  className="w-2.5 h-2.5 rounded-full"
                  style={{
                    background:
                      TOKENS.neutralInk,
                  }}
                />

                <span
                  className="text-sm font-semibold"
                  style={{
                    color: TOKENS.ink,
                  }}
                >
                  Base MuRIL
                </span>

                <span
                  className="text-xs"
                  style={{
                    color: TOKENS.mute,
                  }}
                >
                  general encoder
                </span>
              </div>

              <div className="space-y-3">
                {active.base.map(
                  (item) => (
                    <LiveEvidenceCard
                      key={
                        item.chunk_id ??
                        `base-${item.rank}`
                      }
                      item={item}
                      groundTruthChunkId={
                        active.groundTruthChunkId
                      }
                      supportingText={
                        labeledSupportingText
                      }
                    />
                  )
                )}
              </div>
            </div>

            {/* Fine-Tuned MuRIL */}
            <div>
              <div className="flex items-center gap-2 mb-3">
                <span
                  className="w-2.5 h-2.5 rounded-full"
                  style={{
                    background:
                      TOKENS.green,
                  }}
                />

                <span
                  className="text-sm font-semibold"
                  style={{
                    color:
                      TOKENS.forest,
                  }}
                >
                  Fine-Tuned MuRIL V3
                </span>

                <span
                  className="text-xs"
                  style={{
                    color:
                      TOKENS.mute,
                  }}
                >
                  MNRL + hard negatives
                </span>
              </div>

              <div className="space-y-3">
                {active.finetuned.map(
                  (item) => (
                    <LiveEvidenceCard
                      key={
                        item.chunk_id ??
                        `finetuned-${item.rank}`
                      }
                      item={item}
                      groundTruthChunkId={
                        active.groundTruthChunkId
                      }
                      supportingText={
                        labeledSupportingText
                      }
                    />
                  )
                )}
              </div>
            </div>
          </div>
        </>
      )}

      {!active &&
        !loading &&
        !error && (
          <div
            className="rounded-2xl p-10 text-center"
            style={{
              background:
                TOKENS.mist,
              border:
                `1px dashed ${TOKENS.line}`,
            }}
          >
            <p
              className="text-sm"
              style={{
                color: TOKENS.mute,
              }}
            >
              Enter a query or pick an example
              to retrieve passages from both
              models.
            </p>
          </div>
        )}

      {/* Why fine-tune */}
      <div
        className="mt-12 rounded-2xl p-6 sm:p-8"
        style={{
          background:
            TOKENS.paper,
          border:
            `1px solid ${TOKENS.line}`,
        }}
      >
        <h3
          className="text-base font-semibold mb-2"
          style={{
            color: TOKENS.forest,
          }}
        >
          Why fine-tune MuRIL?
        </h3>

        <p
          className="text-sm leading-relaxed mb-6"
          style={{
            color: TOKENS.mute,
            maxWidth: 640,
          }}
        >
          Base MuRIL understands
          Indian-language text generally.
          Fine-tuning teaches the encoder to
          place agriculture-related questions
          closer to their relevant agricultural
          passages in the embedding space.
        </p>

        <div className="grid sm:grid-cols-2 gap-6">
          <div>
            <div
              className="text-xs font-medium mb-3"
              style={{
                color: TOKENS.mute,
              }}
            >
              Before fine-tuning
            </div>

            <svg
              viewBox="0 0 260 60"
              className="w-full max-w-[260px]"
            >
              <circle
                cx="30"
                cy="30"
                r="7"
                fill={
                  TOKENS.neutralInk
                }
              />

              <text
                x="30"
                y="52"
                fontSize="9"
                textAnchor="middle"
                fill={TOKENS.mute}
              >
                Question
              </text>

              <line
                x1="42"
                y1="30"
                x2="200"
                y2="30"
                stroke={
                  TOKENS.neutralLine
                }
                strokeWidth="2"
                strokeDasharray="3 4"
              />

              <circle
                cx="212"
                cy="30"
                r="7"
                fill="#D7BB98"
              />

              <text
                x="205"
                y="52"
                fontSize="9"
                textAnchor="middle"
                fill={TOKENS.mute}
              >
                Passage
              </text>
            </svg>
          </div>

          <div>
            <div
              className="text-xs font-medium mb-3"
              style={{
                color: TOKENS.mute,
              }}
            >
              After fine-tuning
            </div>

            <svg
              viewBox="0 0 260 60"
              className="w-full max-w-[260px]"
            >
              <circle
                cx="90"
                cy="30"
                r="7"
                fill={TOKENS.green}
              />

              <text
                x="90"
                y="52"
                fontSize="9"
                textAnchor="middle"
                fill={TOKENS.mute}
              >
                Question
              </text>

              <line
                x1="102"
                y1="30"
                x2="128"
                y2="30"
                stroke={TOKENS.green}
                strokeWidth="2"
              />

              <circle
                cx="140"
                cy="30"
                r="7"
                fill={TOKENS.forest}
              />

              <text
                x="140"
                y="52"
                fontSize="9"
                textAnchor="middle"
                fill={TOKENS.mute}
              >
                Passage
              </text>
            </svg>
          </div>
        </div>
      </div>

      {/* Existing metrics section */}
      <div
        className="mt-6 rounded-2xl p-6 sm:p-8"
        style={{
          background:
            TOKENS.paper,
          border:
            `1px solid ${TOKENS.line}`,
        }}
      >
        <p className="accuracy-chart-title">Accuracy@1 across models</p>
        <AccuracyChart />

        <h3
          className="text-base font-semibold mb-4"
          style={{
            color: TOKENS.forest,
          }}
        >
          Held-out agriculture retrieval evaluation
        </h3>

        <div className="overflow-x-auto">
          <table className="w-full text-sm border-collapse">
            <thead>
              <tr
                style={{
                  borderBottom: `1px solid ${TOKENS.line}`,
                }}
              >
                <th
                  className="text-left py-2 font-medium"
                  style={{ color: TOKENS.mute }}
                >
                  Metric
                </th>

                <th
                  className="text-right py-2 font-medium"
                  style={{ color: TOKENS.mute }}
                >
                  Base MuRIL
                </th>

                <th
                  className="text-right py-2 font-medium"
                  style={{ color: TOKENS.mute }}
                >
                  MuRIL V2
                </th>

                <th
                  className="text-right py-2 font-semibold"
                  style={{ color: TOKENS.green }}
                >
                  MuRIL V3
                </th>

                <th
                  className="text-right py-2 font-medium"
                  style={{ color: TOKENS.mute }}
                >
                  E5-base
                </th>

                <th
                  className="text-right py-2 font-medium"
                  style={{ color: TOKENS.mute }}
                >
                  BGE-M3
                </th>
              </tr>
            </thead>

            <tbody>
              {METRICS.map((m) => (
                <tr
                  key={m.metric}
                  style={{
                    borderBottom: `1px solid ${TOKENS.line}`,
                  }}
                >
                  <td
                    className="py-2.5"
                    style={{ color: TOKENS.ink }}
                  >
                    {m.metric}
                  </td>

                  <td
                    className="py-2.5 text-right"
                    style={{
                      color: TOKENS.mute,
                      fontFamily:
                        "'JetBrains Mono', monospace",
                    }}
                  >
                    {m.base}
                  </td>

                  <td
                    className="py-2.5 text-right"
                    style={{
                      color: TOKENS.mute,
                      fontFamily:
                        "'JetBrains Mono', monospace",
                    }}
                  >
                    {m.v2}
                  </td>

                  <td
                    className="py-2.5 text-right font-semibold"
                    style={{
                      color: TOKENS.green,
                      fontFamily:
                        "'JetBrains Mono', monospace",
                    }}
                  >
                    {m.v3}
                  </td>

                  <td
                    className="py-2.5 text-right"
                    style={{
                      color: TOKENS.mute,
                      fontFamily:
                        "'JetBrains Mono', monospace",
                    }}
                  >
                    {m.e5}
                  </td>

                  <td
                    className="py-2.5 text-right"
                    style={{
                      color: TOKENS.mute,
                      fontFamily:
                        "'JetBrains Mono', monospace",
                    }}
                  >
                    {m.bge}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <p
          className="text-xs mt-3"
          style={{
            color: TOKENS.mute,
          }}
        >
          Higher is better. All models are evaluated
          on the same held-out agriculture retrieval set.
        </p>
      </div>
    </div>
  );
}

/* ---------------- root ---------------- */

export default function App() {
  const [tab, setTab] =
    useState("assistant");

  return (
    <div
      style={{
        background:
          TOKENS.paper,
        minHeight: "100vh",
        color: TOKENS.ink,
        fontFamily:
          "'Inter', sans-serif",
      }}
    >
      <style>{FONT_IMPORT}</style>

      <NavBar
        tab={tab}
        setTab={setTab}
      />

      {tab === "assistant" && (
        <AssistantPage setTab={setTab} />
      )}

      {tab === "analysis" && (
        <AnalysisPage />
      )}

      {tab === "compare" && (
        <ComparisonPage />
      )}

      <footer className="max-w-6xl mx-auto px-5 sm:px-8 py-10 text-center">
        <p
          className="text-xs"
          style={{
            color: TOKENS.mute,
          }}
        >
          AgriSahayak AI · Live retrieval powered
          by Fine-Tuned MuRIL V3 + FAISS · Answers
          generated using Gemini API or Local Qwen.
        </p>
      </footer>
    </div>
  );
}
