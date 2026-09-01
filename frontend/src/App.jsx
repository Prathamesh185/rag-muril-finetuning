import { useState, useRef } from "react";
import {
  sendChat,
  retrievePassages,
  compareModels,
} from "./api/client";

/* ---------------------------------------------------------
   AgriSahayak AI — Domain-Aware Sentence Encoder for
   Agricultural RAG. Research demo frontend.

   Palette
   --forest   #234D33  primary text / heading ink
   --green    #2F6D4F  primary accent (fine-tuned model, actions)
   --green-lt #E8F4EB  selected state / evidence bg
   --paper    #FFFFFF  base background
   --mist     #F6F8F5  section background
   --line     #E4E8E2  hairline borders
   --ink      #1C231E  body text
   --mute     #667369  secondary text
   --neutral  #F1F1EF  base MuRIL panel bg
   --neutral-ink #57605A base MuRIL text
   --amber    #B8722B  weaker-rank / caution accent
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
  "गेहूं की पहली सिंचाई कब करनी चाहिए?",
  "धान में कीट नियंत्रण कैसे करें?",
  "जीवामृत कैसे तैयार करें?",
  "प्राकृतिक खेती के क्या लाभ हैं?",
];

const PIPELINE_STEPS = [
  "User Query",
  "Fine-Tuned MuRIL",
  "Query Embedding",
  "FAISS",
  "Top-K Passages",
  "LLM",
  "Grounded Answer",
];

const RETRIEVAL_STEPS = [
  "User Query",
  "Fine-Tuned MuRIL",
  "Query Embedding",
  "FAISS",
  "Top-K Passages",
];

const METRICS = [
  { metric: "Accuracy@1", base: "21.46%", tuned: "70.10%" },
  { metric: "Recall@5", base: "39.39%", tuned: "93.18%" },
  { metric: "Recall@10", base: "48.84%", tuned: "96.62%" },
  { metric: "MRR@10", base: "0.2919", tuned: "0.7999" },
  { metric: "NDCG@10", base: "0.3383", tuned: "0.8410" },
];

const DEMO_QUERIES = [
  {
    id: "irrigation",
    query: "गेहूं की पहली सिंचाई कब करनी चाहिए?",
    groundTruthTitle: "गेहूं की खेती — सिंचाई प्रबंधन",
    baseRank: 4,
    tunedRank: 1,
    base: [
      {
        rank: 1,
        title: "गेहूं की फसल की सामान्य जानकारी",
        similarity: 0.74,
        passage: "गेहूं भारत की प्रमुख रबी फसल है जो उत्तर भारत के मैदानी क्षेत्रों में व्यापक रूप से उगाई जाती है। इसकी खेती अक्टूबर–नवंबर में शुरू होती है।",
        highlight: "गेहूं भारत की प्रमुख रबी फसल है जो उत्तर भारत के मैदानी क्षेत्रों में व्यापक रूप से उगाई जाती है।",
        correct: false,
      },
      {
        rank: 2,
        title: "बीज दर और बुवाई विधि",
        similarity: 0.71,
        passage: "गेहूं की बुवाई पंक्तियों में की जाती है, सामान्यतः 20–22 सेमी की दूरी रखी जाती है। बीज दर मिट्टी और किस्म पर निर्भर करती है।",
        highlight: "गेहूं की बुवाई पंक्तियों में की जाती है, सामान्यतः 20–22 सेमी की दूरी रखी जाती है।",
        correct: false,
      },
      {
        rank: 3,
        title: "उर्वरक प्रबंधन — रबी फसलें",
        similarity: 0.69,
        passage: "नाइट्रोजन, फास्फोरस और पोटाश का संतुलित प्रयोग गेहूं की उपज बढ़ाने में सहायक होता है। पहली खुराक बुवाई के समय दी जाती है।",
        highlight: "नाइट्रोजन, फास्फोरस और पोटाश का संतुलित प्रयोग गेहूं की उपज बढ़ाने में सहायक होता है।",
        correct: false,
      },
    ],
    tuned: [
      {
        rank: 1,
        title: "गेहूं की खेती — सिंचाई प्रबंधन",
        similarity: 0.91,
        passage: "गेहूं की फसल के लिए भूमि तैयार करना आवश्यक है। पहली सिंचाई बुवाई के लगभग 20–25 दिन बाद, क्राउन रूट इनिशिएशन की अवस्था में करनी चाहिए।",
        highlight: "पहली सिंचाई बुवाई के लगभग 20–25 दिन बाद, क्राउन रूट इनिशिएशन की अवस्था में करनी चाहिए।",
        correct: true,
      },
      {
        rank: 2,
        title: "रबी फसल — जल प्रबंधन दिशानिर्देश",
        similarity: 0.85,
        passage: "रबी मौसम की फसलों में सिंचाई का समय फसल की अवस्था पर निर्भर करता है। गेहूं में मुकुट जड़ अवस्था सबसे संवेदनशील मानी जाती है।",
        highlight: "गेहूं में मुकुट जड़ अवस्था सबसे संवेदनशील मानी जाती है।",
        correct: false,
      },
      {
        rank: 3,
        title: "मिट्टी की नमी और सिंचाई अनुसूची",
        similarity: 0.79,
        passage: "सही समय पर सिंचाई करने से पानी की बचत होती है और उपज में वृद्धि होती है।",
        highlight: "सही समय पर सिंचाई करने से पानी की बचत होती है और उपज में वृद्धि होती है।",
        correct: false,
      },
    ],
  },
  {
    id: "jeevamrut",
    query: "जीवामृत कैसे तैयार करें?",
    groundTruthTitle: "जीवामृत निर्माण विधि — प्राकृतिक खेती",
    baseRank: 9,
    tunedRank: 1,
    base: [
      {
        rank: 1,
        title: "प्राकृतिक खेती का परिचय",
        similarity: 0.68,
        passage: "प्राकृतिक खेती एक कम लागत वाली कृषि पद्धति है जिसमें रासायनिक उर्वरकों के स्थान पर स्थानीय संसाधनों का उपयोग किया जाता है।",
        highlight: "प्राकृतिक खेती एक कम लागत वाली कृषि पद्धति है जिसमें रासायनिक उर्वरकों के स्थान पर स्थानीय संसाधनों का उपयोग किया जाता है।",
        correct: false,
      },
      {
        rank: 2,
        title: "देशी गाय आधारित खेती",
        similarity: 0.65,
        passage: "देशी गाय का गोबर और गोमूत्र प्राकृतिक खेती में कई तैयारियों का आधार होते हैं, जिनमें मिट्टी की उर्वरता बढ़ाना शामिल है।",
        highlight: "देशी गाय का गोबर और गोमूत्र प्राकृतिक खेती में कई तैयारियों का आधार होते हैं।",
        correct: false,
      },
      {
        rank: 3,
        title: "जैविक खाद के प्रकार",
        similarity: 0.63,
        passage: "कम्पोस्ट, वर्मीकम्पोस्ट और हरी खाद प्राकृतिक खेती में मिट्टी सुधार के लिए प्रयोग की जाने वाली प्रमुख विधियाँ हैं।",
        highlight: "कम्पोस्ट, वर्मीकम्पोस्ट और हरी खाद प्राकृतिक खेती में मिट्टी सुधार के लिए प्रयोग की जाने वाली प्रमुख विधियाँ हैं।",
        correct: false,
      },
    ],
    tuned: [
      {
        rank: 1,
        title: "जीवामृत निर्माण विधि — प्राकृतिक खेती",
        similarity: 0.89,
        passage: "जीवामृत बनाने के लिए देशी गाय का गोबर, गोमूत्र, गुड़, बेसन और खेत की मेड़ की मिट्टी को पानी में मिलाकर 5–7 दिन तक किण्वित किया जाता है।",
        highlight: "जीवामृत बनाने के लिए देशी गाय का गोबर, गोमूत्र, गुड़, बेसन और खेत की मेड़ की मिट्टी को पानी में मिलाकर 5–7 दिन तक किण्वित किया जाता है।",
        correct: true,
      },
      {
        rank: 2,
        title: "देशी गाय आधारित खेती",
        similarity: 0.81,
        passage: "देशी गाय का गोबर और गोमूत्र प्राकृतिक खेती में कई तैयारियों का आधार होते हैं, जिनमें जीवामृत और बीजामृत शामिल हैं।",
        highlight: "देशी गाय का गोबर और गोमूत्र प्राकृतिक खेती में कई तैयारियों का आधार होते हैं, जिनमें जीवामृत और बीजामृत शामिल हैं।",
        correct: false,
      },
      {
        rank: 3,
        title: "किण्वित जैविक घोल — प्रयोग विधि",
        similarity: 0.77,
        passage: "तैयार घोल को छानकर सिंचाई के पानी के साथ या पत्तियों पर छिड़काव के रूप में प्रयोग किया जा सकता है।",
        highlight: "तैयार घोल को छानकर सिंचाई के पानी के साथ या पत्तियों पर छिड़काव के रूप में प्रयोग किया जा सकता है।",
        correct: false,
      },
    ],
  },
  {
    id: "pest",
    query: "धान में कीट नियंत्रण कैसे करें?",
    groundTruthTitle: "धान — प्रमुख कीट एवं समेकित प्रबंधन",
    baseRank: 6,
    tunedRank: 2,
    base: [
      {
        rank: 1,
        title: "धान की खेती — सामान्य परिचय",
        similarity: 0.72,
        passage: "धान भारत की प्रमुख खरीफ फसल है, जिसकी खेती जलभराव वाले खेतों में की जाती है। रोपाई जून–जुलाई में होती है।",
        highlight: "धान भारत की प्रमुख खरीफ फसल है, जिसकी खेती जलभराव वाले खेतों में की जाती है।",
        correct: false,
      },
      {
        rank: 2,
        title: "धान की उन्नत किस्में",
        similarity: 0.70,
        passage: "क्षेत्र और मिट्टी के अनुसार उपयुक्त उन्नत किस्मों का चयन उपज बढ़ाने में महत्वपूर्ण भूमिका निभाता है।",
        highlight: "क्षेत्र और मिट्टी के अनुसार उपयुक्त उन्नत किस्मों का चयन उपज बढ़ाने में महत्वपूर्ण भूमिका निभाता है।",
        correct: false,
      },
      {
        rank: 3,
        title: "खरीफ फसलों में जल प्रबंधन",
        similarity: 0.68,
        passage: "धान के खेत में लगातार जलभराव बनाए रखना आवश्यक होता है, विशेषकर रोपाई के शुरुआती चरण में।",
        highlight: "धान के खेत में लगातार जलभराव बनाए रखना आवश्यक होता है, विशेषकर रोपाई के शुरुआती चरण में।",
        correct: false,
      },
    ],
    tuned: [
      {
        rank: 1,
        title: "धान का तना छेदक — पहचान एवं नियंत्रण",
        similarity: 0.86,
        passage: "तना छेदक कीट धान की फसल में 'डेड हार्ट' और 'व्हाइट ईयर' लक्षण उत्पन्न करता है। समेकित नियंत्रण में प्रकाश प्रपंच और जैविक शत्रु कीटों का उपयोग किया जाता है।",
        highlight: "समेकित नियंत्रण में प्रकाश प्रपंच और जैविक शत्रु कीटों का उपयोग किया जाता है।",
        correct: false,
      },
      {
        rank: 2,
        title: "धान — प्रमुख कीट एवं समेकित प्रबंधन",
        similarity: 0.84,
        passage: "धान में तना छेदक, पत्ती लपेटक और भूरा फुदका प्रमुख कीट हैं। समेकित कीट प्रबंधन में फसल चक्र, प्रतिरोधी किस्में और आवश्यकतानुसार कीटनाशी प्रयोग शामिल है।",
        highlight: "समेकित कीट प्रबंधन में फसल चक्र, प्रतिरोधी किस्में और आवश्यकतानुसार कीटनाशी प्रयोग शामिल है।",
        correct: true,
      },
      {
        rank: 3,
        title: "भूरा फुदका — प्रबंधन रणनीति",
        similarity: 0.80,
        passage: "भूरा फुदका नमी और घनी रोपाई में तेजी से फैलता है। खेत की समय-समय पर निगरानी अनुशंसित है।",
        highlight: "खेत की समय-समय पर निगरानी अनुशंसित है।",
        correct: false,
      },
    ],
  },
];

/* ---------------- shared UI atoms ---------------- */

function Badge({ children }) {
  return (
    <span
      className="inline-flex items-center rounded-full px-3 py-1 text-xs font-medium"
      style={{ background: TOKENS.greenLt, color: TOKENS.forest, border: `1px solid ${TOKENS.greenLine}` }}
    >
      {children}
    </span>
  );
}

function SimBadge({ value }) {
  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-xs font-medium"
      style={{ background: TOKENS.mist, color: TOKENS.ink, border: `1px solid ${TOKENS.line}`, fontFamily: "'JetBrains Mono', monospace" }}
    >
      cos&nbsp;sim&nbsp;{value.toFixed(2)}
    </span>
  );
}

function EvidenceCard({ item, accent }) {
  const isGreen = accent === "green";
  const before = item.passage.slice(0, item.passage.indexOf(item.highlight));
  const after = item.passage.slice(item.passage.indexOf(item.highlight) + item.highlight.length);
  return (
    <div
      className="rounded-xl p-4 sm:p-5"
      style={{
        background: item.correct ? TOKENS.greenLt : TOKENS.paper,
        border: `1px solid ${item.correct ? TOKENS.greenLine : TOKENS.line}`,
      }}
    >
      <div className="flex items-center justify-between mb-2.5 flex-wrap gap-2">
        <div className="flex items-center gap-2">
          <span
            className="flex items-center justify-center w-6 h-6 rounded-full text-xs font-semibold"
            style={{
              background: isGreen ? TOKENS.green : TOKENS.neutralInk,
              color: "#fff",
            }}
          >
            {item.rank}
          </span>
          <span className="text-sm font-medium" style={{ color: TOKENS.ink }}>
            {item.title}
          </span>
          {item.correct && (
            <span className="text-xs font-semibold" style={{ color: TOKENS.green }}>
              ✓ ground truth
            </span>
          )}
        </div>
        <SimBadge value={item.similarity} />
      </div>
      <p dir="auto" className="text-[15px] leading-relaxed" style={{ color: TOKENS.mute, fontFamily: "'Noto Sans Devanagari', 'Inter', sans-serif" }}>
        {before}
        <mark
          style={{
            background: item.correct ? "#CFEAD8" : TOKENS.greenLt,
            color: TOKENS.forest,
            padding: "0 2px",
            borderRadius: 3,
          }}
        >
          {item.highlight}
        </mark>
        {after}
      </p>
      <div className="mt-3 flex items-center justify-between">
        <span className="text-xs" style={{ color: TOKENS.mute }}>
          Source: Vikaspedia
        </span>
        <a href="https://vikaspedia.in" target="_blank" rel="noreferrer" className="text-xs font-medium hover:underline" style={{ color: TOKENS.green }}>
          View source →
        </a>
      </div>
    </div>
  );
}

function LiveEvidenceCard({ item }) {
  return (
    <div
      className="rounded-xl p-4 sm:p-5"
      style={{
        background: TOKENS.paper,
        border: `1px solid ${TOKENS.line}`,
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
        </div>

        <span
          className="inline-flex items-center rounded-md px-2 py-1 text-xs font-medium"
          style={{
            background: TOKENS.mist,
            color: TOKENS.ink,
            border: `1px solid ${TOKENS.line}`,
            fontFamily: "'JetBrains Mono', monospace",
          }}
        >
          score&nbsp;{Number(item.score).toFixed(3)}
        </span>
      </div>

      <p
        dir="auto"
        className="text-[15px] leading-relaxed"
        style={{
          color: TOKENS.mute,
          fontFamily: "'Noto Sans Devanagari','Inter',sans-serif",
        }}
      >
        {item.text}
      </p>

      <div className="mt-3 flex items-center justify-between gap-3">
        <span className="text-xs" style={{ color: TOKENS.mute }}>
          Source: {item.source || "Unknown"}
        </span>

        {item.url && (
          <a
            href={item.url}
            target="_blank"
            rel="noreferrer"
            className="text-xs font-medium hover:underline"
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
  steps = PIPELINE_STEPS,
}) {
  return (
    <div className="flex items-center overflow-x-auto gap-1.5 pb-1">
      {steps.map((step, i) => {
        const isHero = step === "Fine-Tuned MuRIL";
        return (
          <div key={step} className="flex items-center flex-shrink-0">
            <div
              className={`rounded-lg text-center whitespace-nowrap ${compact ? "px-3 py-2 text-xs" : "px-4 py-3 text-sm"}`}
              style={{
                background: isHero ? TOKENS.green : TOKENS.mist,
                color: isHero ? "#fff" : TOKENS.ink,
                border: `1px solid ${isHero ? TOKENS.green : TOKENS.line}`,
                fontWeight: isHero ? 600 : 500,
              }}
            >
              {step}
            </div>
            {i < steps.length - 1 && (
              <span className="mx-1.5 flex-shrink-0" style={{ color: TOKENS.mute }}>
                →
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
}

function RankBar({ label, rank, maxRank, isWinner, accent }) {
  const pct = Math.max(6, 100 - ((rank - 1) / maxRank) * 100);
  return (
    <div>
      <div className="flex items-baseline justify-between mb-1.5">
        <span className="text-sm font-medium" style={{ color: TOKENS.ink }}>
          {label}
        </span>
        <span className="text-sm font-semibold" style={{ color: isWinner ? TOKENS.green : TOKENS.amber, fontFamily: "'JetBrains Mono', monospace" }}>
          #{rank} {isWinner && "✓"}
        </span>
      </div>
      <div className="h-2.5 rounded-full overflow-hidden" style={{ background: TOKENS.mist }}>
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{ width: `${pct}%`, background: isWinner ? TOKENS.green : "#D7BB98", }}
        />
      </div>
    </div>
  );
}

/* ---------------- top nav ---------------- */

function NavBar({ tab, setTab }) {
  const tabs = [
    { id: "assistant", label: "AI Assistant" },
    { id: "analysis", label: "Retrieval Analysis" },
    { id: "compare", label: "Model Comparison" },
  ];
  return (
    <header
      className="sticky top-0 z-20 backdrop-blur"
      style={{ background: "rgba(255,255,255,0.92)", borderBottom: `1px solid ${TOKENS.line}` }}
    >
      <div className="max-w-6xl mx-auto px-5 sm:px-8 h-16 flex items-center justify-between">
        <div className="flex items-baseline gap-2.5">
          <span
            className="text-[19px] font-semibold tracking-tight"
            style={{ color: TOKENS.forest, fontFamily: "'Newsreader', serif" }}
          >
            AgriSahayak AI
          </span>
          <span className="hidden md:inline text-xs" style={{ color: TOKENS.mute }}>
            Agriculture-Aware Multilingual RAG
          </span>
        </div>
        <nav className="flex items-center gap-1 rounded-full p-1" style={{ background: TOKENS.mist }}>
          {tabs.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className="px-3.5 py-1.5 rounded-full text-sm font-medium transition-colors"
              style={{
                background: tab === t.id ? TOKENS.paper : "transparent",
                color: tab === t.id ? TOKENS.forest : TOKENS.mute,
                boxShadow: tab === t.id ? "0 1px 2px rgba(35,77,51,0.12)" : "none",
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

function AssistantPage() {
  const [query, setQuery] = useState("");
  const [submitted, setSubmitted] = useState(null);

  const [modelChoice, setModelChoice] =
    useState("Gemini API");

  const [answer, setAnswer] = useState("");
  const [retrieved, setRetrieved] = useState([]);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [evOpen, setEvOpen] = useState(false);
  

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
            typeof item.url === "string" &&
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

  if (!submitted) {
    return (
      <div className="relative">
        <div
          className="absolute inset-0 -z-10"
          style={{
            backgroundImage: `radial-gradient(${TOKENS.greenLine} 1px, transparent 1px)`,
            backgroundSize: "22px 22px",
            maskImage:
              "radial-gradient(ellipse 60% 50% at 50% 30%, black 40%, transparent 100%)",
            opacity: 0.5,
          }}
        />

        <div className="max-w-2xl mx-auto px-5 pt-20 pb-24 text-center">
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
            className="text-[15px] mb-8"
            style={{ color: TOKENS.mute }}
          >
            Ask agriculture questions and get answers grounded in retrieved agricultural knowledge.
          </p>

          <div className="flex flex-wrap justify-center gap-2 mb-8">
            {EXAMPLES.map((ex) => (
              <button
                key={ex}
                dir="auto"
                onClick={() => ask(ex)}
                className="px-3.5 py-2 rounded-full text-sm transition-colors"
                style={{
                  background: TOKENS.paper,
                  border: `1px solid ${TOKENS.line}`,
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
              style={{ color: TOKENS.mute }}
            >
              Answer model
            </span>

            <select
              value={modelChoice}
              onChange={(e) =>
                setModelChoice(e.target.value)
              }
              className="text-sm rounded-lg px-3 py-2 outline-none"
              style={{
                background: TOKENS.paper,
                color: TOKENS.ink,
                border: `1px solid ${TOKENS.line}`,
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
              background: TOKENS.paper,
              border: `1px solid ${TOKENS.line}`,
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
              type="button"
              title="Voice input — coming soon"
              className="w-9 h-9 flex-shrink-0 flex items-center justify-center rounded-xl"
              style={{
                color: TOKENS.mute,
                border: `1px solid ${TOKENS.line}`,
              }}
            >
              🎙
            </button>

            <button
              type="submit"
              disabled={loading}
              className="px-4 py-2.5 rounded-xl text-sm font-medium flex-shrink-0"
              style={{
                background: TOKENS.green,
                color: "#fff",
                opacity: loading ? 0.7 : 1,
              }}
            >
              Ask
            </button>
          </form>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-2xl mx-auto px-5 py-10">
      <button
        onClick={resetQuestion}
        className="text-xs font-medium mb-6"
        style={{ color: TOKENS.mute }}
      >
        ← New question
      </button>

      <div className="mb-6">
        <div
          className="text-xs font-medium mb-1.5"
          style={{ color: TOKENS.mute }}
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
          style={{ color: TOKENS.mute }}
        >
          Model: {modelChoice}
        </div>
      </div>

      {loading && (
        <div
          className="rounded-xl p-6 flex items-center gap-3"
          style={{
            background: TOKENS.mist,
            border: `1px solid ${TOKENS.line}`,
          }}
        >
          <span
            className="w-4 h-4 rounded-full animate-spin flex-shrink-0"
            style={{
              border: `2px solid ${TOKENS.greenLine}`,
              borderTopColor: TOKENS.green,
            }}
          />

          <span
            className="text-sm"
            style={{ color: TOKENS.mute }}
          >
            Retrieving agricultural knowledge and generating answer…
          </span>
        </div>
      )}

      {!loading && error && (
        <div
          className="rounded-xl p-5"
          style={{
            background: TOKENS.amberLt,
            border: `1px solid ${TOKENS.amber}`,
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

      {!loading && !error && answer && (
        <>
          <div
            className="rounded-2xl p-5 sm:p-6 mb-4"
            style={{
              background: TOKENS.paper,
              border: `1px solid ${TOKENS.line}`,
              boxShadow:
                "0 1px 3px rgba(28,35,30,0.05)",
            }}
          >
            <div className="mb-3">
              <Badge>
                Generated from retrieved agricultural evidence
              </Badge>
            </div>

            <p
              dir="auto"
              className="text-[15px] leading-relaxed mb-4"
              style={{
                color: TOKENS.ink,
                fontFamily:
                  "'Noto Sans Devanagari','Inter',sans-serif",
              }}
            >
              {answer}
            </p>

            {uniqueSources.length > 0 && (
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
                    color: TOKENS.mute,
                  }}
                >
                  Sources
                </span>

                {uniqueSources.map(
                  (source, index) => (
                    <a
                      key={source.url}
                      href={source.url}
                      target="_blank"
                      rel="noreferrer"
                      dir="auto"
                      className="text-xs font-medium px-2.5 py-1 rounded-full hover:underline"
                      style={{
                        background:
                          TOKENS.greenLt,
                        color: TOKENS.forest,
                      }}
                    >
                      [{index + 1}]{" "}
                      {source.title}
                    </a>
                  )
                )}
              </div>
            )}
          </div>

          {retrieved.length > 0 && (
            <>
              <button
                onClick={() =>
                  setEvOpen((v) => !v)
                }
                className="w-full flex items-center justify-between rounded-xl px-5 py-3.5 text-sm font-medium mb-3"
                style={{
                  background: TOKENS.mist,
                  color: TOKENS.forest,
                  border:
                    `1px solid ${TOKENS.line}`,
                }}
              >
                <span>
                  {evOpen ? "▾" : "▸"} View
                  retrieved evidence (
                  {retrieved.length})
                </span>

                <span
                  className="text-xs font-normal"
                  style={{
                    color: TOKENS.mute,
                  }}
                >
                  via Fine-Tuned MuRIL + FAISS
                </span>
              </button>

              {evOpen && (
                <div className="space-y-3">
                  {retrieved.map((item) => (
                    <LiveEvidenceCard
                      key={
                        item.chunk_id ??
                        `${item.rank}-${item.title}`
                      }
                      item={item}
                    />
                  ))}
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

function AnalysisPage() {
  const [query, setQuery] = useState(
    DEMO_QUERIES[0].query
  );

  const [activeQuery, setActiveQuery] =
    useState("");

  const [retrieved, setRetrieved] =
    useState([]);

  const [loading, setLoading] =
    useState(false);

  const [error, setError] =
    useState("");

  const runRetrieval = async (question) => {
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
      const result = await retrievePassages(
        text,
        5
      );

      setRetrieved(
        Array.isArray(result.retrieved)
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
            fontFamily: "'Newsreader', serif",
          }}
        >
          Retrieval Analysis
        </h2>

        <p
          className="text-sm"
          style={{ color: TOKENS.mute }}
        >
          Inspect the real passages retrieved by
          Fine-Tuned MuRIL and FAISS.
        </p>
      </div>

      <div
        className="rounded-2xl p-5 mb-5"
        style={{
          background: TOKENS.paper,
          border: `1px solid ${TOKENS.line}`,
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
              setQuery(e.target.value)
            }
            placeholder="Enter an agriculture question"
            className="flex-1 rounded-xl px-4 py-2.5 text-[15px] outline-none"
            style={{
              border: `1px solid ${TOKENS.line}`,
              color: TOKENS.ink,
              fontFamily:
                "'Noto Sans Devanagari','Inter',sans-serif",
            }}
          />

          <button
            type="submit"
            disabled={loading}
            className="px-5 py-2.5 rounded-xl text-sm font-medium"
            style={{
              background: TOKENS.green,
              color: "#fff",
              opacity: loading ? 0.7 : 1,
            }}
          >
            Retrieve
          </button>
        </form>

        <div className="flex flex-wrap items-center gap-2 mt-3">
          <span
            className="text-xs"
            style={{ color: TOKENS.mute }}
          >
            Try an example:
          </span>

          {DEMO_QUERIES.map((d) => (
            <button
              key={d.id}
              dir="auto"
              onClick={() =>
                runRetrieval(d.query)
              }
              className="text-xs px-2.5 py-1 rounded-full"
              style={{
                background: TOKENS.mist,
                color: TOKENS.forest,
                border: `1px solid ${TOKENS.line}`,
                fontFamily:
                  "'Noto Sans Devanagari','Inter',sans-serif",
              }}
            >
              {d.query}
            </button>
          ))}
        </div>
      </div>

      <div
        className="rounded-2xl p-5 sm:p-6 mb-8"
        style={{
          background: TOKENS.mist,
          border: `1px solid ${TOKENS.line}`,
        }}
      >
        <div
          className="text-xs font-medium mb-4"
          style={{ color: TOKENS.mute }}
        >
          Query → retrieval pipeline
        </div>

        <PipelineDiagram steps={RETRIEVAL_STEPS} />

        <p
          className="text-xs mt-4"
          style={{ color: TOKENS.mute }}
        >
          <span
            style={{
              color: TOKENS.green,
              fontWeight: 600,
            }}
          >
            Fine-Tuned MuRIL
          </span>{" "}
          produces the normalized query embedding
          used to search the FAISS index.
        </p>
      </div>

      {loading && (
        <div
          className="rounded-xl p-6 flex items-center gap-3"
          style={{
            background: TOKENS.mist,
            border: `1px solid ${TOKENS.line}`,
          }}
        >
          <span
            className="w-4 h-4 rounded-full animate-spin flex-shrink-0"
            style={{
              border:
                `2px solid ${TOKENS.greenLine}`,
              borderTopColor: TOKENS.green,
            }}
          />

          <span
            className="text-sm"
            style={{ color: TOKENS.mute }}
          >
            Searching the Fine-Tuned MuRIL FAISS
            index…
          </span>
        </div>
      )}

      {!loading && error && (
        <div
          className="rounded-xl p-5"
          style={{
            background: TOKENS.amberLt,
            border: `1px solid ${TOKENS.amber}`,
          }}
        >
          <div
            className="text-sm font-medium"
            style={{ color: TOKENS.ink }}
          >
            Retrieval failed
          </div>

          <div
            className="text-sm mt-1"
            style={{ color: TOKENS.mute }}
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
                style={{ color: TOKENS.ink }}
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
              {retrieved.map((item) => (
                <LiveEvidenceCard
                  key={
                    item.chunk_id ??
                    `${item.rank}-${item.title}`
                  }
                  item={item}
                />
              ))}
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
              background: TOKENS.mist,
              border:
                `1px dashed ${TOKENS.line}`,
            }}
          >
            <p
              className="text-sm"
              style={{ color: TOKENS.mute }}
            >
              Enter a question to inspect the real
              FAISS retrieval results.
            </p>
          </div>
        )}
    </div>
  );
}

/* ---------------- Page 3: Model Comparison ---------------- */

function ComparisonPage() {
  const [customQuery, setCustomQuery] =
    useState("");

  const [active, setActive] =
    useState(null);

  const [loading, setLoading] =
    useState(false);

  const [error, setError] =
    useState("");


  const run = async (question) => {
    const text = question?.trim();

    if (!text || loading) {
      return;
    }

    setCustomQuery(text);
    setActive(null);
    setError("");
    setLoading(true);

    try {
      const result = await compareModels(
        text,
        5
      );

      setActive({
        query: text,
        base: Array.isArray(result.base)
          ? result.base
          : [],
        finetuned: Array.isArray(
          result.finetuned
        )
          ? result.finetuned
          : [],
      });
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
          Base MuRIL vs Fine-Tuned MuRIL
        </h2>

        <p
          className="text-sm"
          style={{ color: TOKENS.mute }}
        >
          Compare live retrieval results for
          the same agriculture query.
        </p>
      </div>


      <div
        className="rounded-2xl p-5 mb-4"
        style={{
          background: TOKENS.paper,
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
            placeholder="गेहूं की पहली सिंचाई कब करनी चाहिए?"
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
            className="px-5 py-2.5 rounded-xl text-sm font-medium disabled:opacity-50"
            style={{
              background: TOKENS.green,
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

          {DEMO_QUERIES.map((d) => (
            <button
              key={d.id}
              type="button"
              dir="auto"
              disabled={loading}
              onClick={() =>
                run(d.query)
              }
              className="text-xs px-2.5 py-1 rounded-full disabled:opacity-50"
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
          ))}
        </div>
      </div>


      {loading && (
        <div
          className="rounded-xl p-6 flex items-center gap-3 mb-6"
          style={{
            background: TOKENS.mist,
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
            className="rounded-xl p-4 mb-6"
            style={{
              background: TOKENS.mist,
              border:
                `1px solid ${TOKENS.line}`,
            }}
          >
            <div
              className="text-xs mb-1"
              style={{
                color: TOKENS.mute,
              }}
            >
              Query
            </div>

            <div
              dir="auto"
              className="text-sm font-medium"
              style={{
                color: TOKENS.ink,
                fontFamily:
                  "'Noto Sans Devanagari','Inter',sans-serif",
              }}
            >
              {active.query}
            </div>
          </div>


          <p
            className="text-xs mb-6"
            style={{
              color: TOKENS.mute,
            }}
          >
            Similarity scores belong to each
            model's own embedding space.
            Compare passage relevance and
            ranking rather than comparing
            Base and Fine-Tuned scores
            directly.
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
                  Fine-Tuned MuRIL
                </span>

                <span
                  className="text-xs"
                  style={{
                    color: TOKENS.mute,
                  }}
                >
                  agriculture-aware
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
              background: TOKENS.mist,
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
              Enter a query or pick an
              example to retrieve passages
              from both models.
            </p>
          </div>
        )}


      {/* Why fine-tune */}
      <div
        className="mt-12 rounded-2xl p-6 sm:p-8"
        style={{
          background: TOKENS.paper,
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
          closer to their relevant
          agricultural passages in the
          embedding space.
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
          background: TOKENS.paper,
          border:
            `1px solid ${TOKENS.line}`,
        }}
      >
        <h3
          className="text-base font-semibold mb-4"
          style={{
            color: TOKENS.forest,
          }}
        >
          V2 in-domain retrieval evaluation
        </h3>

        <div className="overflow-x-auto">
          <table className="w-full text-sm border-collapse">
            <thead>
              <tr
                style={{
                  borderBottom:
                    `1px solid ${TOKENS.line}`,
                }}
              >
                <th
                  className="text-left py-2 font-medium"
                  style={{
                    color: TOKENS.mute,
                  }}
                >
                  Metric
                </th>

                <th
                  className="text-right py-2 font-medium"
                  style={{
                    color: TOKENS.mute,
                  }}
                >
                  Base MuRIL
                </th>

                <th
                  className="text-right py-2 font-medium"
                  style={{
                    color:
                      TOKENS.forest,
                  }}
                >
                  Fine-Tuned MuRIL
                </th>
              </tr>
            </thead>

            <tbody>
              {METRICS.map((m) => (
                <tr
                  key={m.metric}
                  style={{
                    borderBottom:
                      `1px solid ${TOKENS.line}`,
                  }}
                >
                  <td
                    className="py-2.5"
                    style={{
                      color:
                        TOKENS.ink,
                    }}
                  >
                    {m.metric}
                  </td>

                  <td
                    className="py-2.5 text-right"
                    style={{
                      color:
                        TOKENS.mute,
                      fontFamily:
                        "'JetBrains Mono', monospace",
                    }}
                  >
                    {m.base}
                  </td>

                  <td
                    className="py-2.5 text-right font-medium"
                    style={{
                      color:
                        TOKENS.green,
                      fontFamily:
                        "'JetBrains Mono', monospace",
                    }}
                  >
                    {m.tuned}
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
          Higher is better. Results are from the V2 held-out agriculture retrieval evaluation.
        </p>
      </div>
    </div>
  );
}

/* ---------------- root ---------------- */

export default function App() {
  const [tab, setTab] = useState("assistant");
  return (
    <div style={{ background: TOKENS.paper, minHeight: "100vh", color: TOKENS.ink, fontFamily: "'Inter', sans-serif" }}>
      <style>{FONT_IMPORT}</style>
      <NavBar tab={tab} setTab={setTab} />
      {tab === "assistant" && <AssistantPage />}
      {tab === "analysis" && <AnalysisPage />}
      {tab === "compare" && <ComparisonPage />}
      <footer className="max-w-6xl mx-auto px-5 sm:px-8 py-10 text-center">
        <p className="text-xs" style={{ color: TOKENS.mute }}>
          AgriSahayak AI · Live retrieval powered by Fine-Tuned MuRIL + FAISS · Answers generated using Gemini API or Local Qwen.
        </p>
      </footer>
    </div>
  );
}
