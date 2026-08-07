# Reflection Report

**AI Mentorship Program**

**Project: AI Health Orchestration System**

---

## 1. Why I Liked This Project

The program is structured around six modules, local LLM deployment, fine-tuning,
vector databases, RAG, agentic AI, and MCP. I picked a health orchestration system
specifically because it's a domain where none of these techniques are optional
extras: a meal plan or gym program that isn't grounded in real guidelines is just the
model making things up convincingly, and a health-adjacent use case makes that failure
mode obvious rather than easy to ignore. Choosing a single, real, end-to-end system as
the vehicle for all six modules. Every module's
exercises had to actually serve the same running application, which meant I couldn't
learn a technique shallowly and move on. It had to work with what came before it.

---

## 2. What Each Module Actually Taught Me

**Module 1 — Local LLM Deployment.** This is where I learned that "run a model
locally" is a systems problem, not a machine learning one. Getting vLLM to serve Llama
3.1 8B on a T4 without OOMing on CUDA graph warmup taught me more about GPU memory
budgeting than any amount of reading about parameter counts would have. The concrete
lesson, 1B parameters ≈ 2GB VRAM in fp16, and quantization is the lever you pull when
that math doesn't fit your hardware is something I now reach for automatically.

**Module 2 — LoRA / QLoRA.** This module reframed fine-tuning for me. I'd understood
LoRA conceptually before freeze the base, train small low-rank matrices but
actually running QLoRA against a quantized base model and hitting the AWQ limitation
(it's inference-only; you can't backprop through it) was the first time I understood
*why* the quantization method you choose for serving constrains your fine-tuning
options later. That's a decision-ordering lesson, not just a technique.

**Module 3 — Vector Databases.** I went in thinking of ChromaDB as "a place to put
embeddings" and came out understanding the indexing/accuracy/latency trade-offs that
make ANN search work at all, HNSW graphs, why collection-per-domain beats one
collection with metadata filters when you want retrieval to actually be correct in
the case that matters (never accidentally retrieving a recipe when the agent asked for
lab reference ranges).

**Module 4 — RAG.** The most useful moment in this module was seeing my own system
fail to use retrieved context, and having to figure out why, not a hypothetical
"here's why RAG helps" but a real before/after comparison on my own pipeline where the
grounded answer returned a 4-tier LDL classification and the base model returned a
flattened, less accurate one. That's the kind of evidence that sticks; abstractly
knowing RAG reduces hallucination is different from watching it happen on a question
I wrote myself.

**Module 5 — Agentic AI / Orchestration.** This is where the project stopped being
"call an LLM a few times" and became genuine system design. Deciding the four agents
had to run sequentially (because Lab Analysis's output is a hard input to Nutrition
and Training) rather than in parallel was a real architectural decision with a real
trade-off, not a default. Discovering that a 7-day meal plan reliably blew past a
2048-token output budget, and that splitting generation into two calls was a better fix
than just raising the limit, taught me to treat token budgets as a first-class design
constraint rather than an afterthought.

**Module 6 — MCP.** This module changed how I think about "should this be a network
boundary or a function call." My first instinct was to convert everything to MCP
because the module was about MCP, but working through *why* the RAG corpus made
sense as an MCP server (a genuinely reusable knowledge service) while agent-to-agent
calls didn't (they already get polymorphism for free from `BaseAgent`, with no network
hop) was a more useful lesson than the protocol mechanics themselves. Debugging the
`structuredContent` vs `content[0].text` confusion, and watching a single misplaced
method definition take down the entire API container, taught me a lot about how much
implicit trust a codebase places in every class actually implementing what its abstract
base promises.

---

## 3. Where the Program Pushed Me Outside "Just Following the Curriculum"

The module write-ups gave me the concepts; the project forced me to make decisions the
curriculum didn't hand me an answer for. Nobody told me to split nutrition generation
into two LLM calls, or to route retrieval through collection-specific ChromaDB
collections instead of one big index, or to convert three agents to MCP instead of
stopping at one to "check the box." Those were judgment calls that came directly out of
watching my own system fail and having to decide what "good enough to move on" meant at
each stage. That's the part of the program I valued most, not the modules
individually, but the requirement to integrate them into something that had to keep
working as each new module added new failure modes.

I also learned to value my own debugging process more. Several of the "key learnings"
documented in the README, the silent MCP fallback that kept reporting success while
quietly returning ungrounded answers, the `IndentationError` in `BaseAgent` that broke
every agent at once, the Python logging config that was silently swallowing every
`INFO` log the whole time, were not things I'd have predicted going in. They were
the kind of lessons that only show up when you build something real enough to have
real bugs, and the program's project-based structure is what put me in a position to
hit them.

---

## 4. Ethical Considerations the Domain Forced Me to Think About

Choosing a health application as the learning vehicle meant I couldn't treat "the model
said something plausible" as good enough, in a way a more neutral domain might have let
me get away with. Working through the Lab Analysis Agent made me think concretely
about the difference between *retrieval-grounded* and *clinically correct*, RAG
lowers hallucination risk, but a chunk pulled from a reference-range corpus is not the
same thing as a clinician's judgment, and building this system made that distinction
tangible rather than abstract. It also made me more attentive to how confidently
structured output, clean JSON, labeled fields, cited-sounding context can read as
more authoritative than the underlying reliability actually supports, independent of
whether the content is right. That's a lesson about AI system design generally, not
just this project: the parts of a system that make it feel trustworthy (schemas,
structure, retrieval citations) and the parts that make it *actually* trustworthy
(evaluation, review, correctness) are not the same work, and it's easy to invest in
the first while believing you're doing the second.

---

## 5. Where I'd Take This Next

If I continued past the program, the module I'd most want to revisit is Module 2,
the fine-tuning dataset was small (21 examples) and built to demonstrate the QLoRA
mechanics rather than to meaningfully improve output quality, and I'd like to go back
with a larger, more deliberately curated dataset now that I understand the rest of the
pipeline it feeds into. I'd also want to build a real evaluation harness for the
agents' outputs, something the six-module structure didn't require but that my own
experience building this system tells me is the natural next skill to learn.

Overall, this program gave me a genuine end-to-end fluency I didn't have going in,
not just "I've heard of LoRA and RAG and MCP," but "I've made real architectural
decisions using all of them together, watched those decisions succeed or fail, and can
explain why." That's the outcome I came in hoping for.