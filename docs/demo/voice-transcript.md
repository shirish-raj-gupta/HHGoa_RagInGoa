# Voice round trip — captured output

Real run against the **production index** (`eng_Latn`, 950,712 passages,
memory-mapped, self-retrieval 0.987), Sarvam realtime STT, Groq `gpt-oss-120b`.
Audio is Sarvam TTS fed to the WebSocket at real-time pace (40 ms frames), so
partials arrive as they would from a live microphone.

Reproduce: `python -m uvicorn src.api.app:app` then drive `/ws` with 16 kHz PCM16.

---

## 1. Grounded answer — every rail passes

```
SPOKEN: "How many calories are in a banana"   (1.53s audio)

   639ms  partial      'How '
  1179ms  partial      'How many'
  1807ms  partial      'How many calories are'
  1807ms  SPECULATIVE  fired on 'How many calories are'
  2147ms  FINAL        'How many calories are in a banana?'
  2199ms  speculation  refreshed=True  divergence=0.382
  3712ms  DONE         core=98.4ms  stt=2143ms  ttft=1385ms

ANSWER   "A typical banana contains roughly 90 calories."
CITE     [eng_Latn:b9cfa49e49d] "one banana only has approximately 90 calories"
CITE     [eng_Latn:5fb48003492] "There are 89 calories in 100g of fresh, raw bananas"

rails    injection ok · unsafe ok · language ok · relevance ok
         citation_spans ok · grounding ok · answer_scope ok
         format ok · language_match ok
```

Both citations resolve to real character spans in genuinely retrieved
passages — that is what `citation_spans` verifies, and a fabricated
`passage_id` is a hard fail.

---

## 2. Refusal — and speculation actually paying off

```
SPOKEN: "What is a corporation"   (1.28s audio)

   674ms  partial      'H'
  1202ms  partial      'Hat is'
  1618ms  partial      'Hat is the corporation'
  1618ms  SPECULATIVE  fired on 'Hat is the corporation'
  1735ms  FINAL        'Hat is a corporation'
  1795ms  speculation  refreshed=False  divergence=0.15
  3483ms  DONE         core=108.1ms  stt=1710ms  ttft=1574ms

REFUSED  ungrounded
rails    injection ok · unsafe ok · language ok · relevance ok
         citation_spans ok · grounding BLOCK
```

Two things worth reading here.

**Speculation kept its result.** Divergence 0.15 is below the 0.25 threshold,
so the retrieval fired at 1618 ms stood and the core loop was genuinely hidden
inside the tail of the utterance. This is the one case where the trick pays —
and it is worth ~10 ms against a ~3.5 s round trip, which is why the README
reports it as ~0.3% rather than as a headline.

**The refusal is correct.** The transcript is `"Hat is a corporation"` — Sarvam
dropped the leading *W*, a TTS onset artifact both it and Whisper reproduce.
Retrieval returned passages that did not support an answer to that, the
output-side grounding rail refused, and the system said nothing rather than
inventing something. The input rails all passed; the block came from
groundedness, which is exactly the layer meant to catch it.
