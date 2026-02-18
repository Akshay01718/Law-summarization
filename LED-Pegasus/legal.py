import torch
import argparse
import re
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------- MODELS ----------------
LED_MODEL = "nsi319/legal-led-base-16384"
PEGASUS_MODEL = "nsi319/legal-pegasus"

MAX_LED_LEN = 16384
MAX_PEGASUS_LEN = 1024


# ---------------- DEVICE ----------------
def get_device():
    if torch.cuda.is_available():
        print(f"✓ Using CUDA: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("✓ Using Apple MPS")
        return torch.device("mps")
    else:
        print("✓ Using CPU")
        return torch.device("cpu")


# ---------------- TEXT UTILS ----------------
def split_sentences(text):
    text = re.sub(r"\s+", " ", text)
    return re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)


def extract_final_order(text):
    tail = text[-1500:]
    match = re.search(
        r"([^.]*?(appeal|petition).*?(allowed|dismissed)[^.]*\.)",
        tail,
        re.IGNORECASE,
    )
    return match.group(1) if match else ""


def extract_key_sentences(text, limit=40):
    sentences = split_sentences(text)
    scored = []

    for i, sent in enumerate(sentences):
        score = 0
        lower = sent.lower()

        if any(k in lower for k in [
            "held", "concluded", "we are of the view",
            "cannot", "suppression", "appointment",
            "disclosure", "execution"
        ]):
            score += 2

        if i < len(sentences) * 0.1:
            score += 1
        if i > len(sentences) * 0.85:
            score += 2

        if len(sent.split()) > 12:
            scored.append((score, sent))

    scored.sort(reverse=True, key=lambda x: x[0])
    return " ".join(s for _, s in scored[:limit])


# ---------------- MODEL WRAPPER ----------------
class Model:
    def __init__(self, model_id, device, is_led=False):
        self.device = device
        self.is_led = is_led
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)
        self.model.eval()

    def generate(self, text, max_len, min_len):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
        ).to(self.device)

        gen_kwargs = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "num_beams": 4,
            "min_length": min_len,
            "max_length": 420,
            "no_repeat_ngram_size": 3,
            "length_penalty": 1.3,
        }

        # ✅ ONLY LED USES GLOBAL ATTENTION
        if self.is_led:
            global_attention_mask = torch.zeros_like(inputs.input_ids)
            global_attention_mask[:, 0] = 1
            gen_kwargs["global_attention_mask"] = global_attention_mask

        with torch.no_grad():
            output = self.model.generate(**gen_kwargs)

        return self.tokenizer.decode(output[0], skip_special_tokens=True)


# ---------------- HYBRID PIPELINE ----------------
def hybrid_summarize(text):
    device = get_device()

    print("Loading LED...")
    led = Model(LED_MODEL, device, is_led=True)

    print("Loading PEGASUS...")
    pegasus = Model(PEGASUS_MODEL, device, is_led=False)

    final_order = extract_final_order(text)
    extracted = extract_key_sentences(text)

    # ---------- STAGE 1: LED (long-context grounding) ----------
    led_input = extracted + "\n\nFINAL ORDER:\n" + final_order
    led_summary = led.generate(
        led_input,
        max_len=MAX_LED_LEN,
        min_len=250,
    )

    # ---------- STAGE 2: PEGASUS (rewrite ONLY) ----------
    pegasus_input = (
        "Rewrite the following legal summary clearly and concisely. "
        "Do NOT add new facts, courts, laws, countries, or outcomes:\n\n"
        + led_summary
    )

    final_summary = pegasus.generate(
        pegasus_input,
        max_len=MAX_PEGASUS_LEN,
        min_len=120,
    )

    # ---------- HARD OUTCOME CHECK ----------
    if ("allowed" in final_order.lower()) != ("allowed" in final_summary.lower()):
        print("⚠ Outcome mismatch detected — returning LED summary")
        return led_summary

    return final_summary


# ---------------- CLI ----------------
def main():
    parser = argparse.ArgumentParser(description="Hybrid LED + PEGASUS Legal Summarizer")
    parser.add_argument("input_file", help="Input legal judgment (.txt)")
    args = parser.parse_args()

    text = Path(args.input_file).read_text(encoding="utf-8").strip()
    summary = hybrid_summarize(text)

    print("\n" + "=" * 60)
    print("FINAL HYBRID SUMMARY")
    print("=" * 60)
    print(summary)

    out_file = Path(args.input_file).stem + "_hybrid_summary.txt"
    Path(out_file).write_text(summary, encoding="utf-8")
    print(f"\nSaved to: {out_file}")


if __name__ == "__main__":
    main()
