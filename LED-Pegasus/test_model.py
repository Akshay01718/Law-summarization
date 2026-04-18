import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_PATH = "legal_summarizer_finetuned/pegasus_v2"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)

text = """
Civil Appeal No. 3582 of 1986.
From the Judgment and Order dated 24.7.1985 of the Delhi High Court in Civil W.P. No. 435 of 1985.
section Rangarajan and Ms. Asha Rani Jain for the Appellant.
V.C. Mahajan, Ms. C.K. Sucharita and C.V. Subba Rao for the Respondents.
32 The Judgment of the Court was delivered by SEN, J.
This appeal by special leave directed against the judgment and order of the High Court of Delhi dated July 24, 1985 raises two questions, namely: (1) Was the Union of India justified in passing an order dated September 17, 1982 in terms of FR 25 declaring the appellant to be unfit to cross the efficiency bar as Assistant Engineer, Central Public Works Department at the stage of Rs.590 in the prerevised scale of pay of Rs.350 590 EB 900 as from October 5, 1966? And (2) Is the appellant entitled to interest on the delayed payment of his pension? This litigation has had a chequered career.
The appellant who was as Assistant Engineer in the Central Public Works Department was placed under suspension pending a departmental enquiry under r.12(2) of the Central Civil Services (Classification, Control & Appeal) Rules, 1965 on September 3, 1959.
He remained on suspension till May 25, 1970 when on repeated representations the Chief Engineer, Central Public Works Department revoked the order of suspension and he was reinstated in service.
During the aforesaid period of suspension, adverse remarks in his confidential reports for the period between April 1, 1957 and August 31, 1957 and between April 1, 1958 and March 31, 1959 were communicated to him on December 16, 1959.
After a period of nearly five years, the departmental proceedings culminated in an order of dismissal from service dated March 12, 1964 but the same on appeal by him, was set aside by the President of India by order dated October 4, 1966 with a direction for the holding of a fresh departmental inquiry under r. 29(1)(c) of the Rules, with a further direction that he shall continue to remain under suspension.
The order of suspension was revoked by the Chief Engineer on May 8, 1970 but the departmental proceedings were kept alive.
As a result of this, the appellant was reinstated in service on May 25, 1970.
Immediately thereafter, he made representation to the Department to pass an order under FR 54 for payment of full pay and allowances for the period of suspension i.e. the period between September 3, 1959 and May 25, 1970 but the same was rejected on the ground that departmental inquiry was still pending.
There was little or no progress in the departmental inquiry.
"""

# ===============================
# SMART HEAD + TAIL TRUNCATION
# ===============================

full_tokens = tokenizer(text)["input_ids"]
print("Original token length:", len(full_tokens))

max_len = 512
half = max_len // 2

if len(full_tokens) > max_len:
    head = full_tokens[:half]
    tail = full_tokens[-half:]
    combined_tokens = head + tail
else:
    combined_tokens = full_tokens

truncated_text = tokenizer.decode(combined_tokens, skip_special_tokens=True)

print("Final token length:", len(tokenizer(truncated_text)["input_ids"]))

# ===============================
# GENERATE SUMMARY
# ===============================

prompt = (
    "Provide a structured legal summary with sections for "
    "Facts, Issues, Procedural History, and Outcome:\n\n"
    + truncated_text
)

inputs = tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True,
    max_length=512
).to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=200,
        num_beams=4,
        early_stopping=True
    )

summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nGenerated Summary:\n")
print(summary)