"""Generate the LoRA fine-tuning dataset of workplace conversations.

What the adapter should learn
-----------------------------
1. Answer workplace questions in 1-2 short, friendly sentences (lower latency:
   fewer generated tokens).
2. Use ONLY the retrieved context, including table-style "record" chunks.
3. Say "I don't have that information in the company documents." when the
   context does not contain the answer (this is the main hallucination fix).
4. Handle everyday workplace requests (short replies, polite drafts) without
   inventing company facts.

No leakage: every example uses randomly generated FICTIONAL companies and
values. The evaluation company "Lumen Robotics" and its documents are never
used here, so the held-out evaluation measures generalisation.

Usage:  python training/make_dataset.py --n 1200 --out training/data
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.prompts import ABSTAIN, build_messages  # noqa: E402

COMPANIES = ["Norcrest Analytics", "Bluefin Logistics", "Orchid Health Systems", "Vantage Grid", "Pinecone Mobility",
             "Harborview Media", "Solace Biotech", "Crescent Fintech", "Ironwood Energy", "Tidewater Software",
             "Maple & Finch Retail", "Quartzline Semiconductors", "Kestrel Aerospace", "Aurora Learning",
             "Summit Ridge Insurance", "Copperleaf Foods"]
FIRST = ["Maya", "Arjun", "Sofia", "Daniel", "Chen", "Fatima", "Lucas", "Ananya", "Grace", "Omar", "Noah", "Leila",
         "Ethan", "Mei", "Carlos", "Zara", "Ivan", "Nia", "Ravi", "Hana"]
LAST = ["Patel", "Kim", "Garcia", "Okafor", "Novak", "Silva", "Chen", "Haddad", "Iyer", "Brown", "Tanaka", "Moreau",
        "Adeyemi", "Kowalski", "Singh", "Rossi"]
CITIES = ["Austin, TX", "Denver, CO", "Seattle, WA", "Chicago, IL", "Atlanta, GA", "Toronto, Canada", "Berlin, Germany",
          "Bengaluru, India", "Dublin, Ireland", "Raleigh, NC"]
TOOLS = {"hr": ["Workday", "BambooHR", "Rippling", "Gusto"], "exp": ["Expensify", "Concur", "Ramp", "Brex"],
         "pw": ["1Password", "Bitwarden", "LastPass", "Keeper"], "vpn": ["GlobalProtect", "Cisco AnyConnect",
         "Tailscale", "Zscaler"], "travel": ["Navan", "Egencia", "TravelPerk"], "wiki": ["Confluence", "Notion",
         "SharePoint"], "chat": ["Slack", "Microsoft Teams"]}
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
HOLIDAYS = [("Labor Day", "September"), ("Thanksgiving Day", "November"), ("Memorial Day", "May"),
            ("New Year's Day", "January"), ("Juneteenth", "June"), ("Independence Day", "July")]
TEAMS = ["Payroll", "IT Service Desk", "Security Operations", "Data Platform", "Customer Success", "Facilities",
         "Procurement", "Legal", "Talent Acquisition", "Site Reliability"]


def name(r):
    return f"{r.choice(FIRST)} {r.choice(LAST)}"


# Each template returns (title, passage, [questions], answer). `r` is a Random.
def t_pto(r, co):
    n = r.choice([12, 15, 18, 20, 22, 25])
    return ("Employee Handbook > Paid Time Off",
            f"Full-time employees at {co} receive {n} days of paid time off per calendar year, accrued monthly.",
            ["How many PTO days do I get per year?", "How much vacation time do full-time employees get?",
             "what's our annual paid time off allowance"],
            f"Full-time employees get {n} days of paid time off per year.")


def t_carry(r, co):
    n, m = r.choice([3, 5, 8, 10]), r.choice(["March 31", "June 30", "January 31"])
    return ("Employee Handbook > PTO Carryover",
            f"Employees may carry over up to {n} unused PTO days into the next year. Carried-over days expire on {m}.",
            ["Can I carry over unused vacation days?", "How many PTO days roll over to next year?",
             "When do carried over PTO days expire?"],
            f"You can carry over up to {n} unused PTO days, and they expire on {m}.")


def t_sick(r, co):
    n, d = r.choice([5, 8, 10, 12]), r.choice([2, 3, 5])
    return ("Employee Handbook > Sick Leave",
            f"Each employee receives {n} paid sick days per year. A doctor's note is required only for absences "
            f"longer than {d} consecutive working days.",
            ["How many sick days do we get?", "Do I need a doctor's note when I'm sick?"],
            f"You get {n} paid sick days per year, and a doctor's note is only needed for absences longer than "
            f"{d} consecutive working days.")


def t_parental(r, co):
    a, b = r.choice([12, 14, 16, 20]), r.choice([6, 8, 10, 12])
    return ("Benefits > Parental Leave",
            f"Birthing parents receive {a} weeks of paid parental leave and non-birthing parents receive {b} weeks.",
            ["How long is parental leave?", "How much paternity leave do non-birthing parents get?"],
            f"Birthing parents get {a} weeks of paid parental leave, and non-birthing parents get {b} weeks.")


def t_office_days(r, co):
    d1, d2 = r.sample(DAYS[:4], 2)
    return ("Hybrid Work Policy", f"Hybrid employees at {co} are expected in the office every {d1} and {d2}.",
            ["Which days do I need to be in the office?", "What are the in-office days for hybrid staff?"],
            f"Hybrid employees are expected in the office on {d1}s and {d2}s.")


def t_stipend(r, co):
    a, b = r.choice([300, 500, 750, 1000]), r.choice([40, 50, 60, 75])
    return ("Remote Work > Stipends",
            f"Remote employees receive a one-time home office stipend of ${a} and a monthly internet reimbursement "
            f"of ${b}.",
            ["Is there a home office stipend?", "Do we get reimbursed for internet?", "How much is the WFH stipend?"],
            f"Remote employees get a one-time ${a} home office stipend plus ${b} per month for internet.")


def t_learning(r, co):
    a = r.choice([1000, 1200, 2000, 2500, 3000])
    return ("Learning and Development", f"Every employee has an annual learning budget of ${a} for courses, books "
            "and conferences. Unused budget does not roll over.",
            ["What is my learning budget?", "Can I get a course reimbursed?", "Is there money for certifications?"],
            f"You have a ${a} annual learning budget for courses, books and conferences, and it does not roll over.")


def t_reviews(r, co):
    m1, m2 = r.choice([("March", "September"), ("April", "October"), ("January", "July"), ("June", "December")])
    return ("Performance Reviews", f"Performance reviews at {co} happen twice a year, in {m1} and {m2}.",
            ["When are performance reviews?", "How often do we have performance reviews?"],
            f"Performance reviews happen twice a year, in {m1} and {m2}.")


def t_expense_deadline(r, co):
    d, tool, rec = r.choice([30, 45, 60, 90]), r.choice(TOOLS["exp"]), r.choice([20, 25, 50, 75])
    return ("Expense Policy > Submitting Expenses", f"Business expenses must be submitted in {tool} within {d} days "
            f"of purchase. Receipts are required for any expense over ${rec}.",
            ["How long do I have to submit an expense?", "Where do I submit expenses?",
             "Do I need a receipt for a small expense?"],
            f"Submit expenses in {tool} within {d} days of the purchase, with receipts for anything over ${rec}.")


def t_meals(r, co):
    a, b = r.choice([50, 60, 75, 80]), r.choice([90, 100, 120])
    return ("Travel Policy > Meals", f"The daily meal allowance is ${a} for domestic travel and ${b} for "
            "international travel.",
            ["What's the meal allowance when traveling?", "How much can I spend on food per day on a business trip?"],
            f"The meal allowance is ${a} per day for domestic travel and ${b} per day for international travel.")


def t_hotel(r, co):
    a = r.choice([200, 225, 250, 275])
    return ("Travel Policy > Hotels", f"Hotel stays should cost no more than ${a} per night.",
            ["What's the hotel limit?", "How much can I spend on a hotel per night?"],
            f"Hotels should cost no more than ${a} per night.")


def t_flights(r, co):
    h, tool = r.choice([5, 6, 8]), r.choice(TOOLS["travel"])
    return ("Travel Policy > Flights", f"Flights must be booked through {tool}. Economy class is required for flights "
            f"under {h} hours; premium economy is allowed for longer flights.",
            ["Can I fly business class?", "How do I book a work flight?", "Can I book premium economy?"],
            f"Book flights through {tool}; economy is required under {h} hours and premium economy is allowed "
            "for longer flights.")


def t_approval(r, co):
    a, b = r.choice([250, 500, 1000]), r.choice([5000, 10000])
    return ("Expense Policy > Approval Limits", f"Expenses up to ${a} are approved by your manager. Expenses above "
            f"${b} require approval from Finance.",
            ["Who approves my expenses?", "Does Finance need to approve large purchases?"],
            f"Your manager approves expenses up to ${a}, and anything above ${b} needs Finance approval.")


def t_password(r, co):
    n, tool = r.choice([12, 14, 16]), r.choice(TOOLS["pw"])
    return ("IT Security > Passwords", f"Passwords must be at least {n} characters and stored in {tool}. "
            "Multi-factor authentication is mandatory for all company systems.",
            ["What are the password requirements?", "Which password manager should I use?", "Is MFA required?"],
            f"Passwords must be at least {n} characters, stored in {tool}, and MFA is required on all systems.")


def t_vpn(r, co):
    tool = r.choice(TOOLS["vpn"])
    return ("IT Security > Remote Access", f"Access to internal systems from outside the office requires the {tool} "
            "VPN. Public Wi-Fi may only be used with the VPN turned on.",
            ["Which VPN do we use?", "Can I use coffee shop Wi-Fi for work?"],
            f"Use the {tool} VPN for internal systems, and only use public Wi-Fi with the VPN on.")


def t_incident(r, co):
    ch, t = r.choice(["#sec-incidents", "#security", "#report-security"]), r.choice(["30 minutes", "1 hour", "2 hours"])
    return ("IT Security > Incident Reporting", f"Suspected security incidents such as phishing or lost devices must be "
            f"reported within {t} in the {ch} channel.",
            ["What do I do if I clicked a phishing link?", "How do I report a lost laptop?",
             "I got a suspicious email, what should I do?"],
            f"Report it within {t} in the {ch} channel so the security team can respond.")


def t_laptop(r, co):
    y, m = r.choice([2, 3, 4]), r.choice(["MacBook Pro", "Dell XPS 13", "ThinkPad T14"])
    return ("IT > Devices", f"New employees receive a {m}. Laptops are refreshed every {y} years.",
            ["What laptop do new hires get?", "How often are laptops replaced?"],
            f"New hires get a {m}, and laptops are refreshed every {y} years.")


def t_deploy(r, co):
    d = r.choice(["Monday through Thursday", "Monday through Wednesday", "Tuesday through Thursday"])
    return ("Engineering > Deployments", f"Production deployments are allowed {d} during business hours. Friday and "
            "weekend deployments are frozen except for critical fixes.",
            ["Can I deploy on Friday?", "When are production deployments allowed?"],
            f"Deployments are allowed {d} during business hours; Fridays and weekends are frozen except for "
            "critical fixes.")


def t_oncall(r, co):
    s, a = r.choice([250, 300, 400, 500]), r.choice([5, 10, 15, 30])
    return ("Engineering > On-Call", f"On-call engineers receive a stipend of ${s} per week. Pages must be "
            f"acknowledged within {a} minutes.",
            ["Do we get paid for on-call?", "How fast do I need to acknowledge a page?"],
            f"On-call pays ${s} per week, and pages must be acknowledged within {a} minutes.")


def t_review_approvals(r, co):
    n = r.choice([1, 2])
    word = "one approval" if n == 1 else "two approvals"
    return ("Engineering > Code Review", f"Every pull request needs at least {word} before it can be merged.",
            ["How many approvals does a PR need?", "Can I merge without a review?"],
            f"Every pull request needs at least {word} before merging.")


def t_401k(r, co):
    p = r.choice([3, 4, 5, 6])
    return ("Benefits > Retirement", f"{co} matches 100% of employee 401(k) contributions up to {p}% of salary.",
            ["Is there a 401k match?", "How much does the company match for retirement?"],
            f"The company matches 100% of your 401(k) contributions up to {p}% of your salary.")


def t_health_start(r, co):
    s = r.choice(["on your first day", "on the first day of the month after your start date", "after 30 days"])
    return ("Onboarding > Benefits", f"Health insurance coverage begins {s}.",
            ["When does my health insurance start?", "When do benefits kick in for new hires?"],
            f"Your health insurance coverage begins {s}.")


# Structured "record" chunks (the same format app/ingest.py produces for CSV/JSON)
def t_team_record(r, co):
    team, lead = r.choice(TEAMS), name(r)
    ch = "#" + team.lower().replace(" ", "-")
    return (f"team_directory row {r.randint(1, 9)}: {team}",
            f"team directory | team: {team} | lead: {lead} | slack channel: {ch} | email: "
            f"{team.split()[0].lower()}@example.com",
            [f"Who leads the {team} team?", f"How do I contact {team}?", f"What's the Slack channel for {team}?"],
            f"The {team} team is led by {lead}, and you can reach them in {ch}.")


def t_office_record(r, co):
    city, mgr, n = r.choice(CITIES), name(r), r.randint(40, 600)
    return (f"office_locations row {r.randint(1, 6)}: {city}",
            f"office locations | city: {city} | employees: {n} | office manager: {mgr} | parking: "
            f"{r.choice(['Free garage parking', 'Paid street parking', 'No parking; bike storage available'])}",
            [f"Who is the office manager in {city.split(',')[0]}?", f"How many people work in the "
             f"{city.split(',')[0]} office?"],
            f"The {city.split(',')[0]} office has {n} employees and its office manager is {mgr}.")


def t_holiday_record(r, co):
    h, month = r.choice(HOLIDAYS)
    day = r.randint(1, 28)
    date = f"2025-{['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'].index(month) + 1:02d}-{day:02d}"  # noqa: E501
    return (f"holidays_2025 row {r.randint(1, 12)}: {h}", f"holidays 2025 | holiday: {h} | date: {date} | "
            f"day of week: {r.choice(DAYS)} | offices: All",
            [f"When is {h} this year?", f"Is {h} a company holiday?"],
            f"Yes, {h} is a company holiday on {date}.")


def t_sla_record(r, co):
    p, fr = r.choice([("P1", "15 minutes"), ("P2", "1 hour"), ("P3", "4 hours")])
    res = r.choice(["4 hours", "1 business day", "2 business days"])
    return (f"it_support_sla: priorities {p[-1]}", f"it support sla | priority: {p} | first response: {fr} | "
            f"resolution target: {res}",
            [f"How fast does IT respond to a {p} ticket?", f"What is the resolution target for {p} tickets?"],
            f"For {p} tickets, IT responds within {fr} and aims to resolve within {res}.")


TEMPLATES = [t_pto, t_carry, t_sick, t_parental, t_office_days, t_stipend, t_learning, t_reviews, t_expense_deadline,
             t_meals, t_hotel, t_flights, t_approval, t_password, t_vpn, t_incident, t_laptop, t_deploy, t_oncall,
             t_review_approvals, t_401k, t_health_start, t_team_record, t_office_record, t_holiday_record, t_sla_record]

# Everyday workplace requests with no company facts involved (answered without context)
CONVERSATIONAL = [
    ("Can you help me write a quick message asking my manager for Friday off?",
     "Sure: \"Hi, would it be okay if I take this Friday off? My work is on track and I'll make sure everything is "
     "covered before I leave. Thanks!\""),
    ("How do I politely decline a meeting I don't need to attend?",
     "You could say: \"Thanks for the invite! I don't think I'm needed for this one, but please share the notes and "
     "I'm happy to help with any follow-ups.\""),
    ("Write a short thank-you note to a teammate who helped me debug.",
     "\"Thanks so much for jumping in to help me debug today. I really appreciate your time and patience!\""),
    ("How should I ask a coworker to review my pull request?",
     "Try: \"Hi! When you have a moment, could you review my PR? It's a small change and I'd love your feedback.\""),
    ("I'm running late to a meeting, what should I send?",
     "Send a quick note: \"Running about 5 minutes late, sorry! Please go ahead and start without me.\""),
    ("Give me a short status update template for standup.",
     "Yesterday: what you finished. Today: what you're working on. Blockers: anything you need help with."),
    ("How can I ask for feedback from my manager?",
     "You could ask: \"Could we spend a few minutes in our next 1:1 on feedback? I'd like to know what I'm doing "
     "well and where I can improve.\""),
    ("Draft a message to reschedule our 1:1.",
     "\"Hi! Something came up at our usual 1:1 time. Could we move it to later this week? Let me know what works.\""),
    ("How do I welcome a new teammate on Slack?",
     "Try: \"Welcome to the team! We're excited to have you. Feel free to reach out with any questions.\""),
    ("What's a professional way to follow up on an unanswered email?",
     "You could write: \"Just following up on my note below. Let me know if you need anything else from me.\""),
    ("Thanks for your help!", "You're welcome! Happy to help anytime."),
    ("Hi", "Hi there! How can I help you with your workplace questions today?"),
]

OUT_OF_SCOPE_NO_CONTEXT = [
    "How many PTO days do I get?", "What's our hotel limit?", "Who is the head of IT?", "When is the office closed?",
    "What's the company's 401k match?", "Which VPN do we use?", "What's the parental leave policy here?",
    "How much is the meal allowance?", "When are performance reviews?", "What is the on-call stipend?",
]


def make_examples(n: int, seed: int) -> list[dict]:
    r = random.Random(seed)
    rows: list[dict] = []
    while len(rows) < n:
        co = r.choice(COMPANIES)
        kind = r.random()
        if kind < 0.62:  # answerable with context (+ distractors)
            target = r.choice(TEMPLATES)
            title, passage, questions, answer = target(r, co)
            others = [t(r, co) for t in r.sample([t for t in TEMPLATES if t is not target], r.randint(2, 3))]
            passages = [(title, passage)] + [(o[0], o[1]) for o in others]
            r.shuffle(passages)
            rows.append({"messages": build_messages(r.choice(questions), passages)
                         + [{"role": "assistant", "content": answer}], "type": "grounded"})
        elif kind < 0.88:  # unanswerable: the answer is NOT in the context
            target = r.choice(TEMPLATES)
            _, _, questions, _ = target(r, co)
            others = [t(r, co) for t in r.sample([t for t in TEMPLATES if t is not target], r.randint(2, 4))]
            rows.append({"messages": build_messages(r.choice(questions), [(o[0], o[1]) for o in others])
                         + [{"role": "assistant", "content": ABSTAIN}], "type": "abstain"})
        elif kind < 0.95:  # everyday conversational request, no retrieval
            q, a = r.choice(CONVERSATIONAL)
            rows.append({"messages": build_messages(q, None) + [{"role": "assistant", "content": a}],
                         "type": "conversational"})
        else:  # company-specific question but retrieval disabled -> do not guess
            q = r.choice(OUT_OF_SCOPE_NO_CONTEXT)
            rows.append({"messages": build_messages(q, None) + [{"role": "assistant", "content": ABSTAIN}],
                         "type": "no_context_abstain"})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--out", default="training/data")
    args = ap.parse_args()

    rows = make_examples(args.n, args.seed)
    n_val = int(len(rows) * args.val_frac)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for split, data in (("train", rows[n_val:]), ("val", rows[:n_val])):
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = {}
    for row in rows:
        counts[row["type"]] = counts.get(row["type"], 0) + 1
    print(f"wrote {len(rows) - n_val} train / {n_val} val examples to {out}  mix={counts}")


if __name__ == "__main__":
    main()
