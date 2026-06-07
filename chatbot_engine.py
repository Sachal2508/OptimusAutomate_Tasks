"""
Task 2: Chatbot with Intent Recognition
CodeAlpha AI Internship
Domain: University FAQ Chatbot (FAST-NUCES / General University)
Intent Recognition: TF-IDF + Cosine Similarity
Multi-turn: Context tracking with last-N-turns memory
"""

import json
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ──────────────────────────────────────────────
# Intent Knowledge Base
# ──────────────────────────────────────────────
INTENTS = [
    {
        "tag": "greeting",
        "patterns": [
            "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
            "what's up", "howdy", "greetings", "assalam o alaikum", "salam", "hola"
        ],
        "responses": [
            "Hello! Welcome to the University FAQ Bot. How can I help you today?",
            "Hi there! I'm here to answer your university-related questions. What do you need?",
            "Hey! Great to see you. Ask me anything about admissions, courses, or campus life!",
        ]
    },
    {
        "tag": "farewell",
        "patterns": [
            "bye", "goodbye", "see you", "later", "take care", "quit", "exit", "done",
            "thanks that's all", "nothing else", "I'm done"
        ],
        "responses": [
            "Goodbye! Good luck with your studies!",
            "Bye! Feel free to return anytime you have questions.",
            "Take care! Best of luck at university! 🎓"
        ]
    },
    {
        "tag": "admissions",
        "patterns": [
            "how to apply", "admission process", "when are admissions open", "entry test",
            "what are the requirements", "how to get admission", "application deadline",
            "admissions 2024", "NTS test", "NET test", "eligibility criteria",
            "can I apply", "admission requirements", "apply for admission"
        ],
        "responses": [
            "Admissions typically open in June-July for Fall semester. You need to: 1) Pass the Entry Test (NET/NTS), 2) Meet the minimum merit score, 3) Submit your FSc/A-Level transcripts online at the university portal. Check the official site for exact dates!",
            "To apply: visit the university admissions portal, fill the online form, upload required documents (CNIC, FSc marks), and pay the application fee. Entry test registration usually opens 2 months before the test date.",
        ]
    },
    {
        "tag": "fee_structure",
        "patterns": [
            "fee structure", "how much is the fee", "tuition fee", "semester fee",
            "fees per semester", "annual fees", "cost of study", "how expensive is it",
            "what are the fees", "financial information", "fee payment"
        ],
        "responses": [
            "Fees vary by program. For CS/Engineering programs, per-semester fees are typically PKR 90,000–130,000. Merit-based scholarships can reduce this significantly. Contact the accounts office or check the official website for the latest fee schedule.",
            "The fee structure depends on your program and semester. CS programs range from ~PKR 90K–130K/semester. Financial aid and scholarships are available for merit students. Visit the Finance Office for details.",
        ]
    },
    {
        "tag": "scholarships",
        "patterns": [
            "scholarship", "scholarships", "financial aid", "fee waiver", "need-based scholarship",
            "merit scholarship", "how to get scholarship", "scholarship criteria",
            "bursary", "financial assistance", "can I get scholarship", "scholarship apply",
            "are there scholarships", "any scholarships", "scholarships available",
            "do you offer scholarships", "scholarship information"
        ],
        "responses": [
            "Several scholarships are available: (1) Merit-based: top GPA students get 25-100% fee waivers. (2) Need-based: financial hardship cases can apply each semester. (3) External: HEC, PEEF, and corporate scholarships. Apply through the Student Affairs office.",
            "Scholarships include merit-based (GPA ≥ 3.5 typically qualifies), need-based (income-based), and government/HEC scholarships. Forms are available at the Registrar's office at the start of each semester.",
        ]
    },
    {
        "tag": "programs",
        "patterns": [
            "what programs", "available degrees", "courses offered", "departments",
            "BS programs", "MS programs", "what can I study", "fields of study",
            "computer science", "software engineering", "electrical engineering",
            "business administration", "MBA", "data science", "AI program",
            "what majors", "program list"
        ],
        "responses": [
            "Programs offered include: BS Computer Science, BS Software Engineering, BS Electrical Engineering, BS Business Administration, BS Data Science, BS Artificial Intelligence, MS/PhD programs in CS and EE. We also have MBA and BBA programs.",
            "Undergraduate programs: CS, SE, EE, Civil Engg, BBA, BS AI, BS Data Science. Graduate programs: MS/PhD in CS, EE, and Business. Check the official academics page for the full list with specializations.",
        ]
    },
    {
        "tag": "gpa_grades",
        "patterns": [
            "GPA", "grade policy", "grading system", "how is GPA calculated", "CGPA",
            "passing grade", "minimum GPA", "what is passing", "grade point",
            "academic standing", "probation", "dismissal policy", "grades"
        ],
        "responses": [
            "The grading scale: A (4.0) = 85-100%, B+ (3.5) = 80-84%, B (3.0) = 75-79%, C+ (2.5) = 70-74%, C (2.0) = 65-69%, D (1.0) = 60-64%, F (0.0) = below 60%. Minimum CGPA to avoid academic probation is usually 2.0.",
            "GPA is calculated on a 4.0 scale. You need a minimum CGPA of 2.0 to remain in good standing. Failing more than 2 courses in a semester may result in academic warning. Honours distinction requires CGPA ≥ 3.5.",
        ]
    },
    {
        "tag": "registration",
        "patterns": [
            "course registration", "how to register courses", "add drop", "withdraw from course",
            "credit hours", "how many courses", "semester registration", "enroll in course",
            "register for class", "course load", "max credits"
        ],
        "responses": [
            "Course registration opens 1-2 weeks before the semester starts through the student portal. You can register for 15-21 credit hours normally. Add/drop period is the first 2 weeks of semester. Withdrawal (with W grade) is allowed up to week 8.",
            "Log into the student portal → Academics → Course Registration. Select courses based on your semester plan and prerequisites. Typical credit load is 18 hours/semester. Contact your academic advisor if you need to exceed the limit.",
        ]
    },
    {
        "tag": "hostel",
        "patterns": [
            "hostel", "accommodation", "dorm", "dormitory", "where to live",
            "hostel fees", "hostel availability", "on campus housing", "hostel apply",
            "student housing", "girls hostel", "boys hostel"
        ],
        "responses": [
            "On-campus hostels are available for both male and female students (separate blocks). Hostel fee is around PKR 8,000–15,000/month including meals. Applications are submitted to the Hostel Management Office before semester start. Seats are limited!",
            "Hostel accommodation is available but limited. Boys' and girls' hostels are separate. Monthly cost: PKR 8K-15K (with food). Apply early through the Hostel Management Office. Off-campus options (PGs, shared apartments) are also popular.",
        ]
    },
    {
        "tag": "library",
        "patterns": [
            "library", "library hours", "library books", "digital library", "e-library",
            "research papers", "IEEE access", "ACM digital library", "library membership",
            "borrow books", "library timings", "library resources"
        ],
        "responses": [
            "The university library is open Monday–Saturday, 8 AM to 9 PM. It has 50,000+ physical books, plus digital access to IEEE Xplore, ACM Digital Library, Springer, and HEC Digital Library. Use your student ID to borrow books (2-week loan period).",
            "Library resources include physical books, journals, and online databases (IEEE, ACM, Springer via HEC). Library hours: Mon-Fri 8AM-9PM, Sat 9AM-5PM. Each student can borrow up to 5 books at a time.",
        ]
    },
    {
        "tag": "exam_schedule",
        "patterns": [
            "exam schedule", "when are exams", "midterm", "final exam", "exam dates",
            "quiz schedule", "assessment dates", "exam timetable", "when is final",
            "midterm dates", "exam hall ticket", "exam registration"
        ],
        "responses": [
            "Midterm exams are typically in week 8-9 of the semester, and finals are in weeks 16-18. The exact schedule is posted on the student portal and notice boards 2 weeks before exams. Check your portal for room assignments.",
            "Exam schedule: Midterms ~ Week 8 (30-40% of grade), Finals ~ Week 16-17 (40-50% of grade). Quizzes and assignments make up the rest. Schedules are released on the portal. Contact the Exam office for hall ticket issues.",
        ]
    },
    {
        "tag": "contact_info",
        "patterns": [
            "contact", "phone number", "email", "address", "location", "how to reach",
            "office hours", "administration contact", "registrar", "helpline",
            "where is the university", "how to contact", "support"
        ],
        "responses": [
            "Main contacts: Registrar Office: registrar@nu.edu.pk | Admissions: admissions@nu.edu.pk | IT Helpdesk: helpdesk@nu.edu.pk | Phone: (021) 111-128-128. Office hours: Mon-Fri 9 AM – 5 PM.",
            "You can reach us at: 📧 info@nu.edu.pk | 📞 (021) 111-128-128 | 📍 FAST-NUCES campuses in Karachi, Lahore, Islamabad, Peshawar, Faisalabad, and Chiniot-Faisalabad. Visit the website for campus-specific contacts.",
        ]
    },
    {
        "tag": "career_services",
        "patterns": [
            "career", "job placement", "internship", "career office", "job fair",
            "placement rate", "graduate jobs", "career counseling", "resume help",
            "industry connections", "campus recruitment", "job opportunities"
        ],
        "responses": [
            "The Career Development Center (CDC) offers resume workshops, mock interviews, and hosts 2 job fairs per year. Companies like Systems Ltd, IBM, Google, and local tech firms recruit directly on campus. Visit CDC in the admin block.",
            "Career services include: internship placements, on-campus recruitment by 100+ companies, career counseling, CV review, and LinkedIn workshops. The annual job fair is held in April. CS/SE graduates have ~90% placement within 6 months.",
        ]
    },
    {
        "tag": "mental_health",
        "patterns": [
            "counseling", "mental health", "stress", "anxiety", "therapist",
            "psychological support", "struggling academically", "feeling overwhelmed",
            "help me", "I need help", "burnout", "depression support"
        ],
        "responses": [
            "The university has a Student Counseling Center with qualified psychologists available for free sessions. You can book an appointment online or walk in. Your privacy is fully respected. Remember: seeking help is a sign of strength, not weakness. 💙",
            "Please reach out to our Student Counseling Center — confidential support is available for academic stress, anxiety, or personal issues. Sessions are free for enrolled students. You can also talk to your academic advisor or department chair.",
        ]
    },
    {
        "tag": "thanks",
        "patterns": [
            "thank you", "thanks", "thank you so much", "great", "helpful", "appreciate it",
            "that helps", "got it", "understood", "perfect", "wonderful", "awesome"
        ],
        "responses": [
            "You're welcome! Is there anything else I can help you with?",
            "Happy to help! Feel free to ask if you have more questions.",
            "Glad that was useful! Anything else on your mind?",
        ]
    },
    {
        "tag": "unknown",
        "patterns": [],
        "responses": [
            "I'm not sure I understand that. Could you rephrase your question? I can help with admissions, fees, programs, exams, hostels, scholarships, and more.",
            "Hmm, that's outside my knowledge area. Try asking about: admissions, fee structure, programs, scholarships, GPA/grades, registration, hostel, library, or exams.",
            "I didn't quite get that. I'm best at answering university-related questions — admissions, academics, campus life, fees, etc.",
        ]
    },
]

# ──────────────────────────────────────────────
# ChatbotEngine class
# ──────────────────────────────────────────────
class ChatbotEngine:
    CONFIDENCE_THRESHOLD = 0.15

    def __init__(self):
        self.intents = INTENTS
        self._build_vectorizer()
        self.context_window = []   # last N turns [(role, text)]
        self.last_intent = None

    def _build_vectorizer(self):
        """Build TF-IDF matrix from all intent patterns."""
        self.pattern_texts = []
        self.pattern_tags = []
        for intent in self.intents:
            for pat in intent["patterns"]:
                self.pattern_texts.append(pat.lower())
                self.pattern_tags.append(intent["tag"])
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            analyzer="word",
            stop_words="english",
            min_df=1,
        )
        if self.pattern_texts:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.pattern_texts)

    def _preprocess(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return text

    def _recognize_intent(self, user_input: str) -> tuple[str, float]:
        processed = self._preprocess(user_input)
        try:
            vec = self.vectorizer.transform([processed])
        except Exception:
            return "unknown", 0.0
        sims = cosine_similarity(vec, self.tfidf_matrix).flatten()
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        best_tag = self.pattern_tags[best_idx] if best_score >= self.CONFIDENCE_THRESHOLD else "unknown"
        return best_tag, best_score

    def _context_hint(self, user_input: str) -> str | None:
        """If user says 'more info' or 'tell me more', continue with last intent."""
        follow_ups = ["more", "tell me more", "elaborate", "explain more", "details", "and?", "what else"]
        if any(f in user_input.lower() for f in follow_ups) and self.last_intent:
            return self.last_intent
        return None

    def _get_response(self, tag: str) -> str:
        for intent in self.intents:
            if intent["tag"] == tag:
                responses = intent["responses"]
                # Round-robin to avoid repetition
                idx = hash(tag + str(len(self.context_window))) % len(responses)
                return responses[idx]
        return "I'm not sure how to answer that."

    def chat(self, user_message: str) -> dict:
        """Process user message, return response dict with intent, confidence, response."""
        # Context-based override
        context_tag = self._context_hint(user_message)
        if context_tag:
            tag, score = context_tag, 1.0
        else:
            tag, score = self._recognize_intent(user_message)

        response = self._get_response(tag)
        self.last_intent = tag if tag != "unknown" else self.last_intent

        # Track context
        self.context_window.append(("user", user_message))
        self.context_window.append(("bot", response))
        if len(self.context_window) > 20:
            self.context_window = self.context_window[-20:]

        return {
            "intent": tag,
            "confidence": round(score, 3),
            "response": response,
            "is_farewell": tag == "farewell",
        }

    def reset(self):
        self.context_window = []
        self.last_intent = None
