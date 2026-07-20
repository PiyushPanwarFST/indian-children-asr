"""
Generates understanding_notes.pdf -- a clean study document
for Indian Children Speech Recognition project.
Run this script every time new content is added.
"""

from fpdf import FPDF
from fpdf.enums import XPos, YPos

# ── Setup ──────────────────────────────────────────────────
class PDF(FPDF):

    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, "Indian Children Speech Recognition - Study Notes",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        self.ln(1)
        self.set_draw_color(180, 180, 180)
        self.line(self.l_margin, self.get_y(),
                  self.w - self.r_margin, self.get_y())
        self.ln(3)

    def footer(self):
        self.set_y(-13)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")

    # ── helpers ───────────────────────────────────────────

    def title_block(self, text):
        self.set_font("Helvetica", "B", 20)
        self.set_text_color(30, 30, 30)
        self.ln(4)
        self.multi_cell(0, 10, text, align="C",
                        new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)

    def subtitle(self, text):
        self.set_font("Helvetica", "", 11)
        self.set_text_color(90, 90, 90)
        self.multi_cell(0, 6, text, align="C",
                        new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(6)

    def section(self, num, text):
        self.ln(5)
        self.set_fill_color(44, 62, 80)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 9, f"  {num}.  {text}", fill=True,
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(3)
        self.set_text_color(30, 30, 30)

    def subsection(self, text):
        self.ln(2)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(41, 128, 185)
        self.multi_cell(0, 7, text,
                        new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(30, 30, 30)
        self.ln(1)

    def body(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 6, text,
                        new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)

    def note_box(self, text):
        """Orange-bordered note box"""
        self.set_fill_color(255, 249, 230)
        self.set_draw_color(230, 126, 34)
        self.set_line_width(0.5)
        x = self.get_x()
        y = self.get_y()
        self.set_font("Helvetica", "", 10)
        # measure height
        lines = text.split("\n")
        h = len(lines) * 6 + 6
        self.rect(x, y, self.w - self.l_margin - self.r_margin,
                  h, style="FD")
        self.set_xy(x + 3, y + 3)
        self.set_text_color(120, 60, 0)
        self.multi_cell(self.w - self.l_margin - self.r_margin - 6,
                        6, text,
                        new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(40, 40, 40)
        self.set_line_width(0.2)
        self.set_draw_color(0, 0, 0)
        self.ln(3)

    def code_block(self, text, color=(245, 245, 245),
                   border_color=(150, 150, 150)):
        """Monospaced block -- used for step-by-step flows"""
        self.set_fill_color(*color)
        self.set_draw_color(*border_color)
        self.set_line_width(0.3)
        x = self.get_x()
        y = self.get_y()
        lines = text.split("\n")
        h = len(lines) * 5.2 + 4
        # page break check
        if y + h > self.h - self.b_margin - 10:
            self.add_page()
            x = self.get_x()
            y = self.get_y()
        self.rect(x, y, self.w - self.l_margin - self.r_margin,
                  h, style="FD")
        self.set_xy(x + 3, y + 2)
        self.set_font("Courier", "", 8.5)
        self.set_text_color(30, 30, 30)
        self.multi_cell(self.w - self.l_margin - self.r_margin - 6,
                        5.2, text,
                        new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font("Helvetica", "", 10)
        self.set_draw_color(0, 0, 0)
        self.ln(3)

    def path_header(self, text, rgb):
        self.set_fill_color(*rgb)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 10)
        self.cell(0, 8, f"  {text}", fill=True,
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(40, 40, 40)
        self.ln(1)

    def two_col_table(self, headers, rows, col_widths):
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(44, 62, 80)
        self.set_text_color(255, 255, 255)
        for h, w in zip(headers, col_widths):
            self.cell(w, 7, f" {h}", fill=True, border=1)
        self.ln()
        self.set_font("Helvetica", "", 9)
        fill = False
        for row in rows:
            self.set_fill_color(240, 244, 248) if fill \
                else self.set_fill_color(255, 255, 255)
            self.set_text_color(30, 30, 30)
            for cell, w in zip(row, col_widths):
                self.cell(w, 6, f" {cell}", fill=True, border=1)
            self.ln()
            fill = not fill
        self.ln(3)


# ══════════════════════════════════════════════════════════
pdf = PDF()
pdf.set_margins(18, 20, 18)
pdf.set_auto_page_break(auto=True, margin=18)
pdf.add_page()

# ── Cover ─────────────────────────────────────────────────
pdf.title_block("My Research Understanding Notes")
pdf.subtitle(
    "Topic: Indian Children Speech Recognition\n"
    "Base Paper: CARE -- Dutta & Ganapathy, IEEE TASLP 2025\n"
    "Target Conference: ICASSP 2026"
)

pdf.set_draw_color(44, 62, 80)
pdf.set_line_width(0.8)
pdf.line(pdf.l_margin, pdf.get_y(),
         pdf.w - pdf.r_margin, pdf.get_y())
pdf.ln(6)

# ══════════════════════════════════════════════════════════
# SECTION 1 -- PROJECT OVERVIEW
# ══════════════════════════════════════════════════════════
pdf.section("1", "Project Overview")

pdf.body(
    "Goal: Build an ASR system specifically for Indian children's speech "
    "in Hindi, Marathi, and Indian English.\n"
    "Dataset: ASER -- 123.72 hours, 81,423 clips, 5,260 children aged 5-18.\n"
    "Base Paper: CARE framework (originally for SER) adapted for ASR.\n"
    "Target: ICASSP 2026."
)

# ══════════════════════════════════════════════════════════
# SECTION 2 -- ASER DATASET
# ══════════════════════════════════════════════════════════
pdf.section("2", "ASER Dataset Summary")

pdf.two_col_table(
    ["Property", "Value"],
    [
        ["Total audio clips",       "81,423"],
        ["Unique children",         "5,260"],
        ["Total duration",          "123.72 hours"],
        ["Average clip length",     "5.47 seconds"],
        ["Age range",               "5 to 18 years"],
        ["Hindi clips / duration",  "15,880 clips -- 53.84 hrs"],
        ["English clips / duration","60,867 clips -- 47.19 hrs"],
        ["Marathi clips / duration", "4,421 clips -- 22.39 hrs"],
        ["Rajasthan (Hindi)",       "31,114 clips -- 46.17 hrs"],
        ["UP (Hindi)",              "31,509 clips -- 44.09 hrs"],
        ["Maharashtra (Marathi)",   "18,800 clips -- 33.45 hrs"],
        ["Correct readings",        "65,008  (79.8%)"],
        ["Incorrect readings",      "16,404  (20.2%)"],
        ["ASR train split",         "10,939 clips -- 58.16 hrs"],
        ["ASR dev split",           " 1,404 clips --  7.68 hrs"],
        ["ASR test split",          " 1,530 clips --  7.35 hrs"],
    ],
    [80, 100]
)

pdf.note_box(
    "IMPORTANT -- Splits are speaker-independent (split by child_id).\n"
    "No child's voice appears in more than one split.\n"
    "This prevents speaker leakage and ensures honest WER evaluation.\n"
    "Split method: stratified by region, 80/10/10 ratio by number of children."
)

# ══════════════════════════════════════════════════════════
# SECTION 3 -- UNDERSTANDING CARE
# ══════════════════════════════════════════════════════════
pdf.section("3", "Understanding the CARE Paper")

pdf.subsection("What CARE Does (Simple Version)")
pdf.body(
    "CARE has two expert listeners looking at the same audio simultaneously:\n"
    "  - Semantic expert: understands the MEANING of what is said\n"
    "    (like a professor reading the words)\n"
    "  - Acoustic expert: understands HOW it is said\n"
    "    (pitch, speed, energy -- like a music producer)\n\n"
    "Both experts are trained at the same time using teacher-student learning.\n"
    "Their outputs are merged for the final task (emotion in CARE; text in our project).\n\n"
    "Paper reference: Section III-B, pages 3680-3681."
)

# ── Phase 1 ───────────────────────────────────────────────
pdf.add_page()
pdf.subsection("Phase 1: Pre-Training")
pdf.body(
    "Paper location: Section III-B, pages 3680-3681.\n\n"
    "Running example: Audio clip -- child saying 'Radha ke paas ek tota hai'\n"
    "(5 seconds long = 80,000 numbers of raw audio data)\n\n"
    "Key idea: THREE things happen to the same audio simultaneously.\n"
    "  PATH A and PATH B produce the CORRECT ANSWERS (frozen teachers).\n"
    "  PATH C produces the MODEL'S ATTEMPT (the student -- this trains)."
)

pdf.ln(2)
pdf.body("What raw audio looks like as data:")
pdf.code_block(
    "Raw Audio File (.mp3 / .wav)\n"
    "= just a list of numbers representing air pressure over time\n\n"
    "[0.002, 0.008, -0.003, 0.012, -0.007, 0.019, ...]\n"
    " <------------- 16,000 numbers per second ----------->\n\n"
    "A 5 second clip = 80,000 numbers in a list. Nothing else. Just numbers.\n"
    "This raw list of 80,000 numbers is the input to all three paths."
)

# PATH A
pdf.path_header("PATH A -- Correct Semantic Answer  (Y_text)", (41, 128, 185))
pdf.code_block(
    "INPUT\n"
    "============================================================\n"
    "Raw audio:  [0.002, 0.008, -0.003, 0.012, ...]  <- 80,000 numbers\n"
    "============================================================\n"
    "        |\n"
    "        v\n"
    "[WHISPER ASR -- converts audio to text]  (frozen, not trained)\n"
    "        |\n"
    "OUTPUT\n"
    "============================================================\n"
    "Plain text string: 'Radha ke paas ek tota hai'\n"
    "(just a sentence, like a WhatsApp message)\n"
    "============================================================\n"
    "        |\n"
    "        v\n"
    "[RoBERTa -- reads text, converts to meaning numbers]  (frozen)\n"
    "        |\n"
    "OUTPUT\n"
    "============================================================\n"
    "12 rows of vectors (one per RoBERTa layer), each = 768 numbers\n\n"
    "Layer 1:  [0.21, -0.43, 0.87, 0.12, ...]  <- 768 numbers\n"
    "Layer 2:  [0.33, -0.21, 0.54, 0.09, ...]  <- 768 numbers\n"
    "...\n"
    "Layer 12: [0.67, -0.89, 0.23, 0.45, ...]  <- 768 numbers\n"
    "============================================================\n"
    "        |\n"
    "        v\n"
    "[Average Pool -- collapse 12 rows into 1 single row]\n"
    "        |\n"
    "OUTPUT = Y_text  (THE CORRECT SEMANTIC ANSWER)\n"
    "============================================================\n"
    "ONE row: [0.45, -0.62, 0.58, 0.24, ...]  <- 768 numbers\n\n"
    "Shape:   768 numbers in a single row\n"
    "Meaning: mathematical summary of what this sentence MEANS\n"
    "============================================================"
)

# PATH B
pdf.path_header("PATH B -- Correct Acoustic Answer  (Y_pase)", (192, 57, 43))
pdf.code_block(
    "INPUT\n"
    "============================================================\n"
    "Same raw audio:  [0.002, 0.008, -0.003, 0.012, ...]  <- 80,000 numbers\n"
    "============================================================\n"
    "        |\n"
    "        v\n"
    "[PASE+ -- analyzes voice quality every 20ms]  (frozen)\n"
    "        |\n"
    "OUTPUT (before downsampling)\n"
    "============================================================\n"
    "50 rows x 256 numbers  (5 sec clip, 1 row per 20ms)\n\n"
    "Row 1  (0-20ms):   [0.12, 0.87, -0.34, ...]  <- 256 numbers\n"
    "Row 2  (20-40ms):  [0.45, 0.23, -0.67, ...]  <- 256 numbers\n"
    "...\n"
    "Row 50 (end):      [0.09, 0.56, -0.12, ...]  <- 256 numbers\n"
    "============================================================\n"
    "        |\n"
    "        v\n"
    "[Downsample by 2 -- keep every other row to match CARE speed]\n"
    "        |\n"
    "OUTPUT = Y_pase  (THE CORRECT ACOUSTIC ANSWER)\n"
    "============================================================\n"
    "25 rows x 256 numbers\n\n"
    "Row 1:  [0.12, 0.87, -0.34, ...]  <- 256 numbers\n"
    "Row 3:  [0.44, 0.21, -0.65, ...]  <- 256 numbers\n"
    "...\n"
    "Row 25: [0.08, 0.53, -0.11, ...]  <- 256 numbers\n\n"
    "Shape:   25 rows, each row has 256 numbers\n"
    "Meaning: voice quality measurements at 25 time points\n"
    "         (pitch, energy, speed at each moment in time)\n"
    "============================================================"
)

# PATH C
pdf.add_page()
pdf.path_header("PATH C -- CARE Model Outputs  (Y_sem_hat and Y_acoust_hat)",
                (39, 174, 96))
pdf.code_block(
    "INPUT\n"
    "============================================================\n"
    "Same raw audio (third time):  [0.002, 0.008, -0.003, ...]  <- 80,000 numbers\n"
    "============================================================\n"
    "        |\n"
    "        v\n"
    "[CNN Feature Extractor -- compress waveform to frames]  (TRAINS)\n"
    "        |\n"
    "OUTPUT\n"
    "============================================================\n"
    "25 rows x 768 numbers  (one row every 40ms)\n\n"
    "Row 1:  [0.33, 0.71, -0.22, ...]  <- 768 numbers\n"
    "Row 2:  [0.41, 0.68, -0.31, ...]  <- 768 numbers\n"
    "...\n"
    "Row 25: [0.28, 0.59, -0.18, ...]  <- 768 numbers\n"
    "============================================================\n"
    "        |\n"
    "        v\n"
    "[Common Encoder -- WavLM layers 1 to 6]  (TRAINS)\n"
    "Learns: phonemes, pitch boundaries, basic Indian sounds\n"
    "        |\n"
    "OUTPUT\n"
    "============================================================\n"
    "Still 25 rows x 768 numbers\n"
    "BUT now each row is richer -- contains phoneme-level understanding\n"
    "============================================================\n"
    "        |\n"
    "        |----------------------------|\n"
    "        v                            v\n"
    "[ACOUSTIC BRANCH]           [SEMANTIC BRANCH]\n"
    "WavLM layers 7-12           RoBERTa layers + Conv Adapters\n"
    "(TRAINS)                    (conv adapters TRAIN, RoBERTa frozen)\n"
    "        |                            |\n"
    "OUTPUT                       OUTPUT\n"
    "=================================    ================================\n"
    "25 rows x 256 numbers               ONE row x 768 numbers\n"
    "(FC layer reduces 768 to 256)       (average pooling across 25 rows)\n\n"
    "Y_acoust_hat:                        Y_sem_hat:\n"
    "Row 1:  [0.08, 0.79, -0.41, ...]    [0.39, -0.71, 0.49, ...]\n"
    "Row 2:  [0.32, 0.19, -0.71, ...]     <- 768 numbers ->\n"
    "...\n"
    "Row 25: [0.03, 0.49, -0.18, ...]\n"
    "================================     ================================"
)

# Summary table
pdf.subsection("Final Summary -- All 4 Outputs of Phase 1")
pdf.two_col_table(
    ["Symbol", "Shape", "Produced By", "Role"],
    [
        ["Y_text",       "768 numbers, 1 row",    "RoBERTa (frozen)",        "Correct semantic answer"],
        ["Y_pase",       "25 rows x 256 numbers", "PASE+ (frozen)",          "Correct acoustic answer"],
        ["Y_sem_hat",    "768 numbers, 1 row",    "Semantic branch (trains)","Model's semantic attempt"],
        ["Y_acoust_hat", "25 rows x 256 numbers", "Acoustic branch (trains)","Model's acoustic attempt"],
    ],
    [32, 48, 52, 48]
)

# Loss
pdf.subsection("Loss Calculation")
pdf.code_block(
    "SEMANTIC LOSS (L_sem):\n"
    "Correct answer Y_text    : [0.45, -0.62, 0.58, ...]  <- 768 numbers\n"
    "Model attempt  Y_sem_hat : [0.39, -0.71, 0.49, ...]  <- 768 numbers\n\n"
    "MSE = average of (each number difference)^2\n"
    "    = (0.45-0.39)^2 + (-0.62-(-0.71))^2 + (0.58-0.49)^2 + ...\n"
    "    = one single number  e.g.  0.043\n\n"
    "ACOUSTIC LOSS (L_acoust):\n"
    "Correct answer Y_pase       : 25 rows x 256 numbers\n"
    "Model attempt  Y_acoust_hat : 25 rows x 256 numbers\n\n"
    "MSE = average difference^2 across all 25 rows and 256 columns\n"
    "    = one single number  e.g.  0.071\n\n"
    "TOTAL LOSS:\n"
    "L_total = L_sem + L_acoust = 0.043 + 0.071 = 0.114\n\n"
    "Both losses are just ONE number each. Not vectors. Just a score.\n"
    "Model updates itself to make this smaller.\n"
    "After thousands of clips -> L_total near zero -> Phase 1 DONE."
)

pdf.note_box(
    "PHASE 1 CONCLUSION:\n"
    "After Phase 1, CARE is a trained model that can take raw audio\n"
    "and produce TWO things simultaneously:\n"
    "  - A 768-number vector summarizing the MEANING of what was said\n"
    "  - A 25x256 grid describing HOW it was said over time\n"
    "The teachers (RoBERTa and PASE+) are no longer needed.\n"
    "CARE learned to do their jobs itself -- from audio alone."
)

# ── Phase 2 ───────────────────────────────────────────────
pdf.add_page()
pdf.subsection("Phase 2: Fine-Tuning (Downstream Task)")

pdf.body(
    "Paper location: Section III-B-3 'Inference' (page 3681) + Section III-D-2 (page 3682)\n\n"
    "Phase 1 was about TEACHING the model.\n"
    "Phase 2 is about USING the model to predict emotions.\n\n"
    "Teachers (RoBERTa, PASE+) -> NO LONGER NEEDED\n"
    "CARE backbone from Phase 1 -> LOADED AND FROZEN (mostly)\n"
    "New thing that trains -> only 13 small weights + a tiny classifier"
)

pdf.path_header("STEP 1 -- Pass audio through FROZEN CARE backbone", (44, 62, 80))
pdf.code_block(
    "INPUT:  Raw audio [0.002, 0.008, ...] <- 80,000 numbers\n\n"
    "        v  CNN Feature Extractor  (FROZEN)\n"
    "        v  Common Encoder 6 layers  (FROZEN)\n"
    "        v  Acoustic Branch + Semantic Branch  (FROZEN)\n\n"
    "Everything is frozen. Nothing updates here in Phase 2.\n"
    "The backbone just produces outputs."
)

pdf.path_header("STEP 2 -- Collect ALL 13 Layer Outputs", (44, 62, 80))
pdf.code_block(
    "We tap the output at EVERY layer -- not just the final one.\n\n"
    "Layer 0  (CNN output)     : 25 rows x 768 numbers\n"
    "Layer 1  (Common layer 1) : 25 rows x 768 numbers\n"
    "Layer 2  (Common layer 2) : 25 rows x 768 numbers\n"
    "Layer 3  (Common layer 3) : 25 rows x 768 numbers\n"
    "Layer 4  (Common layer 4) : 25 rows x 768 numbers\n"
    "Layer 5  (Common layer 5) : 25 rows x 768 numbers\n"
    "Layer 6  (Common layer 6) : 25 rows x 768 numbers\n"
    "                                  <- 7 outputs so far (all 25x768)\n\n"
    "At each branch layer, semantic + acoustic are joined side by side:\n\n"
    "  Semantic layer 1: [0.21, 0.43, ...]  <- 768 numbers\n"
    "  Acoustic layer 1: [0.54, 0.11, ...]  <- 768 numbers\n"
    "  Joined:           [0.21, 0.43, ..., 0.54, 0.11, ...]  <- 1536 numbers\n\n"
    "Branch layer 1 : 25 rows x 1536 numbers\n"
    "Branch layer 2 : 25 rows x 1536 numbers\n"
    "Branch layer 3 : 25 rows x 1536 numbers\n"
    "Branch layer 4 : 25 rows x 1536 numbers\n"
    "Branch layer 5 : 25 rows x 1536 numbers\n"
    "Branch layer 6 : 25 rows x 1536 numbers\n"
    "                                  <- 6 more outputs (all 25x1536)\n\n"
    "TOTAL: 13 layer outputs collected.\n\n"
    "WHY all layers and not just the last?\n"
    "  Layer 1-3  -> basic phoneme patterns\n"
    "  Layer 4-6  -> rhythm, speaking rate\n"
    "  Layer 7-9  -> speaker characteristics\n"
    "  Layer 10-12-> high-level emotion content\n"
    "Using ALL = using everything the model learned."
)

pdf.path_header("STEP 3 -- Convex Combination (Learned Weighted Average)", (44, 62, 80))
pdf.code_block(
    "13 learned weights: w0, w1, w2 ... w12\n"
    "Rule: w0 + w1 + w2 + ... + w12 = 1  (must always sum to 1)\n\n"
    "Combined = w0 x Layer0 + w1 x Layer1 + ... + w12 x Layer12\n\n"
    "These 13 weights are the ONLY THING THAT TRAINS in Phase 2\n"
    "(along with the classifier below).\n\n"
    "Example learned weights:\n"
    "  w0  (CNN)       = 0.02  <- basic features, less useful for emotion\n"
    "  w1  (Common 1)  = 0.03\n"
    "  ...\n"
    "  w9  (Branch 3)  = 0.14  <- high weight -> most useful for emotion\n"
    "  ...\n"
    "  w12 (Branch 6)  = 0.06\n"
    "  SUM             = 1.00  (always)\n\n"
    "OUTPUT after combination: 25 rows x 1536 numbers"
)

pdf.path_header("STEP 4 -- Mean Pool Across Time", (44, 62, 80))
pdf.code_block(
    "We have 25 rows x 1536. Classifier needs ONE fixed-size vector.\n\n"
    "Row 1  : [0.51, 0.38, -0.44, ...]  <- 1536 numbers\n"
    "Row 2  : [0.47, 0.42, -0.39, ...]  <- 1536 numbers\n"
    "...\n"
    "Row 25 : [0.53, 0.35, -0.41, ...]  <- 1536 numbers\n"
    "              |\n"
    "          Average all rows\n"
    "              |\n"
    "OUTPUT: ONE row x 1536 numbers  [0.50, 0.38, -0.41, ...]\n"
    "This single vector represents the ENTIRE audio clip."
)

pdf.path_header("STEP 5 -- Classification Head", (44, 62, 80))
pdf.code_block(
    "INPUT: 1536 numbers\n"
    "        |\n"
    "        v\n"
    "[Linear Layer: 1536 -> 256]   <- TRAINS in Phase 2\n"
    "        |\n"
    "[ReLU activation]\n"
    "(makes all negative numbers = 0, keeps positives)\n"
    "        |\n"
    "        v\n"
    "[Linear Layer: 256 -> 4]      <- TRAINS in Phase 2\n"
    "(4 = number of emotion classes: Angry / Happy / Sad / Neutral)\n"
    "        |\n"
    "[Softmax -- converts 4 numbers to 4 probabilities summing to 1]\n"
    "        |\n"
    "OUTPUT\n"
    "Angry  : 0.85  (85%)\n"
    "Happy  : 0.05  ( 5%)\n"
    "Sad    : 0.08  ( 8%)\n"
    "Neutral: 0.02  ( 2%)\n"
    "-> Predicted emotion = ANGRY  (highest probability)"
)

pdf.path_header("STEP 6 -- Loss in Phase 2", (44, 62, 80))
pdf.code_block(
    "Phase 1 loss = MSE (comparing number vectors to teacher outputs)\n"
    "Phase 2 loss = Cross-Entropy (predicted class vs correct class label)\n\n"
    "Ground truth label : ANGRY\n"
    "Model prediction   : Angry 85%, Happy 5%, Sad 8%, Neutral 2%\n\n"
    "Cross-Entropy Loss = -log(probability given to correct class)\n"
    "                   = -log(0.85) = 0.16   <- small = good prediction\n\n"
    "Bad example:\n"
    "Model predicts: Angry 10%, Happy 60%, Sad 20%, Neutral 10%\n"
    "Loss = -log(0.10) = 2.30   <- large = bad prediction\n\n"
    "Only 13 weights + classification head update.\n"
    "Backbone stays completely frozen."
)

# Phase 1 vs Phase 2 table
pdf.subsection("Phase 1 vs Phase 2 -- Key Differences")
pdf.two_col_table(
    ["", "PHASE 1", "PHASE 2"],
    [
        ["Goal",           "Teach the backbone",         "Use backbone to predict"],
        ["Teachers needed","YES (RoBERTa + PASE+)",       "NO"],
        ["Labels needed",  "NO (unsupervised)",           "YES (emotion labels)"],
        ["What trains",    "Full backbone (CNN + Common + branches)", "Only 13 weights + classification head"],
        ["Loss type",      "MSE (vector comparison)",     "Cross-entropy (class prediction)"],
        ["Output",         "Y_sem_hat, Y_acoust_hat",     "Emotion probabilities"],
    ],
    [38, 70, 72]
)

pdf.note_box(
    "PHASE 2 CONCLUSION:\n"
    "Phase 2 is lightweight on purpose. The heavy work (understanding speech)\n"
    "was already done in Phase 1. Phase 2 just learns HOW TO READ those\n"
    "representations for the specific task (emotion detection here;\n"
    "text transcription in our Indian children ASR project)."
)

# ══════════════════════════════════════════════════════════
# SECTION 4 -- OUR PROJECT ARCHITECTURE (placeholder)
# ══════════════════════════════════════════════════════════
pdf.add_page()
pdf.section("4", "Our Proposed Architecture -- Indian Children ASR")
pdf.note_box(
    "TO BE ADDED -- Architecture design in progress.\n"
    "Will be filled after full CARE understanding is complete."
)

# ══════════════════════════════════════════════════════════
# SECTION 5 -- PLANNING & NEXT STEPS (placeholder)
# ══════════════════════════════════════════════════════════
pdf.section("5", "Planning and Next Steps")
pdf.note_box(
    "TO BE ADDED."
)

# ── Save ──────────────────────────────────────────────────
out = "/home/hp/Indain_children_spech/paper_draft/understanding_notes.pdf"
pdf.output(out)
print(f"PDF saved -> {out}")
