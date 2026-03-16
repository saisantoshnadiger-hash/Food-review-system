# ╔══════════════════════════════════════════════════════════════════╗
# ║        Hotel Bakasura — Food Review Sentiment Analysis          ║
# ║        pip install nltk pandas scikit-learn matplotlib          ║
# ║        seaborn pillow                                           ║
# ╚══════════════════════════════════════════════════════════════════╝

import tkinter as tk
from tkinter import ttk, messagebox
import sqlite3, pandas as pd, nltk, re, hashlib, math
import matplotlib.pyplot as plt, seaborn as sns, warnings
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
warnings.filterwarnings('ignore')

[nltk.download(p, quiet=True) for p in ['stopwords','punkt','wordnet','vader_lexicon','punkt_tab']]
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ── Palette ────────────────────────────────────────────────────────
BG      = "#070B14"   # deep navy black
PANEL   = "#0D1220"   # card background
SURFACE = "#111827"   # input / surface
BORDER  = "#1E2D45"   # subtle border
ACC     = "#F97316"   # saffron orange (brand)
ACC2    = "#FB923C"   # lighter orange
GOLD    = "#F59E0B"   # warm gold
GRN     = "#10B981"   # emerald
RED     = "#F43F5E"   # rose
BLU     = "#38BDF8"   # sky blue
PUR     = "#A78BFA"   # lavender
WHT     = "#F1F5F9"   # near-white
GRY     = "#64748B"   # slate grey
MUTED   = "#334155"   # muted
HOVER   = "#162032"   # hover state
SEL     = "#1C2D42"   # selected

FONTS = {
    "logo"    : ("Georgia",            22, "bold"),
    "title"   : ("Georgia",            17, "bold"),
    "heading" : ("Segoe UI Semibold",  13, "bold"),
    "body"    : ("Segoe UI",           11),
    "small"   : ("Segoe UI",            9),
    "mono"    : ("Consolas",           10),
    "stat"    : ("Georgia",            26, "bold"),
    "stat_sm" : ("Georgia",            18, "bold"),
}

FOODS = ['idli','dosa','vada','meals','veg biryani','egg biryani',
         'chicken biryani','mutton biryani','ice cream','noodles',
         'orange juice','pine apple juice','papaya juice']

MENU = {
    'idli'            : ("🫓", 40,  "Soft steamed rice cakes with sambar & chutney"),
    'dosa'            : ("🥞", 60,  "Crispy rice crepe with coconut chutney & sambar"),
    'vada'            : ("🍩", 50,  "Crispy fried lentil doughnut with coconut chutney"),
    'meals'           : ("🍱", 120, "Full South Indian meals with rice, dal, sabzi & more"),
    'veg biryani'     : ("🍚", 130, "Fragrant basmati rice with fresh vegetables & spices"),
    'egg biryani'     : ("🍳", 150, "Basmati rice layered with spiced boiled eggs"),
    'chicken biryani' : ("🍗", 180, "Aromatic biryani with tender chicken & whole spices"),
    'mutton biryani'  : ("🥩", 220, "Slow-cooked mutton biryani with rich gravy"),
    'ice cream'       : ("🍨", 80,  "Creamy scoops in assorted flavours"),
    'noodles'         : ("🍜", 100, "Stir-fried noodles with vegetables & sauces"),
    'orange juice'    : ("🍊", 60,  "Freshly squeezed chilled orange juice"),
    'pine apple juice': ("🍍", 60,  "Sweet & tangy fresh pineapple juice"),
    'papaya juice'    : ("🍈", 55,  "Healthy & refreshing fresh papaya juice"),
}

RAW = [
    ("C001","dosa","The dosa was crispy and absolutely delicious! Best I've ever had.",5),
    ("C002","chicken biryani","Excellent chicken biryani, very flavorful and aromatic rice.",5),
    ("C003","idli","Soft and fluffy idli with great sambar. Really enjoyed it.",4),
    ("C004","veg biryani","Good veg biryani, nicely cooked with fresh vegetables.",4),
    ("C005","ice cream","Amazing ice cream! Very creamy and sweet. Loved it.",5),
    ("C006","orange juice","Fresh orange juice, perfectly sweet and chilled.",5),
    ("C007","mutton biryani","Outstanding mutton biryani. Tender meat, rich spices.",5),
    ("C008","meals","The meals were wholesome and tasty. Great value for money.",4),
    ("C009","vada","Crispy vada with coconut chutney. Perfect combination!",4),
    ("C010","noodles","Very tasty noodles, well-seasoned and hot.",4),
    ("C011","papaya juice","Refreshing papaya juice! Healthy and tasty.",4),
    ("C012","pine apple juice","Sweet and tangy pineapple juice. Highly recommend!",5),
    ("C013","egg biryani","Egg biryani was super tasty and well-prepared.",4),
    ("C014","dosa","Dosa was soggy and not crispy at all. Very disappointing.",2),
    ("C015","chicken biryani","Chicken biryani was undercooked and tasteless. Bad experience.",1),
    ("C016","idli","Idli was too hard and cold. Not fresh at all.",1),
    ("C017","meals","Meals were bland and overpriced. Very poor quality.",2),
    ("C018","noodles","Noodles were overcooked and too oily. Did not like it.",2),
    ("C019","orange juice","Orange juice tasted artificial and too sweet. Waste of money.",1),
    ("C020","veg biryani","Veg biryani lacked flavor. Rice was overcooked and mushy.",2),
    ("C021","ice cream","Ice cream was not fresh. It had an odd taste. Terrible!",1),
    ("C022","vada","Vada was too oily and not properly cooked inside.",2),
    ("C023","mutton biryani","Mutton was tough and chewy. Highly disappointed.",2),
    ("C024","papaya juice","Papaya juice was watery and had no taste. Not good.",1),
    ("C025","egg biryani","Egg biryani smelled bad. I couldn't eat it at all.",1),
    ("C026","dosa","Nice crispy dosa with tasty chutney and sambar.",4),
    ("C027","chicken biryani","Good chicken biryani but slightly spicy for my taste.",3),
    ("C028","idli","Average idli. Nothing special but acceptable.",3),
    ("C029","meals","Nice full meals with variety. Would come again.",4),
    ("C030","mutton biryani","Rich and flavorful mutton biryani! Highly satisfying.",5),
]

# ── NLP Setup ──────────────────────────────────────────────────────
lem   = WordNetLemmatizer()
stops = set(stopwords.words('english'))
sia   = SentimentIntensityAnalyzer()

def clean(t):
    t = re.sub(r'[^a-zA-Z\s]', '', t.lower())
    return ' '.join(lem.lemmatize(w) for w in word_tokenize(t) if w not in stops)

def vader_pred(t):
    s = sia.polarity_scores(t)['compound']
    return 'positive' if s >= 0.05 else ('negative' if s <= -0.05 else 'neutral')

def build_model(dataframe):
    X  = dataframe['cleaned']
    y  = dataframe['sentiment'].map({'positive': 1, 'negative': 0})
    tf = TfidfVectorizer(max_features=500, ngram_range=(1,2))
    Xt = tf.fit_transform(X)
    st = y if (y.nunique() > 1 and len(y) >= 10) else None
    Xr, Xe, yr, ye = train_test_split(Xt, y, test_size=0.2, random_state=42, stratify=st)
    m  = LogisticRegression(max_iter=1000, random_state=42).fit(Xr, yr)
    return tf, m, accuracy_score(ye, m.predict(Xe))

df = pd.DataFrame(RAW, columns=['customer_id','food_item','review','rating'])
df['sentiment'] = df['rating'].apply(lambda x: 'positive' if x >= 3 else 'negative')
df['cleaned']   = df['review'].apply(clean)
tfidf, model, acc = build_model(df)

conn = sqlite3.connect('food_reviews.db')
cur  = conn.cursor()
cur.execute('CREATE TABLE IF NOT EXISTS food_reviews('
            'id INTEGER PRIMARY KEY AUTOINCREMENT,'
            'customer_id TEXT, food_item TEXT, review TEXT,'
            'rating INTEGER, sentiment TEXT)')
df.to_sql('food_reviews', conn, if_exists='replace', index=False)
conn.commit()

OID = "OWNER_001"
OHH = hashlib.sha256("restaurant@2024".encode()).hexdigest()


# ══════════════════════════════════════════════════════════════════
#  WIDGET FACTORY
# ══════════════════════════════════════════════════════════════════
def _lbl(parent, text, font=None, fg=WHT, bg=BG, **kw):
    return tk.Label(parent, text=text, font=font or FONTS["body"],
                    fg=fg, bg=bg, **kw)

def _btn(parent, text, cmd, bg=ACC, fg="white", font=None, **kw):
    b = tk.Button(parent, text=text, command=cmd,
                  font=font or FONTS["heading"],
                  bg=bg, fg=fg, relief="flat", bd=0,
                  activebackground=ACC2, activeforeground="white",
                  cursor="hand2", **kw)
    b.bind("<Enter>", lambda e: b.config(bg=_lighten(bg)))
    b.bind("<Leave>", lambda e: b.config(bg=bg))
    return b

def _lighten(hex_color):
    """Return a slightly lighter shade for hover."""
    try:
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        factor = 1.15
        r = min(255, int(r * factor))
        g = min(255, int(g * factor))
        b = min(255, int(b * factor))
        return f"#{r:02x}{g:02x}{b:02x}"
    except Exception:
        return hex_color

def _ent(parent, show=None, **kw):
    return tk.Entry(parent, font=FONTS["body"],
                    bg=SURFACE, fg=WHT, insertbackground=ACC,
                    relief="flat", bd=0, show=show or '', **kw)

def _card(parent, **kw):
    return tk.Frame(parent, bg=PANEL, **kw)

def _divider(parent, color=BORDER, padx=30):
    tk.Frame(parent, bg=color, height=1).pack(fill="x", padx=padx, pady=(4, 14))

def _section_label(parent, text):
    f = tk.Frame(parent, bg=BG)
    f.pack(fill="x", padx=30, pady=(2, 8))
    tk.Frame(f, bg=ACC, width=3).pack(side="left", fill="y", padx=(0, 10))
    _lbl(f, text, FONTS["heading"], ACC, BG).pack(side="left", anchor="w")

def _combo(parent, var, vals, width=22, **kw):
    sty = ttk.Style()
    sty.theme_use("clam")
    sty.configure("Dark.TCombobox",
                  fieldbackground=SURFACE, background=SURFACE,
                  foreground=WHT, arrowcolor=ACC,
                  selectbackground=SEL, selectforeground=WHT,
                  bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER)
    sty.map("Dark.TCombobox",
            fieldbackground=[("readonly", SURFACE)],
            foreground=[("readonly", WHT)])
    return ttk.Combobox(parent, textvariable=var, values=vals,
                        font=FONTS["body"], state="readonly",
                        width=width, style="Dark.TCombobox", **kw)

def _center_window(root, w, h):
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    root.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

def _stat_card(parent, value, label, color=WHT, width=None, min_width=130):
    kw = {"width": width} if width else {}
    f  = tk.Frame(parent, bg=SURFACE, **kw)
    f.pack(side="left", padx=7, pady=5, ipadx=16, ipady=14)
    if width:
        f.pack_propagate(False)
    _lbl(f, str(value), FONTS["stat_sm"], color, SURFACE).pack(pady=(8, 2))
    _lbl(f, label, FONTS["small"], GRY, SURFACE).pack(pady=(0, 8))
    return f

def _configure_treeview_style():
    sty = ttk.Style()
    sty.theme_use("clam")
    sty.configure("BK.Treeview",
                  background=PANEL, foreground=WHT,
                  fieldbackground=PANEL, rowheight=34,
                  font=FONTS["body"], borderwidth=0)
    sty.configure("BK.Treeview.Heading",
                  background=SURFACE, foreground=ACC,
                  font=FONTS["heading"], borderwidth=0,
                  relief="flat")
    sty.map("BK.Treeview",
            background=[("selected", SEL)],
            foreground=[("selected", WHT)])


# ══════════════════════════════════════════════════════════════════
#  MATPLOTLIB THEME
# ══════════════════════════════════════════════════════════════════
def _apply_chart_theme(ax_list):
    for ax in (ax_list if hasattr(ax_list, '__iter__') else [ax_list]):
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=GRY, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)
        ax.xaxis.label.set_color(GRY)
        ax.yaxis.label.set_color(GRY)
        ax.title.set_color(WHT)


# ══════════════════════════════════════════════════════════════════
#  SPLASH / LOGIN WINDOW
# ══════════════════════════════════════════════════════════════════
class LoginWindow:
    def __init__(self, root):
        self.root = root
        root.title("Hotel Bakasura")
        root.configure(bg=BG)
        root.resizable(False, False)
        _center_window(root, 460, 600)
        self._build()

    def _build(self):
        r = self.root

        # ── Brand strip ──
        brand = tk.Frame(r, bg=BG)
        brand.pack(pady=(30, 0))

        # Decorative top accent line
        tk.Frame(r, bg=ACC, height=3).pack(fill="x")
        tk.Frame(r, bg=BG, height=20).pack()

        _lbl(brand, "🍽️", ("Segoe UI Emoji", 38), ACC, BG).pack()
        _lbl(brand, "Hotel Bakasura", FONTS["logo"], WHT, BG).pack(pady=(4, 0))
        _lbl(brand, "FOOD REVIEW  ·  SENTIMENT ANALYSIS",
             ("Segoe UI", 8, "bold"), GOLD, BG, letterSpacing=4).pack(pady=(2, 0))

        tk.Frame(r, bg=BG, height=20).pack()

        # ── Card ──
        card = tk.Frame(r, bg=PANEL, bd=0)
        card.pack(padx=40, fill="x")

        # Top accent bar on card
        tk.Frame(card, bg=ACC, height=2).pack(fill="x")

        inner = tk.Frame(card, bg=PANEL)
        inner.pack(fill="x", padx=28, pady=20)

        _lbl(inner, "Sign In", FONTS["title"], WHT, PANEL).pack(anchor="w", pady=(0, 14))

        # Role selector
        _lbl(inner, "LOGIN AS", FONTS["small"], GRY, PANEL).pack(anchor="w")
        role_frame = tk.Frame(inner, bg=SURFACE)
        role_frame.pack(fill="x", pady=(4, 16))
        self.role = tk.StringVar(value="owner")

        for txt, val, col in [("👑  Owner", "owner", ACC), ("🧑  Customer", "customer", BLU)]:
            rb = tk.Radiobutton(role_frame, text=txt, variable=self.role, value=val,
                                bg=SURFACE, fg=WHT, selectcolor=SEL,
                                activebackground=SURFACE, activeforeground=col,
                                font=FONTS["body"], cursor="hand2",
                                command=self._toggle_fields)
            rb.pack(side="left", padx=16, pady=10)

        # ID field
        self._id_lbl = _lbl(inner, "OWNER ID", FONTS["small"], GRY, PANEL)
        self._id_lbl.pack(anchor="w")
        self.id_ent = _ent(inner)
        self.id_ent.pack(fill="x", pady=(4, 14), ipady=9)
        self.id_ent.insert(0, "OWNER_001")

        # Password field
        self._pw_lbl = _lbl(inner, "PASSWORD", FONTS["small"], GRY, PANEL)
        self._pw_lbl.pack(anchor="w")
        self.pw_ent = _ent(inner, show="●")
        self.pw_ent.pack(fill="x", pady=(4, 4), ipady=9)
        self.pw_ent.insert(0, "restaurant@2024")

        _lbl(inner, "Default: restaurant@2024", FONTS["small"], MUTED, PANEL).pack(
            anchor="w", pady=(2, 16))

        # Login button
        _btn(inner, "SIGN IN  →", self._login).pack(fill="x", ipady=12)

        # Footer hint
        tk.Frame(r, bg=BG, height=10).pack()
        _lbl(r, "Customer? Switch role above & enter your name — no password needed.",
             FONTS["small"], GRY, BG, wraplength=380, justify="center").pack()

        # Bindings
        r.bind("<Return>", lambda e: self._login())
        self.id_ent.bind("<Return>", lambda e: self.pw_ent.focus_set())
        self.pw_ent.bind("<Return>", lambda e: self._login())

    def _toggle_fields(self):
        if self.role.get() == "customer":
            self._id_lbl.config(text="YOUR NAME")
            self._pw_lbl.config(text="PASSWORD  (not required)")
            self.pw_ent.config(state="disabled", bg=MUTED)
        else:
            self._id_lbl.config(text="OWNER ID")
            self._pw_lbl.config(text="PASSWORD")
            self.pw_ent.config(state="normal", bg=SURFACE)

    def _login(self):
        uid  = self.id_ent.get().strip()
        pw   = self.pw_ent.get().strip()
        role = self.role.get()
        if role == "owner":
            if uid == OID and hashlib.sha256(pw.encode()).hexdigest() == OHH:
                self.root.destroy(); launch_main("owner", uid)
            else:
                messagebox.showerror("Access Denied", "❌  Invalid Owner ID or Password!")
        else:
            if not uid:
                messagebox.showwarning("Name Required", "Please enter your name."); return
            self.root.destroy(); launch_main("customer", uid)


# ══════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════
class App:
    def __init__(self, root, role, username):
        self.root     = root
        self.role     = role
        self.user     = username
        self._nav_btns  = {}
        self._active    = None
        root.title(f"Hotel Bakasura  —  {role.capitalize()}: {username}")
        root.configure(bg=BG)
        root.resizable(True, True)
        _center_window(root, 1240, 780)
        _configure_treeview_style()
        self._build_shell()

    # ── Shell layout ─────────────────────────────────────────────
    def _build_shell(self):
        # ── Top bar ──
        topbar = tk.Frame(self.root, bg=PANEL, height=56)
        topbar.pack(fill="x")
        topbar.pack_propagate(False)
        tk.Frame(topbar, bg=ACC, width=4).pack(side="left", fill="y")
        _lbl(topbar, "🍽️  Hotel Bakasura",
             FONTS["title"], WHT, PANEL).pack(side="left", padx=18, pady=14)
        _lbl(topbar, "Food Review Sentiment Analysis",
             FONTS["small"], GRY, PANEL).pack(side="left", pady=14)

        # Role badge
        badge_bg = ACC if self.role == "owner" else BLU
        badge_f  = tk.Frame(topbar, bg=badge_bg)
        badge_f.pack(side="right", padx=12, pady=14)
        _lbl(badge_f, f"  {self.role.upper()}  ",
             ("Segoe UI", 8, "bold"), "white", badge_bg).pack(ipady=3, ipadx=2)

        _lbl(topbar, f"  {self.user}", FONTS["body"], GRY, PANEL).pack(
            side="right", pady=14)
        _lbl(topbar, "👤", ("Segoe UI Emoji", 14), GRY, PANEL).pack(
            side="right", pady=14, padx=(12, 0))

        # ── Body row ──
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True)

        # ── Sidebar ──
        self._sidebar = tk.Frame(body, bg=PANEL, width=220)
        self._sidebar.pack(fill="y", side="left")
        self._sidebar.pack_propagate(False)

        tk.Frame(self._sidebar, bg=BORDER, height=1).pack(fill="x")
        _lbl(self._sidebar, "NAVIGATION",
             FONTS["small"], GRY, PANEL).pack(anchor="w", padx=20, pady=(20, 8))

        # ── Scrollable content area ──
        outer = tk.Frame(body, bg=BG)
        outer.pack(fill="both", expand=True)

        self._canvas = tk.Canvas(outer, bg=BG, highlightthickness=0)
        self._vscroll = ttk.Scrollbar(outer, orient="vertical",
                                      command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=self._vscroll.set)
        self._vscroll.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)

        self.content = tk.Frame(self._canvas, bg=BG)
        self._win_id = self._canvas.create_window((0, 0),
                                                   window=self.content,
                                                   anchor="nw")
        self.content.bind("<Configure>",
            lambda e: self._canvas.configure(
                scrollregion=self._canvas.bbox("all")))
        self._canvas.bind("<Configure>",
            lambda e: self._canvas.itemconfig(self._win_id, width=e.width))

        def _scroll(e):
            if e.delta:
                self._canvas.yview_scroll(int(-e.delta / 120), "units")
            elif e.num == 4:
                self._canvas.yview_scroll(-1, "units")
            elif e.num == 5:
                self._canvas.yview_scroll(1, "units")

        for w in (self._canvas, self.content, self.root):
            w.bind("<MouseWheel>", _scroll)
            w.bind("<Button-4>",   _scroll)
            w.bind("<Button-5>",   _scroll)

        # ── Navigation buttons ──
        if self.role == "owner":
            pages = [
                ("📊", "Dashboard",    self.pg_dashboard),
                ("🔍", "Analyze Data", self.pg_analyze),
                ("📋", "All Reviews",  self.pg_reviews),
                ("📈", "Food Stats",   self.pg_stats),
                ("⚙️", "ML Model",     self.pg_model),
            ]
        else:
            pages = [("📝", "Submit Review", self.pg_submit)]

        for icon, label, cmd in pages:
            full = f"{icon}  {label}"
            frame = tk.Frame(self._sidebar, bg=PANEL)
            frame.pack(fill="x", pady=1)

            b = tk.Button(
                frame, text=full,
                font=FONTS["body"],
                bg=PANEL, fg=WHT, relief="flat", bd=0,
                activebackground=HOVER, activeforeground=ACC,
                cursor="hand2", anchor="w", padx=20,
                command=lambda c=cmd, lb=full: self._navigate(c, lb)
            )
            b.pack(fill="x", ipady=12)

            b.bind("<Enter>",
                   lambda e, x=b: x.config(bg=HOVER) if x.cget("bg") != ACC else None)
            b.bind("<Leave>",
                   lambda e, x=b: x.config(bg=PANEL) if x.cget("bg") != ACC else None)

            self._nav_btns[full] = b

        # Spacer + logout
        tk.Frame(self._sidebar, bg=BG).pack(fill="both", expand=True)
        tk.Frame(self._sidebar, bg=BORDER, height=1).pack(fill="x", padx=12, pady=8)
        _btn(self._sidebar, "  🚪  Logout", self._logout,
             PANEL, RED, FONTS["small"]).pack(fill="x", ipady=10, pady=(0, 10))

        # ── First page ──
        first_cmd, first_lbl = (
            (self.pg_submit, f"📝  Submit Review")
            if self.role == "customer"
            else (self.pg_dashboard, f"📊  Dashboard")
        )
        self._navigate(first_cmd, first_lbl)

    # ── Nav helpers ───────────────────────────────────────────────
    def _navigate(self, cmd, label):
        for lb, b in self._nav_btns.items():
            b.config(bg=ACC if lb == label else PANEL,
                     fg=WHT)
        self._active = label
        cmd()

    def _clear(self):
        for w in self.content.winfo_children():
            w.destroy()
        self._canvas.yview_moveto(0)
        plt.close('all')

    def _page_header(self, title, subtitle=""):
        hf = tk.Frame(self.content, bg=BG)
        hf.pack(fill="x", padx=30, pady=(28, 6))

        left = tk.Frame(hf, bg=BG)
        left.pack(side="left", fill="y")
        tk.Frame(left, bg=ACC, width=4, height=36).pack(side="left", fill="y", padx=(0, 12))
        text_f = tk.Frame(left, bg=BG)
        text_f.pack(side="left")
        _lbl(text_f, title, FONTS["title"], WHT, BG).pack(anchor="w")
        if subtitle:
            _lbl(text_f, subtitle, FONTS["small"], GRY, BG).pack(anchor="w", pady=(2, 0))

        tk.Frame(self.content, bg=BORDER, height=1).pack(fill="x", padx=30, pady=(10, 18))

    def _logout(self):
        self.root.destroy()
        start_login()

    # ══════════════════════════════════════════════════════════════
    #  PAGE: SUBMIT REVIEW
    # ══════════════════════════════════════════════════════════════
    def pg_submit(self, preselect=None):
        self._clear()
        self._page_header("📝  Submit a Review",
                          "Share your dining experience — your feedback helps us improve")

        wrap = tk.Frame(self.content, bg=BG)
        wrap.pack(fill="both", expand=True, padx=30, pady=5)

        # ── Left: form ──
        form_card = _card(wrap)
        form_card.pack(side="left", fill="both", expand=True, padx=(0, 12))

        inner = tk.Frame(form_card, bg=PANEL)
        inner.pack(fill="x", padx=24, pady=20)

        # Customer ID
        _lbl(inner, "CUSTOMER ID", FONTS["small"], GRY, PANEL).pack(anchor="w")
        cid_ent = _ent(inner)
        cid_ent.pack(fill="x", pady=(4, 14), ipady=9)
        cid_ent.insert(0, f"C{len(df)+1:03d}")

        # Food item
        _lbl(inner, "FOOD ITEM", FONTS["small"], GRY, PANEL).pack(anchor="w")
        food_var = tk.StringVar(value=preselect or FOODS[0])
        _combo(inner, food_var, FOODS, width=40).pack(fill="x", pady=(4, 14), ipady=6)

        # Star rating
        _lbl(inner, "YOUR RATING", FONTS["small"], GRY, PANEL).pack(anchor="w")

        HINTS   = ["", "😞 Terrible", "😕 Poor", "😐 Average", "🙂 Good", "😍 Excellent"]
        HCOLORS = ["", RED, RED, GOLD, GRN, GRN]
        rating_var = tk.IntVar(value=5)
        star_row   = tk.Frame(inner, bg=PANEL)
        star_row.pack(anchor="w", pady=(4, 2))
        hint_lbl = _lbl(inner, HINTS[5], FONTS["small"], GRN, PANEL)
        hint_lbl.pack(anchor="w", pady=(0, 12))
        stars = []

        def set_rating(r):
            rating_var.set(r)
            hint_lbl.config(text=HINTS[r], fg=HCOLORS[r])
            for i, s in enumerate(stars):
                s.config(fg=GOLD if i < r else MUTED,
                         font=("Segoe UI Emoji", 20 if i < r else 17))

        for i in range(1, 6):
            s = tk.Button(star_row, text="★",
                          font=("Segoe UI Emoji", 20),
                          bg=PANEL, fg=GOLD, relief="flat", bd=0,
                          cursor="hand2",
                          command=lambda r=i: set_rating(r))
            s.pack(side="left", padx=2)
            stars.append(s)

        # Review text
        _lbl(inner, "YOUR REVIEW", FONTS["small"], GRY, PANEL).pack(anchor="w")
        review_box = tk.Text(inner, font=FONTS["body"],
                             bg=SURFACE, fg=WHT, insertbackground=ACC,
                             relief="flat", bd=0, height=5, wrap="word")
        review_box.pack(fill="x", pady=(4, 2))
        char_lbl = _lbl(inner, "0 / 300", FONTS["small"], GRY, PANEL)
        char_lbl.pack(anchor="e", pady=(0, 6))

        def _count(*_):
            n = len(review_box.get("1.0", "end").strip())
            char_lbl.config(text=f"{n} / 300",
                            fg=RED if n > 300 else GRY)

        review_box.bind("<KeyRelease>", _count)

        result_lbl = _lbl(inner, "", FONTS["body"], GRN, PANEL,
                          wraplength=500, justify="left")

        def submit():
            global df, tfidf, model, acc
            rev = review_box.get("1.0", "end").strip()
            if not rev:
                messagebox.showwarning("Missing", "Please write a review!")
                return
            if len(rev) > 300:
                messagebox.showwarning("Too Long", "Keep your review under 300 characters.")
                return
            sent = 'positive' if rating_var.get() >= 3 else 'negative'
            cl   = clean(rev)
            df   = pd.concat([df, pd.DataFrame([{
                "customer_id": cid_ent.get().strip(),
                "food_item": food_var.get(),
                "review": rev, "rating": rating_var.get(),
                "sentiment": sent, "cleaned": cl
            }])], ignore_index=True)
            cur.execute(
                "INSERT INTO food_reviews(customer_id,food_item,review,rating,sentiment)"
                "VALUES(?,?,?,?,?)",
                (cid_ent.get().strip(), food_var.get(), rev, rating_var.get(), sent))
            conn.commit()
            try:
                tfidf, model, acc = build_model(df)
            except Exception:
                pass
            icon = "✅" if sent == "positive" else "❌"
            msg  = ("Positive review recorded — thank you!"
                    if sent == "positive"
                    else "Negative review noted. We'll work to improve!")
            col  = GRN if sent == "positive" else RED
            result_lbl.config(text=f"  {icon}  {msg}", fg=col,
                               bg=SURFACE)
            result_lbl.pack(fill="x", pady=(4, 14))
            cid_ent.delete(0, "end")
            cid_ent.insert(0, f"C{len(df)+1:03d}")
            review_box.delete("1.0", "end")
            set_rating(5); _count()

        _btn(inner, "SUBMIT REVIEW  ✈", submit).pack(
            fill="x", ipady=12, pady=(8, 4))
        result_lbl.pack(fill="x", pady=(0, 4))

        # ── Right: tips panel ──
        tips_card = _card(wrap, width=270)
        tips_card.pack(side="left", fill="y")
        tips_card.pack_propagate(False)

        tip_inner = tk.Frame(tips_card, bg=PANEL)
        tip_inner.pack(fill="both", expand=True, padx=16, pady=20)

        _lbl(tip_inner, "💡  Review Tips",
             FONTS["heading"], ACC, PANEL).pack(anchor="w", pady=(0, 12))

        for tip in ["Be specific about the dish",
                    "Mention taste, texture & portion size",
                    "Keep it honest and respectful",
                    "Rate fairly — 5★ = excellent"]:
            tf = tk.Frame(tip_inner, bg=SURFACE)
            tf.pack(fill="x", pady=3)
            tk.Frame(tf, bg=ACC, width=3).pack(side="left", fill="y")
            _lbl(tf, tip, FONTS["small"], GRY, SURFACE,
                 wraplength=200, justify="left").pack(
                     side="left", padx=10, pady=8)

        tk.Frame(tip_inner, bg=BORDER, height=1).pack(fill="x", pady=14)
        _lbl(tip_inner, "📊  At a Glance",
             FONTS["body"], WHT, PANEL).pack(anchor="w", pady=(0, 8))

        total   = len(df)
        pos_pct = round(len(df[df.sentiment == 'positive']) / total * 100) if total else 0
        for v, l in [(total, "Total reviews"),
                     (f"{pos_pct}%", "Positive rate"),
                     (len(FOODS), "Items available")]:
            sf = tk.Frame(tip_inner, bg=SURFACE)
            sf.pack(fill="x", pady=3)
            _lbl(sf, str(v), FONTS["stat_sm"], ACC, SURFACE).pack(
                side="left", padx=12, pady=8)
            _lbl(sf, l, FONTS["small"], GRY, SURFACE).pack(side="left")

    # ══════════════════════════════════════════════════════════════
    #  PAGE: DASHBOARD
    # ══════════════════════════════════════════════════════════════
    def pg_dashboard(self):
        self._clear()
        self._page_header("📊  Analytics Dashboard",
                          "Visual overview of all reviews in the database")

        fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5))
        fig.patch.set_facecolor(BG)
        fig.subplots_adjust(hspace=0.38, wspace=0.32,
                            left=0.08, right=0.97,
                            top=0.93, bottom=0.1)
        _apply_chart_theme(axes.flat)

        # 1. Sentiment pie
        sc = df['sentiment'].value_counts()
        wedge_props = {'linewidth': 2, 'edgecolor': BG}
        axes[0, 0].pie(sc, labels=sc.index, autopct='%1.1f%%',
                       colors=[GRN, RED], startangle=90,
                       wedgeprops=wedge_props,
                       textprops={'color': WHT, 'fontsize': 10})
        axes[0, 0].set_title('Sentiment Distribution', fontsize=11, pad=12)

        # 2. Avg rating bar
        avg_f  = df.groupby('food_item')['rating'].mean().sort_values()
        colors = [GRN if r >= 3.5 else GOLD if r >= 2.5 else RED
                  for r in avg_f.values]
        axes[0, 1].barh(avg_f.index, avg_f.values, color=colors,
                         edgecolor="none", height=0.65)
        axes[0, 1].axvline(x=3, color=ACC, linestyle='--',
                            linewidth=1.2, alpha=0.8, label='Avg threshold')
        axes[0, 1].set_title('Avg Rating per Food Item', fontsize=11, pad=12)
        axes[0, 1].set_xlabel('Rating')
        axes[0, 1].legend(facecolor=SURFACE, edgecolor=BORDER,
                           labelcolor=WHT, fontsize=8)
        axes[0, 1].tick_params(axis='y', labelsize=7)

        # 3. Pos vs neg
        grouped = df.groupby(['food_item', 'sentiment']).size().unstack(fill_value=0)
        grouped.plot(kind='bar', ax=axes[1, 0],
                     color=[RED, GRN], edgecolor="none", width=0.7)
        axes[1, 0].set_title('Positive vs Negative per Item', fontsize=11, pad=12)
        axes[1, 0].tick_params(axis='x', rotation=45, labelsize=7)
        axes[1, 0].legend(facecolor=SURFACE, edgecolor=BORDER,
                           labelcolor=WHT, fontsize=8)

        # 4. Rating distribution
        df['rating'].value_counts().sort_index().plot(
            kind='bar', ax=axes[1, 1], color=ACC, edgecolor="none", width=0.65)
        axes[1, 1].set_title('Rating Distribution', fontsize=11, pad=12)
        axes[1, 1].set_xlabel('Rating')
        axes[1, 1].tick_params(axis='x', rotation=0)

        canvas = FigureCanvasTkAgg(fig, master=self.content)
        canvas.draw()
        canvas.get_tk_widget().pack(padx=30, pady=(0, 20),
                                    fill="both", expand=True)

    # ══════════════════════════════════════════════════════════════
    #  PAGE: ANALYZE DATA
    # ══════════════════════════════════════════════════════════════
    def pg_analyze(self):
        self._clear()
        self._page_header("🔍  Analyze Data",
                          "Sentiment analysis by food item or free-text review")

        # ── Summary cards ──
        _section_label(self.content, "Overall Summary")
        stats_row = tk.Frame(self.content, bg=BG)
        stats_row.pack(fill="x", padx=30, pady=(0, 16))

        total = len(df)
        pos   = len(df[df.sentiment == 'positive'])
        neg   = total - pos
        avg   = round(df.rating.mean(), 2)

        for val, label, col in [
            (total,                              "Total Reviews",   WHT),
            (f"{avg} ★",                         "Avg Rating",      GRN if avg >= 3 else RED),
            (f"{pos}  ({round(pos/total*100,1)}%)", "Positive",      GRN),
            (f"{neg}  ({round(neg/total*100,1)}%)", "Negative",      RED),
        ]:
            _stat_card(stats_row, val, label, col)

        _divider(self.content)

        # ── By food item ──
        _section_label(self.content, "Analyze by Food Item")
        row = tk.Frame(self.content, bg=BG)
        row.pack(fill="x", padx=30, pady=(0, 8))
        _lbl(row, "Select item:", FONTS["body"], GRY, BG).pack(side="left")
        food_var = tk.StringVar(value=FOODS[0])
        _combo(row, food_var, FOODS, width=24).pack(side="left", padx=10)

        food_result = _card(self.content)
        food_result.pack(fill="x", padx=30, pady=(0, 16))

        def show_food(*_):
            for w in food_result.winfo_children():
                w.destroy()
            fd = df[df.food_item == food_var.get()]
            if fd.empty:
                _lbl(food_result, "No data available.",
                     FONTS["body"], GRY, PANEL).pack(pady=16)
                return

            ft  = len(fd)
            fp  = len(fd[fd.sentiment == 'positive'])
            fn  = ft - fp
            fa  = round(fd.rating.mean(), 2)
            fnp = round(fn / ft * 100, 1)

            sr = tk.Frame(food_result, bg=PANEL)
            sr.pack(fill="x", padx=20, pady=14)

            for v, l, c in [(ft, "Reviews", WHT),
                            (f"{fa} ★", "Avg Rating", GRN if fa >= 3 else RED),
                            (fp, "Positive", GRN),
                            (fn, "Negative", RED),
                            (f"{fnp}%", "Neg Rate", RED if fnp > 40 else GOLD)]:
                _stat_card(sr, v, l, c)

            # Rating bar breakdown
            bf = tk.Frame(food_result, bg=PANEL)
            bf.pack(fill="x", padx=20, pady=(0, 8))
            _lbl(bf, "Rating Breakdown", FONTS["small"], GRY, PANEL).pack(
                anchor="w", padx=4, pady=(6, 4))
            rc = fd.rating.value_counts().sort_index()
            for s in range(5, 0, -1):
                cnt = rc.get(s, 0)
                bw  = int((cnt / ft) * 180) if ft else 0
                rw  = tk.Frame(bf, bg=PANEL)
                rw.pack(fill="x", padx=4, pady=2)
                _lbl(rw, "★" * s, FONTS["small"], GOLD, PANEL,
                     width=8, anchor="w").pack(side="left")
                bar_bg = GRN if s >= 3 else RED
                tk.Frame(rw, bg=bar_bg, width=max(bw, 2),
                         height=12).pack(side="left", padx=(0, 6))
                _lbl(rw, str(cnt), FONTS["small"], GRY, PANEL).pack(side="left")

            # Recommendation badge
            if fnp > 50:
                rec, rc2 = f"⚠️  HIGH RISK — {food_var.get()} needs urgent improvement", RED
            elif fnp > 25:
                rec, rc2 = f"⚡  MODERATE — Some negative feedback on {food_var.get()}", GOLD
            else:
                rec, rc2 = f"✅  GOOD — {food_var.get()} is well received by customers!", GRN

            badge = tk.Frame(food_result, bg=SURFACE)
            badge.pack(fill="x", padx=20, pady=(4, 14))
            tk.Frame(badge, bg=rc2, width=4).pack(side="left", fill="y")
            _lbl(badge, rec, FONTS["body"], rc2, SURFACE,
                 wraplength=680, justify="left").pack(
                     side="left", padx=14, pady=10)

        food_var.trace("w", show_food)
        show_food()

        _divider(self.content)

        # ── Free-text analysis ──
        _section_label(self.content, "Analyze Any Review Text")
        text_card = _card(self.content)
        text_card.pack(fill="x", padx=30, pady=(0, 8))

        ti = tk.Frame(text_card, bg=PANEL)
        ti.pack(fill="x", padx=24, pady=16)
        _lbl(ti, "Paste or type a customer review below:",
             FONTS["small"], GRY, PANEL).pack(anchor="w", pady=(0, 6))
        review_box = tk.Text(ti, font=FONTS["body"],
                             bg=SURFACE, fg=WHT, insertbackground=ACC,
                             relief="flat", bd=0, height=4, wrap="word")
        review_box.pack(fill="x", ipady=6)

        result_card = _card(self.content)
        result_card.pack(fill="x", padx=30, pady=(4, 24))

        def run_analysis():
            text = review_box.get("1.0", "end").strip()
            if not text:
                messagebox.showwarning("Empty", "Please enter review text.")
                return
            for w in result_card.winfo_children():
                w.destroy()

            cl    = clean(text)
            vec   = tfidf.transform([cl])
            pred  = model.predict(vec)[0]
            prob  = model.predict_proba(vec)[0]
            vd    = vader_pred(text)
            sent  = "positive" if pred == 1 else "negative"
            conf  = max(prob) * 100
            clr   = GRN if sent == "positive" else RED

            ri = tk.Frame(result_card, bg=PANEL)
            ri.pack(fill="x", padx=20, pady=14)

            emoji = "😊" if sent == "positive" else "😞"
            _lbl(ri, f"{emoji}  {sent.upper()}",
                 ("Georgia", 18, "bold"), clr, PANEL).pack(side="left")
            _lbl(ri, f"  Confidence: {conf:.1f}%",
                 FONTS["body"], GRY, PANEL).pack(side="left", padx=14)

            det = tk.Frame(result_card, bg=PANEL)
            det.pack(fill="x", padx=20, pady=(0, 4))

            vd_col = (GRN if vd == "positive"
                      else RED if vd == "negative" else BLU)
            for label, value, col in [
                ("ML Model Result",  sent.upper(),                           clr),
                ("VADER NLP Result", vd.upper(),                             vd_col),
                ("Cleaned Tokens",   (cl[:90] + "…") if len(cl) > 90 else cl, GRY),
            ]:
                row = tk.Frame(det, bg=SURFACE)
                row.pack(fill="x", pady=2)
                _lbl(row, label, FONTS["small"], GRY, SURFACE,
                     width=22, anchor="w").pack(side="left", padx=12, pady=10)
                _lbl(row, value, FONTS["body"], col, SURFACE,
                     wraplength=500, justify="left").pack(side="left", padx=8)

            tip  = ("✅  Positive feedback — no action needed."
                    if sent == "positive"
                    else "⚠️  Negative feedback — owner should investigate.")
            _lbl(result_card, tip, ("Segoe UI", 10, "italic"), clr, PANEL,
                 wraplength=680).pack(anchor="w", padx=20, pady=(4, 16))

        _btn(ti, "RUN ANALYSIS", run_analysis).pack(
            fill="x", ipady=11, pady=(10, 0))

    # ══════════════════════════════════════════════════════════════
    #  PAGE: ALL REVIEWS
    # ══════════════════════════════════════════════════════════════
    def pg_reviews(self):
        self._clear()
        self._page_header("📋  All Reviews",
                          f"Total: {len(df)} reviews  ·  Live database")

        # Filter bar
        fb = tk.Frame(self.content, bg=SURFACE)
        fb.pack(fill="x", padx=30, pady=(0, 12))
        fi = tk.Frame(fb, bg=SURFACE)
        fi.pack(padx=16, pady=10, anchor="w")

        _lbl(fi, "Food:", FONTS["small"], GRY, SURFACE).pack(side="left")
        food_var = tk.StringVar(value="All")
        _combo(fi, food_var, ["All"] + FOODS, width=18).pack(side="left", padx=(8, 20))

        _lbl(fi, "Sentiment:", FONTS["small"], GRY, SURFACE).pack(side="left")
        sent_var = tk.StringVar(value="All")
        _combo(fi, sent_var, ["All", "positive", "negative"], width=12).pack(
            side="left", padx=8)

        # Treeview
        tree_frame = tk.Frame(self.content, bg=BG)
        tree_frame.pack(padx=30, pady=5, fill="both", expand=True)

        cols = ("customer_id", "food_item", "rating", "sentiment", "review")
        tree = ttk.Treeview(tree_frame, columns=cols,
                            show="headings", style="BK.Treeview")

        for col, hd, w, anc in [
            ("customer_id", "ID",        75,  "center"),
            ("food_item",   "Food Item", 140,  "w"),
            ("rating",      "Rating",    80,   "center"),
            ("sentiment",   "Sentiment", 100,  "center"),
            ("review",      "Review",    500,  "w"),
        ]:
            tree.heading(col, text=hd)
            tree.column(col, width=w, anchor=anc)

        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        def load_table():
            for i in tree.get_children():
                tree.delete(i)
            d = df.copy()
            if food_var.get() != "All":
                d = d[d.food_item == food_var.get()]
            if sent_var.get() != "All":
                d = d[d.sentiment == sent_var.get()]
            for _, row in d.iterrows():
                stars = "★" * int(row.rating) + "☆" * (5 - int(row.rating))
                tree.insert("", "end",
                            values=(row.customer_id, row.food_item,
                                    stars, row.sentiment, row.review),
                            tags=("pos" if row.sentiment == "positive" else "neg",))
            tree.tag_configure("pos", foreground=GRN)
            tree.tag_configure("neg", foreground=RED)

        food_var.bind("<<ComboboxSelected>>", lambda e: load_table())
        sent_var.trace("w", lambda *a: load_table())
        load_table()

    # ══════════════════════════════════════════════════════════════
    #  PAGE: FOOD STATS
    # ══════════════════════════════════════════════════════════════
    def pg_stats(self):
        self._clear()
        self._page_header("📈  Food Item Statistics",
                          "Detailed performance breakdown per menu item")

        top = tk.Frame(self.content, bg=BG)
        top.pack(fill="x", padx=30, pady=(0, 16))
        _lbl(top, "Select item:", FONTS["body"], GRY, BG).pack(side="left")
        stat_var = tk.StringVar(value=FOODS[0])
        _combo(top, stat_var, FOODS, width=24).pack(side="left", padx=10)

        stats_frame = tk.Frame(self.content, bg=BG)
        stats_frame.pack(fill="both", expand=True, padx=30)

        def show_stats(*_):
            for w in stats_frame.winfo_children():
                w.destroy()
            d = df[df.food_item == stat_var.get()]
            if d.empty:
                _lbl(stats_frame, "No data found for this item.",
                     FONTS["body"], GRY, BG).pack(pady=20)
                return

            total = len(d)
            pos   = len(d[d.sentiment == 'positive'])
            neg   = total - pos
            nr    = round(neg / total * 100, 1)
            ar    = round(d.rating.mean(), 2)

            # Stat cards
            cr = tk.Frame(stats_frame, bg=BG)
            cr.pack(fill="x", pady=(0, 16))
            for v, l, c in [(total, "Total Reviews", WHT),
                            (pos, "Positive", GRN),
                            (neg, "Negative", RED),
                            (f"{nr}%", "Negative Rate", RED if nr > 40 else GOLD),
                            (f"{ar}/5", "Avg Rating", GRN if ar >= 3 else RED)]:
                _stat_card(cr, v, l, c)

            # Review list
            _lbl(stats_frame, "Reviews", FONTS["heading"], WHT, BG).pack(
                anchor="w", pady=(0, 6))
            rev_frame = tk.Frame(stats_frame, bg=BG)
            rev_frame.pack(fill="both", expand=True)

            vsb = tk.Scrollbar(rev_frame)
            vsb.pack(side="right", fill="y")
            lb = tk.Listbox(rev_frame, font=FONTS["body"],
                            bg=PANEL, fg=WHT, relief="flat", bd=0,
                            selectbackground=SEL,
                            yscrollcommand=vsb.set,
                            activestyle="none")
            lb.pack(fill="both", expand=True)
            vsb.config(command=lb.yview)

            for _, row in d.iterrows():
                icon = "✅" if row.sentiment == "positive" else "❌"
                lb.insert("end",
                    f"  {icon}  [{row.customer_id}]  ★{row.rating}  —  {row.review}")

        stat_var.trace("w", show_stats)
        show_stats()

    # ══════════════════════════════════════════════════════════════
    #  PAGE: ML MODEL INFO
    # ══════════════════════════════════════════════════════════════
    def pg_model(self):
        self._clear()
        self._page_header("⚙️  ML Model Information",
                          "Logistic Regression + TF-IDF + VADER sentiment pipeline")

        y  = df.sentiment.map({'positive': 1, 'negative': 0})
        Xt = tfidf.transform(df.cleaned)
        st = y if (y.nunique() > 1 and len(y) >= 10) else None
        Xr, Xe, yr, ye = train_test_split(Xt, y, test_size=0.2,
                                           random_state=42, stratify=st)
        yp  = model.predict(Xe)
        rep = classification_report(ye, yp,
                                    target_names=['Negative', 'Positive'])
        ac  = accuracy_score(ye, yp)

        # Metric cards
        metric_row = tk.Frame(self.content, bg=BG)
        metric_row.pack(fill="x", padx=30, pady=(0, 16))
        for v, l, c in [
            (f"{ac*100:.1f}%", "Model Accuracy", GRN),
            (len(df),          "Total Reviews",  WHT),
            (Xr.shape[0],      "Train Samples",  BLU),
            (Xe.shape[0],      "Test Samples",   GOLD),
        ]:
            _stat_card(metric_row, v, l, c)

        # Classification report
        _section_label(self.content, "Classification Report")
        report_card = _card(self.content)
        report_card.pack(fill="x", padx=30, pady=(0, 16))
        rep_box = tk.Text(report_card, font=FONTS["mono"],
                          bg=SURFACE, fg=GRN, relief="flat",
                          bd=0, height=11, state="normal")
        rep_box.pack(padx=20, pady=16, fill="x")
        rep_box.insert("1.0", rep)
        rep_box.config(state="disabled")

        # Confusion matrix
        _section_label(self.content, "Confusion Matrix")
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(SURFACE)

        cm = confusion_matrix(ye, yp)
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', ax=ax,
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'],
                    linewidths=1, linecolor=BG,
                    annot_kws={"size": 13, "color": WHT})
        ax.set_ylabel('Actual', color=GRY)
        ax.set_xlabel('Predicted', color=GRY)
        ax.tick_params(colors=GRY)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)
        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.content)
        canvas.draw()
        canvas.get_tk_widget().pack(padx=30, pady=(0, 24))


# ══════════════════════════════════════════════════════════════════
#  ENTRY POINTS
# ══════════════════════════════════════════════════════════════════
def start_login():
    root = tk.Tk()
    LoginWindow(root)
    root.mainloop()

def launch_main(role, username):
    root = tk.Tk()
    App(root, role, username)
    root.mainloop()

if __name__ == "__main__":
    start_login()
