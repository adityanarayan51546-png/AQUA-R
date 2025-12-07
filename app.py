import cv2
import time
import os
import sys
import numpy as np
import random
import math
import socket
from datetime import datetime
from flask import Flask, render_template_string, Response, request, redirect, url_for, session, jsonify
from ultralytics import YOLO

# ==========================================
# 1. إعدادات السيرفر واللغات
# ==========================================
app = Flask(__name__)
app.secret_key = 'aqua_r_super_secret_key'

MODEL_WATER_PATH = "water_hyacinth.pt"
MODEL_RUBBISH_PATH = "rubbish.pt"

# بيانات الروبوت
robot_status = {
    "battery": 92,
    "status": "Patrolling",
    "lat": 30.0444, 
    "lng": 31.2357,
    "location_name": "Nile River, Sector 4",
    "trash_count": 12,
    "plant_count": 5
}

start_lat = 30.0444
start_lng = 31.2357

# --- قاموس الترجمة (Translation Dictionary) ---
TRANSLATIONS = {
    'en': {
        'dir': 'ltr', 'font': 'Roboto',
        'title': 'AQUA-R', 'subtitle': 'Autonomous River Cleaning System',
        'login_guest': 'Enter as Guest', 'or': 'OR', 'login_account': 'Login with Account',
        'username': 'Username', 'password': 'Password', 'secure_login': 'Secure Login',
        'google_login': 'Continue with Google', 'apple_login': 'Continue with Apple',
        'dashboard': 'Dashboard', 'store': 'Store', 'support': 'Support', 'logout': 'Exit',
        'welcome': 'WELCOME', 'live_vision': 'Real-Time Vision', 'live': 'LIVE',
        'system_status': 'System Status', 'battery': 'BATTERY', 'status': 'STATUS', 'trash': 'TRASH',
        'summon': '📍 Summon Robot (GPS)', 'stop': '🛑 Emergency Stop', 'gps_track': '🗺️ Live GPS Tracking',
        'chat_header': 'AQUA-BOT Assistant', 'chat_welcome': 'Hello! How can I help you?',
        'buy_now': 'Buy Now', 'checkout': 'Checkout', 'payment': 'Payment Details',
        'card_number': 'Card Number', 'expiry': 'Expiry Date (MM/YY)', 'cvv': 'CVV', 'pay_btn': 'Confirm Payment',
        'success_msg': 'Thank you for your trust! The robot is on its way.',
        'invalid_card': 'Error: Card is expired or invalid.',
        'shipping': 'Shipping Address', 'full_name': 'Full Name', 'address': 'Address',
        'contact_dev': 'Contact Developers', 'send_ticket': 'Send a Ticket',
        'subject': 'Subject', 'message': 'Message', 'describe': 'Describe the issue...',
        'submit': 'Submit Request', 'chat_placeholder': 'Ask something...'
    },
    'ar': {
        'dir': 'rtl', 'font': 'Cairo',
        'title': 'أكوا-آر', 'subtitle': 'نظام تنظيف الأنهار الذكي',
        'login_guest': 'دخول كزائر', 'or': 'أو', 'login_account': 'تسجيل الدخول بحساب',
        'username': 'اسم المستخدم', 'password': 'كلمة المرور', 'secure_login': 'دخول آمن',
        'google_login': 'المتابعة باستخدام Google', 'apple_login': 'المتابعة باستخدام Apple',
        'dashboard': 'لوحة التحكم', 'store': 'المتجر', 'support': 'الدعم الفني', 'logout': 'خروج',
        'welcome': 'مرحباً بك', 'live_vision': 'البث المباشر', 'live': 'مباشر',
        'system_status': 'حالة النظام', 'battery': 'البطارية', 'status': 'الحالة', 'trash': 'النفايات',
        'summon': '📍 استدعاء الروبوت (GPS)', 'stop': '🛑 إيقاف طوارئ', 'gps_track': '🗺️ تتبع الموقع الحى',
        'chat_header': 'مساعد أكوا الذكي', 'chat_welcome': 'أهلاً بك! كيف يمكنني مساعدتك؟',
        'buy_now': 'شراء الآن', 'checkout': 'إتمام الشراء', 'payment': 'بيانات الدفع',
        'card_number': 'رقم البطاقة', 'expiry': 'تاريخ الانتهاء (سنة/شهر)', 'cvv': 'رمز الأمان', 'pay_btn': 'تأكيد الدفع',
        'success_msg': 'شكراً لثقتك بنا! تم استلام طلبك والروبوت في الطريق إليك.',
        'invalid_card': 'خطأ: البطاقة منتهية الصلاحية أو غير صالحة.',
        'shipping': 'عنوان الشحن', 'full_name': 'الاسم الكامل', 'address': 'العنوان',
        'contact_dev': 'تواصل مع المطورين', 'send_ticket': 'أرسل تذكرة دعم',
        'subject': 'الموضوع', 'message': 'الرسالة', 'describe': 'اشرح المشكلة بالتفصيل...',
        'submit': 'إرسال الطلب', 'chat_placeholder': 'اسأل شيئاً...'
    },
    'fr': {
        'dir': 'ltr', 'font': 'Roboto',
        'title': 'AQUA-R', 'subtitle': 'Système de Nettoyage Autonome',
        'login_guest': 'Entrer comme Invité', 'or': 'OU', 'login_account': 'Connexion Compte',
        'username': 'Nom d\'utilisateur', 'password': 'Mot de passe', 'secure_login': 'Connexion Sécurisée',
        'google_login': 'Continuer avec Google', 'apple_login': 'Continuer avec Apple',
        'dashboard': 'Tableau de bord', 'store': 'Boutique', 'support': 'Support', 'logout': 'Sortie',
        'welcome': 'BIENVENUE', 'live_vision': 'Vision en Temps Réel', 'live': 'EN DIRECT',
        'system_status': 'État du Système', 'battery': 'BATTERIE', 'status': 'STATUT', 'trash': 'DÉCHETS',
        'summon': '📍 Appeler le Robot', 'stop': '🛑 Arrêt d\'urgence', 'gps_track': '🗺️ Suivi GPS',
        'chat_header': 'Assistant AQUA-BOT', 'chat_welcome': 'Bonjour! Comment puis-je vous aider?',
        'buy_now': 'Acheter', 'checkout': 'Caisse', 'payment': 'Détails de Paiement',
        'card_number': 'Numéro de Carte', 'expiry': 'Date d\'expiration (MM/YY)', 'cvv': 'CVV', 'pay_btn': 'Confirmer',
        'success_msg': 'Merci de votre confiance! Le robot est en route.',
        'invalid_card': 'Erreur: Carte expirée ou invalide.',
        'shipping': 'Adresse de Livraison', 'full_name': 'Nom Complet', 'address': 'Adresse',
        'contact_dev': 'Contacter les Développeurs', 'send_ticket': 'Envoyer un Billet',
        'subject': 'Sujet', 'message': 'Message', 'describe': 'Décrivez le problème...',
        'submit': 'Envoyer la Demande', 'chat_placeholder': 'Demandez quelque chose...'
    }
}

# دالة مساعدة للحصول على النصوص حسب اللغة
def get_text(key):
    lang = session.get('lang', 'en')
    return TRANSLATIONS[lang].get(key, key)

def get_dir():
    lang = session.get('lang', 'en')
    return TRANSLATIONS[lang]['dir']

def get_font():
    lang = session.get('lang', 'en')
    return TRANSLATIONS[lang]['font']

# ==========================================
# 2. إعداد الذكاء الاصطناعي (كما هو)
# ==========================================
print("Loading AQUA-R AI Core...")
try:
    model_water = YOLO(MODEL_WATER_PATH)
    model_rubbish = YOLO(MODEL_RUBBISH_PATH)
    print("✅ AQUA-R Vision System: ONLINE")
except Exception as e:
    print(f"⚠️ AI Warning: {e}")
    model_water = None
    model_rubbish = None

def collect_results_as_dict(results, source_label):
    out=[]
    if results is None or getattr(results,"boxes",None) is None: return out
    for b in results.boxes:
        xy = b.xyxy[0].cpu().numpy().tolist()
        conf = float(b.conf[0])
        cls = int(b.cls[0])
        name = results.names[cls] if hasattr(results,"names") and results.names else str(cls)
        out.append({"bbox":xy, "conf":conf, "cls":cls, "name":name, "source":source_label})
    return out

def smart_ai_process(frame):
    if model_water is None or model_rubbish is None: return frame
    small_frame = cv2.resize(frame, (640, 480))
    res_w = model_water(small_frame, conf=0.25, verbose=False)[0]
    res_r = model_rubbish(small_frame, conf=0.25, verbose=False)[0]
    dets_w = collect_results_as_dict(res_w, "water")
    dets_r = collect_results_as_dict(res_r, "rubbish")
    all_cands = dets_w + dets_r
    
    w_count = sum(1 for d in all_cands if d['source'] == 'water')
    r_count = sum(1 for d in all_cands if d['source'] == 'rubbish')
    
    if r_count > 0: 
        robot_status['status'] = "Cleaning Trash..."
        robot_status['trash_count'] += 1
    elif w_count > 0: 
        robot_status['status'] = "Processing Plants..."
    elif robot_status['status'] not in ["Summoned", "Moving to Owner..."]: 
        robot_status['status'] = "Patrolling"

    h, w, _ = frame.shape
    scale_x, scale_y = w/640, h/480
    for d in all_cands:
        x1, y1, x2, y2 = d['bbox']
        x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
        y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
        color = (0, 255, 127) if d['source'] == 'water' else (0, 69, 255) 
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 3)
        label = f"{d['name']} {d['conf']*100:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1-25), (x1+tw+10, y1), color, -1)
        cv2.putText(frame, label, (x1+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    return frame

def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): cap = cv2.VideoCapture("download (2).jpg")
    while True:
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frame = smart_ai_process(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ==========================================
# 3. تصميم الموقع (Templates)
# ==========================================

STYLE_GLOBAL = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&family=Orbitron:wght@400;700&family=Roboto:wght@300;400;700&display=swap');
    
    :root { --primary: #00f2ff; --secondary: #0078ff; --bg: #050a14; --card-bg: rgba(20, 30, 50, 0.9); --text: #e0f7fa; }
    body { font-family: 'Roboto', sans-serif; background: radial-gradient(circle, #0a1525 0%, #000000 100%); color: var(--text); margin: 0; overflow-x: hidden; }
    
    .navbar { background: rgba(0,0,0,0.9); padding: 15px 30px; display: flex; justify-content: space-between; align-items: center; position: fixed; top: 0; width: 100%; z-index: 1000; border-bottom: 1px solid var(--primary); box-sizing: border-box; }
    .logo { font-family: 'Orbitron', sans-serif; font-size: 24px; color: var(--primary); display: flex; align-items: center; gap: 10px; }
    .nav-links a { color: white; text-decoration: none; margin: 0 15px; font-weight: bold; transition: 0.3s; }
    .nav-links a:hover { color: var(--primary); }
    
    .lang-btn { background: transparent; border: 1px solid var(--primary); color: var(--primary); padding: 5px 10px; cursor: pointer; border-radius: 5px; margin-left: 5px; }
    .lang-btn:hover { background: var(--primary); color: black; }

    .main-content { padding: 100px 20px; max-width: 1400px; margin: auto; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
    .card { background: var(--card-bg); border: 1px solid rgba(0, 242, 255, 0.2); border-radius: 15px; padding: 20px; position: relative; }
    
    .btn { background: linear-gradient(45deg, var(--secondary), var(--primary)); color: black; padding: 12px 25px; border: none; border-radius: 30px; cursor: pointer; font-weight: bold; width: 100%; margin-top: 10px; }
    .btn:hover { transform: scale(1.02); box-shadow: 0 0 15px var(--primary); }
    
    .btn-social { width: 100%; margin-top: 10px; padding: 12px; border-radius: 5px; border: 1px solid #444; background: #151515; color: white; display: flex; align-items: center; justify-content: center; gap: 10px; cursor: pointer; text-decoration: none; }
    .btn-social:hover { background: #252525; }
    .btn-social img { width: 20px; }

    input, textarea, select { width: 100%; padding: 12px; margin: 8px 0; background: rgba(0,0,0,0.5); border: 1px solid #333; color: white; border-radius: 5px; box-sizing: border-box; }
    
    /* RTL Support */
    body[dir="rtl"] { font-family: 'Cairo', sans-serif; }
    body[dir="rtl"] .logo { font-family: 'Cairo', sans-serif; }

    /* Chatbot Styles */
    .chatbot-btn {
        position: fixed; bottom: 30px; right: 30px;
        width: 70px; height: 70px;
        background: var(--primary); border-radius: 50%;
        display: flex; justify-content: center; align-items: center;
        cursor: pointer; z-index: 2000;
        box-shadow: 0 0 20px var(--primary);
        animation: pulse 2s infinite;
        overflow: hidden;
    }
    .chatbot-btn img { width: 80%; height: 80%; object-fit: contain; }
    
    .chat-window {
        position: fixed; bottom: 110px; right: 30px;
        width: 350px; height: 450px;
        background: #111; border: 1px solid var(--primary);
        border-radius: 15px; z-index: 2000;
        display: none; flex-direction: column;
        overflow: hidden;
        box-shadow: 0 0 30px rgba(0,0,0,0.8);
    }
    .chat-header { background: linear-gradient(90deg, var(--secondary), var(--primary)); padding: 15px; font-weight: bold; color: black; }
    .chat-body { flex: 1; padding: 15px; overflow-y: auto; font-size: 14px; background: rgba(255,255,255,0.05); }
    .chat-msg { margin-bottom: 10px; padding: 10px 15px; border-radius: 15px; max-width: 80%; line-height: 1.4; }
    .bot-msg { background: #222; color: var(--primary); align-self: flex-start; border-bottom-left-radius: 2px; border: 1px solid #333; }
    .user-msg { background: var(--secondary); color: white; align-self: flex-end; margin-left: auto; border-bottom-right-radius: 2px; }
    
    @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(0, 242, 255, 0.7); } 70% { box-shadow: 0 0 0 15px rgba(0, 242, 255, 0); } 100% { box-shadow: 0 0 0 0 rgba(0, 242, 255, 0); } }
</style>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
"""

HTML_LOGIN = """
<!DOCTYPE html>
<html dir="{{ t('dir') }}">
<head><title>{{ t('title') }}</title>""" + STYLE_GLOBAL + """</head>
<body dir="{{ t('dir') }}">
    <div style="position:absolute; top:20px; right:20px; z-index:2000;">
        <a href="/set_lang/en" class="lang-btn">EN</a>
        <a href="/set_lang/ar" class="lang-btn">عربي</a>
        <a href="/set_lang/fr" class="lang-btn">FR</a>
    </div>
    <div style="height: 100vh; display: flex; justify-content: center; align-items: center; background: url('https://source.unsplash.com/1600x900/?water,tech') center/cover;">
        <div style="background: rgba(10, 20, 30, 0.95); padding: 40px; border-radius: 20px; border: 1px solid var(--primary); width: 400px; text-align: center; box-shadow: 0 0 50px rgba(0, 242, 255, 0.2);">
            <img src="/static/logo.png" width="80" style="filter: drop-shadow(0 0 10px var(--primary));">
            <h1>{{ t('title') }}</h1>
            <p style="color:#888;">{{ t('subtitle') }}</p>
            
            <div id="main-login">
                <form action="/login_guest" method="POST">
                    <input type="text" name="guest_name" placeholder="{{ t('login_guest') }}..." required>
                    <button class="btn">{{ t('login_guest') }}</button>
                </form>
                <p>{{ t('or') }}</p>
                <button onclick="document.getElementById('main-login').style.display='none'; document.getElementById('social-login').style.display='block';" style="background:none; border:none; color:var(--primary); cursor:pointer; text-decoration:underline;">
                    {{ t('login_account') }}
                </button>
            </div>

            <div id="social-login" style="display:none;">
                <a href="/login_google_sim" class="btn-social">
                    <img src="/static/google.png"> {{ t('google_login') }}
                </a>
                <a href="/login_apple_sim" class="btn-social">
                    <img src="/static/apple.png" style="filter: invert(1);"> {{ t('apple_login') }}
                </a>
                <p style="margin-top:15px; border-top:1px solid #333; padding-top:10px; font-size:12px; color:#666;">Or use Standard Login</p>
                <form action="/login_admin" method="POST">
                    <input type="text" name="username" placeholder="{{ t('username') }}">
                    <input type="password" name="password" placeholder="{{ t('password') }}">
                    <button class="btn" style="background:#333;">{{ t('secure_login') }}</button>
                </form>
                <br>
                <button onclick="document.getElementById('social-login').style.display='none'; document.getElementById('main-login').style.display='block';" style="background:none; border:none; color:#888; cursor:pointer;">Back</button>
            </div>
        </div>
    </div>
</body>
</html>
"""

HTML_GOOGLE_LOGIN = """
<!DOCTYPE html>
<html>
<head><title>Sign in - Google Accounts</title>
<style>
    body { background: #fff; font-family: 'Roboto', arial, sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
    .box { border: 1px solid #dadce0; border-radius: 8px; padding: 40px; width: 350px; text-align: center; }
    .input-field { width: 100%; padding: 13px 15px; margin: 10px 0; border: 1px solid #dadce0; border-radius: 4px; box-sizing: border-box; font-size: 16px; }
    .btn { background: #1a73e8; color: white; font-weight: bold; padding: 10px 24px; border-radius: 4px; border: none; cursor: pointer; float: right; margin-top: 20px; }
</style>
</head>
<body>
    <div class="box">
        <img src="https://upload.wikimedia.org/wikipedia/commons/2/2f/Google_2015_logo.svg" width="75" style="margin-bottom:10px;">
        <h2 style="font-weight:400; margin: 0;">Sign in</h2>
        <p style="font-size:16px;">to continue to AQUA-R</p>
        <form action="/auth_google" method="POST">
            <input type="email" name="email" class="input-field" placeholder="Email or phone" required>
            <p style="text-align:left; color:#1a73e8; font-size:14px; font-weight:bold; cursor:pointer;">Forgot email?</p>
            <div style="text-align:left; margin-top:40px;">
                <span style="color:#1a73e8; font-weight:bold; cursor:pointer;">Create account</span>
                <button class="btn">Next</button>
            </div>
        </form>
    </div>
</body>
</html>
"""

HTML_APPLE_LOGIN = """
<!DOCTYPE html>
<html>
<head><title>Sign in with Apple</title>
<style>
    body { background: #333; font-family: -apple-system, BlinkMacSystemFont, sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; color:white; }
    .box { text-align: center; width: 300px; }
    .input-field { width: 100%; padding: 15px; margin: 5px 0; border: 1px solid #555; border-radius: 10px; background: #000; color: white; font-size: 16px; box-sizing: border-box;}
    .btn { background: white; color: black; font-size: 18px; width: 100%; padding: 10px; border-radius: 10px; border: none; cursor: pointer; margin-top: 20px; }
</style>
</head>
<body>
    <div class="box">
        <img src="https://cdn-icons-png.flaticon.com/512/0/747.png" width="50" style="filter: invert(1); margin-bottom: 20px;">
        <h2 style="margin-bottom: 30px;">Sign in with Apple ID</h2>
        <form action="/auth_apple" method="POST">
            <input type="email" name="email" class="input-field" placeholder="Apple ID" required>
            <input type="password" name="password" class="input-field" placeholder="Password" required>
            <button class="btn">➔</button>
        </form>
    </div>
</body>
</html>
"""

HTML_DASHBOARD = """
<!DOCTYPE html>
<html dir="{{ t('dir') }}">
<head><title>{{ t('dashboard') }}</title>""" + STYLE_GLOBAL + """
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
</head>
<body dir="{{ t('dir') }}">
    <div class="navbar">
        <div class="logo"><img src="/static/logo.png" width="30"> {{ t('title') }}</div>
        <div class="nav-links">
            <a href="/dashboard" style="color:var(--primary)">{{ t('dashboard') }}</a>
            <a href="/store">{{ t('store') }}</a>
            <a href="/support">{{ t('support') }}</a>
            <a href="/logout" style="color:#ff4444;">{{ t('logout') }}</a>
        </div>
        <div>
            <a href="/set_lang/en" class="lang-btn">EN</a>
            <a href="/set_lang/ar" class="lang-btn">AR</a>
            <a href="/set_lang/fr" class="lang-btn">FR</a>
        </div>
    </div>

    <div class="main-content">
        <h1>{{ t('welcome') }}, <span style="color:var(--primary)">{{ user }}</span></h1>
        
        <div class="grid">
            <div class="card" style="grid-column: span 2;">
                <h2>📡 {{ t('live_vision') }} <span style="background:red; font-size:12px; padding:2px 5px; border-radius:3px;">{{ t('live') }}</span></h2>
                <div style="background:black; height:350px; border-radius:10px; overflow:hidden; display:flex; justify-content:center;">
                    <img src="{{ url_for('video_feed') }}" style="height:100%;">
                </div>
            </div>

            <div class="card">
                <h2>⚙️ {{ t('system_status') }}</h2>
                <p>🔋 {{ t('battery') }}: <strong id="batt" style="color:var(--primary)">--%</strong></p>
                <div style="background:#333; height:8px; border-radius:4px;"><div id="batt-bar" style="width:0%; height:100%; background:var(--primary);"></div></div>
                <p>📡 {{ t('status') }}: <strong id="stat">--</strong></p>
                <p>🗑️ {{ t('trash') }}: <strong id="trash">0</strong></p>
                <button onclick="fetch('/api/summon', {method:'POST'}).then(r=>alert('Robot Summoned!'))" class="btn">{{ t('summon') }}</button>
                <button class="btn" style="background:#d00;">{{ t('stop') }}</button>
            </div>

            <div class="card" style="grid-column: span 3; height:400px;">
                <h2>{{ t('gps_track') }}</h2>
                <div id="map" style="height:320px; border-radius:10px;"></div>
            </div>
        </div>
    </div>

    <!-- Chatbot Widget with Robot Icon -->
    <div class="chatbot-btn" onclick="toggleChat()">
        <!-- استخدمنا أيقونة روبوت ملونة هنا ليكون شكلها جذاب -->
        <img src="https://cdn-icons-png.flaticon.com/512/4712/4712009.png" alt="Chat">
    </div>
    
    <div class="chat-window" id="chatWindow">
        <div class="chat-header">
            {{ t('chat_header') }}
            <span style="float:right; cursor:pointer;" onclick="toggleChat()">✖</span>
        </div>
        <div class="chat-body" id="chatBody">
            <div class="chat-msg bot-msg">{{ t('chat_welcome') }}</div>
        </div>
        <div style="padding:10px; background:#1a1a1a; display:flex;">
            <input type="text" id="chatInput" placeholder="{{ t('chat_placeholder') }}" style="margin:0; border-radius:20px 0 0 20px;">
            <button onclick="sendMessage()" style="border-radius:0 20px 20px 0; border:none; background:var(--primary); cursor:pointer; padding:0 15px;">➤</button>
        </div>
    </div>

    <script>
        var map = L.map('map').setView([30.0444, 31.2357], 15);
        L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png').addTo(map);
        var marker = L.marker([30.0444, 31.2357]).addTo(map).bindPopup("AQUA-R Unit").openPopup();

        setInterval(() => {
            fetch('/api/status').then(r => r.json()).then(data => {
                document.getElementById('batt').innerText = data.battery + '%';
                document.getElementById('batt-bar').style.width = data.battery + '%';
                document.getElementById('stat').innerText = data.status;
                document.getElementById('trash').innerText = data.trash_count;
                var ll = new L.LatLng(data.lat, data.lng);
                marker.setLatLng(ll); map.panTo(ll);
            });
        }, 1000);

        // Chatbot Logic
        function toggleChat() {
            var w = document.getElementById('chatWindow');
            w.style.display = (w.style.display === 'flex') ? 'none' : 'flex';
        }
        
        function sendMessage() {
            var input = document.getElementById('chatInput');
            var msg = input.value;
            if(!msg) return;
            
            // Add user message
            var chatBody = document.getElementById('chatBody');
            chatBody.innerHTML += `<div class="chat-msg user-msg">${msg}</div>`;
            input.value = '';
            chatBody.scrollTop = chatBody.scrollHeight;
            
            // Send to Backend
            fetch('/api/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({msg: msg})
            })
            .then(response => response.json())
            .then(data => {
                chatBody.innerHTML += `<div class="chat-msg bot-msg">${data.reply}</div>`;
                chatBody.scrollTop = chatBody.scrollHeight;
            });
        }
        
        // Enter key to send
        document.getElementById("chatInput").addEventListener("keyup", function(event) {
            if (event.key === "Enter") sendMessage();
        });
    </script>
</body>
</html>
"""

HTML_STORE = """
<!DOCTYPE html>
<html dir="{{ t('dir') }}">
<head><title>{{ t('store') }}</title>""" + STYLE_GLOBAL + """</head>
<body dir="{{ t('dir') }}">
    <div class="navbar">
        <div class="logo">🛒 {{ t('store') }}</div>
        <div class="nav-links"><a href="/dashboard">{{ t('dashboard') }}</a></div>
    </div>
    <div class="main-content">
        <div class="grid">
            <div class="card">
                <div style="height:200px; background:#111; display:flex; justify-content:center; align-items:center;">[IMG V1]</div>
                <h2>AQUA-R Scout</h2>
                <h3>$3,999</h3>
                <a href="/checkout?item=AQUA-R Scout&price=3999" class="btn">{{ t('buy_now') }}</a>
            </div>
            <div class="card" style="border-color:var(--primary);">
                <div style="height:200px; background:#111; display:flex; justify-content:center; align-items:center;">[IMG V2]</div>
                <h2>AQUA-R Guardian</h2>
                <h3>$8,500</h3>
                <a href="/checkout?item=AQUA-R Guardian&price=8500" class="btn">{{ t('buy_now') }}</a>
            </div>
            <div class="card">
                <div style="height:200px; background:#111; display:flex; justify-content:center; align-items:center;">[IMG V3]</div>
                <h2>AQUA-R Leviathan</h2>
                <h3>$15,000</h3>
                <a href="/checkout?item=AQUA-R Leviathan&price=15000" class="btn">{{ t('buy_now') }}</a>
            </div>
        </div>
    </div>
</body>
</html>
"""

HTML_SUPPORT = """
<!DOCTYPE html>
<html dir="{{ t('dir') }}">
<head><title>{{ t('support') }}</title>""" + STYLE_GLOBAL + """</head>
<body dir="{{ t('dir') }}">
    <div class="navbar">
        <div class="logo">📞 {{ t('support') }}</div>
        <div class="nav-links"><a href="/dashboard">{{ t('dashboard') }}</a></div>
    </div>
    <div class="main-content">
        <div class="grid">
            <div class="card">
                <h2>{{ t('contact_dev') }}</h2>
                <p>Email: support@aquarobot.com</p>
                <p>Phone: +20 123 456 7890</p>
                <div style="margin-top:20px;">
                    <button class="btn-social">LinkedIn Page</button>
                    <button class="btn-social">GitHub Repository</button>
                    <button class="btn-social">Facebook Community</button>
                </div>
            </div>
            <div class="card">
                <h2>{{ t('send_ticket') }}</h2>
                <form>
                    <input type="text" placeholder="{{ t('username') }}" value="{{ user }}">
                    <input type="text" placeholder="{{ t('subject') }}">
                    <textarea rows="5" placeholder="{{ t('describe') }}"></textarea>
                    <button class="btn" style="width:100%">{{ t('submit') }}</button>
                </form>
            </div>
        </div>
    </div>
</body>
</html>
"""

HTML_CHECKOUT = """
<!DOCTYPE html>
<html dir="{{ t('dir') }}">
<head><title>{{ t('checkout') }}</title>""" + STYLE_GLOBAL + """</head>
<body dir="{{ t('dir') }}">
    <div class="navbar"><div class="logo">💳 {{ t('checkout') }}</div><div class="nav-links"><a href="/store">Back</a></div></div>
    
    <div class="main-content" style="max-width:600px;">
        <div class="card">
            <h2 style="border-bottom:1px solid #333; padding-bottom:10px;">{{ t('payment') }}</h2>
            <div style="background:rgba(0, 242, 255, 0.1); padding:15px; border-radius:5px; margin-bottom:20px;">
                <h3 style="margin:0;">Item: {{ item }}</h3>
                <h2 style="margin:0; color:var(--primary);">${{ price }}</h2>
            </div>

            <form action="/process_payment" method="POST">
                <h3>{{ t('shipping') }}</h3>
                <input type="text" name="name" placeholder="{{ t('full_name') }}" required>
                <input type="text" name="address" placeholder="{{ t('address') }}" required>
                
                <h3>{{ t('payment') }}</h3>
                <div style="display:flex; gap:10px;">
                    <img src="https://cdn-icons-png.flaticon.com/512/196/196578.png" width="40">
                    <img src="https://cdn-icons-png.flaticon.com/512/196/196566.png" width="40">
                </div>
                <input type="text" name="card_num" placeholder="{{ t('card_number') }} (16 digits)" maxlength="16" required>
                <div style="display:flex; gap:10px;">
                    <input type="text" name="expiry" placeholder="MM/YY" maxlength="5" required>
                    <input type="text" name="cvv" placeholder="CVV" maxlength="3" required>
                </div>
                
                <button class="btn" style="font-size:18px;">{{ t('pay_btn') }}</button>
            </form>
        </div>
    </div>
</body>
</html>
"""

HTML_SUCCESS = """
<!DOCTYPE html>
<html dir="{{ t('dir') }}">
<head><title>Success</title>""" + STYLE_GLOBAL + """</head>
<body style="display:flex; justify-content:center; align-items:center; height:100vh; text-align:center;">
    <div class="card" style="padding:50px;">
        <div style="font-size:60px;">🎉</div>
        <h1 style="color:var(--primary);">Payment Successful!</h1>
        <p style="font-size:18px; margin:20px 0;">{{ t('success_msg') }}</p>
        <a href="/dashboard" class="btn">Return to Dashboard</a>
    </div>
</body>
</html>
"""

HTML_ERROR = """
<!DOCTYPE html>
<html dir="{{ t('dir') }}">
<head><title>Error</title>""" + STYLE_GLOBAL + """</head>
<body style="display:flex; justify-content:center; align-items:center; height:100vh; text-align:center;">
    <div class="card" style="padding:50px; border-color:red;">
        <div style="font-size:60px;">🚫</div>
        <h1 style="color:red;">Payment Failed</h1>
        <p style="font-size:18px; margin:20px 0;">{{ error }}</p>
        <a href="/store" class="btn" style="background:#333;">Try Again</a>
    </div>
</body>
</html>
"""

# ==========================================
# 4. التوجيه والمنطق (Routes & Logic)
# ==========================================

@app.context_processor
def inject_user():
    return dict(t=get_text)

@app.route('/set_lang/<lang>')
def set_lang(lang):
    if lang in TRANSLATIONS:
        session['lang'] = lang
    return redirect(request.referrer or '/')

@app.route('/')
def index():
    if 'user' in session: return redirect(url_for('dashboard'))
    return render_template_string(HTML_LOGIN)

@app.route('/login_guest', methods=['POST'])
def login_guest():
    session['user'] = f"Guest {request.form.get('guest_name')}"
    return redirect(url_for('dashboard'))

@app.route('/login_admin', methods=['POST'])
def login_admin():
    if request.form['username'] == 'admin' and request.form['password'] == '123':
        session['user'] = "Commander Admin"
        return redirect(url_for('dashboard'))
    return "Invalid Password"

@app.route('/login_google_sim')
def login_google_sim(): return render_template_string(HTML_GOOGLE_LOGIN)

@app.route('/auth_google', methods=['POST'])
def auth_google():
    email = request.form['email']
    session['user'] = email  
    return redirect(url_for('dashboard'))

@app.route('/login_apple_sim')
def login_apple_sim(): return render_template_string(HTML_APPLE_LOGIN)

@app.route('/auth_apple', methods=['POST'])
def auth_apple():
    email = request.form['email']
    session['user'] = "Apple User (" + email.split('@')[0] + ")"
    return redirect(url_for('dashboard'))

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'user' not in session: return redirect(url_for('index'))
    return render_template_string(HTML_DASHBOARD, user=session['user'])

@app.route('/store')
def store():
    if 'user' not in session: return redirect(url_for('index'))
    return render_template_string(HTML_STORE)

@app.route('/support')
def support():
    if 'user' not in session: return redirect(url_for('index'))
    return render_template_string(HTML_SUPPORT, user=session['user'])

@app.route('/checkout')
def checkout():
    if 'user' not in session: return redirect(url_for('index'))
    item = request.args.get('item', 'Robot')
    price = request.args.get('price', '0')
    return render_template_string(HTML_CHECKOUT, item=item, price=price)

@app.route('/process_payment', methods=['POST'])
def process_payment():
    expiry = request.form['expiry'] # MM/YY
    try:
        exp_date = datetime.strptime(expiry, "%m/%y")
        current_date = datetime.now()
        if exp_date.year < current_date.year or (exp_date.year == current_date.year and exp_date.month < current_date.month):
            return render_template_string(HTML_ERROR, error=get_text('invalid_card'))
        return render_template_string(HTML_SUCCESS)
    except:
        return render_template_string(HTML_ERROR, error="Invalid Date Format (Use MM/YY)")

@app.route('/video_feed')
def video_feed():
    if 'user' not in session: return redirect(url_for('index'))
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def api_status():
    t = time.time()
    offset = math.sin(t * 0.5) * 0.0005 
    robot_status['lng'] = start_lng + offset
    robot_status['lat'] = start_lat + (math.cos(t * 0.3) * 0.0001)
    return jsonify(robot_status)

@app.route('/api/summon', methods=['POST'])
def api_summon():
    robot_status['status'] = "Moving to Owner..."
    return jsonify({"msg": "Robot is moving to your GPS coordinates."})

# -----------------
# Chatbot Logic
# -----------------
@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.json
    msg = data.get('msg', '').lower()
    lang = session.get('lang', 'en')
    
    # قاموس الردود الذكي (Smart Response Dictionary)
    responses = {
        'en': {
            'default': "I'm not sure how to answer that yet.",
            'hello': "Hello! Ready to clean some rivers?",
            'battery': f"Current battery level is {robot_status['battery']}% and stable.",
            'trash': f"I have collected {robot_status['trash_count']} trash items so far.",
            'location': "I am currently monitoring the Nile River sector.",
            'price': "Our robots start from $3,999. Check the store for more!",
            'work': "I use advanced YOLO AI to detect pollution and an automated arm to collect it."
        },
        'ar': {
            'default': "عذراً، لا أفهم هذا السؤال بعد.",
            'hello': "أهلاً بك! أنا جاهز لتنظيف الأنهار.",
            'battery': f"مستوى البطارية الحالي {robot_status['battery']}% ومستقر.",
            'trash': f"لقد جمعت {robot_status['trash_count']} قطعة من النفايات حتى الآن.",
            'location': "أنا حالياً أراقب قطاع نهر النيل.",
            'price': "أسعار الروبوتات تبدأ من 3,999 دولار. تفقد المتجر للمزيد!",
            'work': "أستخدم ذكاءً اصطناعياً متطوراً (YOLO) لاكتشاف التلوث وذراعاً آلياً لجمعه."
        },
        'fr': {
            'default': "Je ne suis pas sûr de comprendre.",
            'hello': "Bonjour! Prêt à nettoyer les rivières?",
            'battery': f"Le niveau de batterie est de {robot_status['battery']}% stable.",
            'trash': f"J'ai collecté {robot_status['trash_count']} déchets jusqu'à présent.",
            'location': "Je surveille actuellement le secteur du Nil.",
            'price': "Nos robots commencent à partir de 3 999 $. Vérifiez le magasin!",
            'work': "J'utilise une IA avancée pour détecter la pollution et un bras automatisé pour la collecter."
        }
    }
    
    # تحديد الكلمات المفتاحية
    reply = responses[lang]['default']
    if any(x in msg for x in ['hello', 'hi', 'مرحبا', 'اهلا', 'bonjour']):
        reply = responses[lang]['hello']
    elif any(x in msg for x in ['battery', 'charge', 'بطارية', 'شحن', 'batterie']):
        reply = responses[lang]['battery']
    elif any(x in msg for x in ['trash', 'garbage', 'waste', 'قمامة', 'زبالة', 'déchets']):
        reply = responses[lang]['trash']
    elif any(x in msg for x in ['location', 'where', 'موقع', 'اين', 'emplacement']):
        reply = responses[lang]['location']
    elif any(x in msg for x in ['price', 'cost', 'buy', 'سعر', 'شراء', 'prix']):
        reply = responses[lang]['price']
    elif any(x in msg for x in ['work', 'how', 'كيف', 'عمل', 'comment']):
        reply = responses[lang]['work']

    return jsonify({'reply': reply})

if __name__ == '__main__':
    try:
        host_name = socket.gethostname()
        local_ip = socket.gethostbyname(host_name)
    except:
        local_ip = "127.0.0.1"

    print("\n" + "="*50)
    print(f"🚀 AQUA-R SERVER STARTED!")
    print(f"🌍 Open this link in Google Chrome: http://{local_ip}:5000")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)