from django.shortcuts import render
from django.shortcuts import redirect
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.utils.decorators import method_decorator

from django.contrib import auth
from django.contrib.auth.models import User

from django.db import IntegrityError
from django.core.exceptions import ValidationError
from django.contrib.auth.password_validation import validate_password
import logging

logger = logging.getLogger(__name__)
from django.contrib.auth import get_user_model

import os
from openai import OpenAI
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import base64
import json
import csv
from dotenv import load_dotenv

from .models import Chat
from django.utils import timezone  

from bonechat.models import Chat
from django.contrib.auth.decorators import login_required

from PIL import Image

# ===== Load ENV & Configure =====
load_dotenv()
API = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=API)

# สร้างโมเดล (ยังมี system_instruction เหมือนเดิม)
llm = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    generation_config=GenerationConfig(
        temperature=0.7,
        top_p=1,
        top_k=1,
    ),
    system_instruction="You are a doctor. Your name is Bone Analysis AI."
)

# ====== Chat Memory Helpers ======
MAX_TURNS = 12  # เก็บล่าสุด 12 เทิร์น (24 ข้อความ: user+model)

def _get_turns(session):
    """
    ดึง history ที่เราเก็บใน session: [{"role":"user|model","content":"..."}]
    """
    return session.get("chat_history", [])

def _save_turns(session, turns):
    """
    เซฟเฉพาะเทิร์นล่าสุดไม่เกิน MAX_TURNS
    """
    trimmed = turns[-MAX_TURNS*2:]  # user+model = *2
    session["chat_history"] = trimmed
    session.modified = True

def _to_gemini_history(turns):
    """
    แปลงเป็นฟอร์แมตที่ Gemini ต้องการ: [{"role":"user","parts":[...]}]
    """
    out = []
    for t in turns:
        out.append({"role": t["role"], "parts": [t["content"]]})
    return out

def _append_turn(session, role, content):
    turns = _get_turns(session)
    turns.append({"role": role, "content": content})
    _save_turns(session, turns)

def _reset_chat(session):
    session["chat_history"] = []
    session.modified = True

# ====== Core LLM Call with Memory ======
def call_gemini_with_memory(session, user_prompt: str) -> str:
    """
    สร้าง chat session ใหม่ทุกครั้ง โดยใส่ history จาก session เข้าไป
    ส่งข้อความล่าสุด แล้วบันทึกคำตอบกลับเข้า session
    """
    try:
        turns = _get_turns(session)
        chat = llm.start_chat(history=_to_gemini_history(turns))
        resp = chat.send_message(user_prompt)

        # เก็บประวัติ (เราเก็บแบบบาง ๆ ของตัวเอง เพื่อให้ serializable แน่ ๆ)
        _append_turn(session, "user", user_prompt)
        _append_turn(session, "model", resp.text)

        return resp.text
    except Exception as e:
        return f"Sorry, I encountered an error while processing your request. ({e})"

# ====== Django View ======
@require_http_methods(["GET", "POST"])
def chatbot(request):
    #ทำให้เวลาเราดูข้อมูลที่เราเคย Chat ตงกับ User
    chats = Chat.objects.filter(user=request.user)
    # reset history ถ้าผู้ใช้ส่งคำสั่งพิเศษ
    if request.method == "POST" and request.POST.get("reset") == "1":
        _reset_chat(request.session)
        return JsonResponse({"ok": True, "message": "Chat history has been reset."})

    if request.method == 'POST':
        message = request.POST.get('message')
        if not message:
            return JsonResponse({'error': 'Empty message'}, status=400)

        # ตัวอย่าง: รองรับพิมพ์ "/reset"
        if message.strip().lower() == "/reset":
            _reset_chat(request.session)
            return JsonResponse({"ok": True, "message": "Chat history has been reset."})

        response = call_gemini_with_memory(request.session, message)

        chat = Chat(user=request.user, message=message, response=response, created_at=timezone.now())
        chat.save()
        return JsonResponse({'message': message, 'response': response})

    # GET -> render UI
    return render(request, 'chatbot.html')


def login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = auth.authenticate(request, username=username, password=password)
        if user is not None:
            auth.login(request, user)
            return redirect('chatbot')
        else:
            error_message = 'Invalid username or password'
            return render(request, 'login.html', {'error_message': error_message})
    else:
        return render(request, 'login.html')

def register(request):
    if request.method == 'POST':
        username  = request.POST.get('username','').strip()
        email     = request.POST.get('email','').strip()
        password1 = request.POST.get('password1','')
        password2 = request.POST.get('password2','')

        if password1 != password2:
            return render(request, 'register.html', {'error_message': 'Passwords do not match'})

        User = get_user_model()
        if User.objects.filter(username=username).exists():
            return render(request, 'register.html', {'error_message': 'Username is already taken'})

        try:
            validate_password(password1)   # ถ้ามีตัวตรวจความแข็งแรงของรหัสผ่าน จะได้ข้อความที่ชัดเจน
        except ValidationError as e:
            return render(request, 'register.html', {'error_message': ' '.join(e.messages)})

        try:
            user = User.objects.create_user(username=username, email=email, password=password1)
        except IntegrityError as e:
            logger.exception("Integrity error")
            return render(request, 'register.html', {'error_message': f'Integrity error: {e}'})
        except Exception as e:
            logger.exception("Create user failed")
            return render(request, 'register.html', {'error_message': f'Create user failed: {e}'})

        try:
            auth.login(request, user)
        except Exception as e:
            logger.exception("Login failed after creating user")
            return render(request, 'register.html', {'error_message': f'Account created but login failed: {e}'})

        return redirect('chatbot')  # ชื่อเส้นทางของคุณถูกแล้ว
    return render(request, 'register.html')

def logout(request):
    auth.logout(request)
    return redirect('chatbot')

def folder(request):
    return render(request, 'folder.html')

def histories(request):
    qs = Chat.objects.select_related('user').order_by('-created_at')
    # ถ้าไม่ใช่ superuser ให้เห็นเฉพาะของตัวเอง
    if not request.user.is_superuser:
        qs = qs.filter(user=request.user)
    return render(request, 'histories.html', {'chats': qs})

