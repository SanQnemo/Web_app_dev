from django.shortcuts import render
from django.shortcuts import redirect
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_protect
from django.utils.decorators import method_decorator

from django.contrib import auth
from django.contrib.auth.models import User

from django.db import IntegrityError
from django.core.exceptions import ValidationError
from django.contrib.auth.password_validation import validate_password
import logging

logger = logging.getLogger(__name__)
from django.contrib.auth import get_user_model
from .llms import generate_reply

import os

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
    """Start a fresh chat from session history and get a reply."""
    try:
        # Convert our history to Gemini history format
        turns = _get_turns(session)
        gem_history = []
        for t in turns:
            role = t.get('role')
            txt  = t.get('text','')
            if role in ('user','model') and txt:
                gem_history.append({'role': role, 'parts': [{'text': txt}]})
        # Generate
        reply = generate_reply(gem_history, user_prompt)
        # Save back
        _append_turn(session, 'user', user_prompt)
        _append_turn(session, 'model', reply)
        return reply
    except Exception as e:
        logger.exception("LLM call failed")
        return "⚠️ LLM error: {}".format(e)

        return f"Sorry, I encountered an error while processing your request. ({e})"

# ====== Django View ======
@require_http_methods(["GET", "POST"])
@csrf_protect
def chatbot(request):
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
        return JsonResponse({'message': message, 'response': response})

    # GET -> render UI
    return render(request, 'chatbot.html')


def login(request):
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
